import sys
import numpy as np
import argparse
import wandb


from tqdm import tqdm
import os
import logging
from TextRetrosynthesis.text_datasets import USPTOText50k
from TextRetrosynthesis.tasks import *
from TextRetrosynthesis.core import EngineText
import torch
from torchdrug import core, models, tasks
from torch.utils import data as torch_data
from transformers.optimization import get_linear_schedule_with_warmup
import json
import csv
def get_reaction_task(args):
    if args['reaction_arch'] == 'CenterIdentificationText':
        return CenterIdentificationText
    elif args['reaction_arch'] == 'CenterIdentificationTruncate':
        return CenterIdentificationTruncate
    else:
        raise ValueError(f"Unknown reaction task {args['reaction']}")

def main(args):

    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
    handler = logging.FileHandler('{}/train.log'.format(args['save_dir']))
    handler.setFormatter(format)
    logger = logging.getLogger("")
    logger.addHandler(handler)

    if args['debug']:
        data_name = f'{args["data_name"]}_debug.csv'
    else:
        data_name = f'{args["data_name"]}_train.csv'

    valid_data_name = f'{args["data_name"]}_valid.csv'
    test_data_name = f'{args["data_name"]}_test.csv'
    torch.manual_seed(args['fold'])
    reaction_dataset = USPTOText50k(f"{args['data_dir']}/",
                                             file_name=data_name,
                                             atom_feature="center_identification", as_synthon=False,
                                             with_hydrogen=False, kekulize=True, verbose=1)
    reaction_train, _, _ = reaction_dataset.split(ratios=[1, 0, 0])
    reaction_dataset_valid = USPTOText50k(f"{args['data_dir']}/",
                                file_name=valid_data_name,
                                atom_feature="center_identification", as_synthon=False,
                                with_hydrogen=False, kekulize=True, verbose=1)
    _, reaction_valid, _ = reaction_dataset_valid.split(ratios=[0, 1, 0])
    reaction_dataset_test = USPTOText50k(f"{args['data_dir']}/",
                                file_name=test_data_name,
                                atom_feature="center_identification", as_synthon=False, 
                                with_hydrogen=False, kekulize=True, verbose=1)
    _, _, reaction_test = reaction_dataset_test.split(ratios=[0, 0, 1])
    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[256, 256, 256, 256, 256, 256],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)
    
    reaction_arch = get_reaction_task(args)
    #temp code#
    reaction_task = reaction_arch(reaction_model,
                                             feature=("graph", "atom", "bond"),
                                             PLM=args['PLM'],
                                             PLM_d=args['PLM_d'],
                                             max_len=64)
    model_parameters = filter(lambda p: p.requires_grad, reaction_task.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Number of parameters: {params}")

    reaction_steps = len(reaction_train) // args['reaction_batch_size'] * args['reaction_epoch']
    reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=args['reaction_lr'])
    reaction_scheduler = get_linear_schedule_with_warmup(reaction_optimizer,
                                                        num_warmup_steps=args['reaction_warmups'],
                                                        num_training_steps=reaction_steps)
    reaction_solver = EngineText(reaction_task, reaction_train, reaction_valid,
                                reaction_test, reaction_optimizer, scheduler=reaction_scheduler, 
                                gpus=[args['device']],
                                batch_size=args['reaction_batch_size'],
                                log_interval=args['log_interval'])

    best_metric = 0
    reaction_metric = None
    for e in range(args['reaction_epoch']):
        reaction_solver.train(num_epoch=1)
        reaction_solver.save(f"{args['save_dir']}/reaction_last.pth")
        val_metrics = reaction_solver.evaluate("valid")
        if val_metrics[args['best_metric_reaction']] > best_metric:
            best_metric = val_metrics[args['best_metric_reaction']]
            reaction_metric = val_metrics
            reaction_solver.save(f"{args['save_dir']}/reaction_best.pth")
        wandb.log({"reaction best metrics": best_metric})
        wandb.log({f"{args['best_metric_reaction']}": val_metrics[args['best_metric_reaction']]})
        wandb.log({"reaction lr": reaction_scheduler.get_last_lr()[0]})
        logger.info(f'Epoch {e} best metrics: {best_metric} learning rate: {reaction_scheduler.get_last_lr()[0]}')

    f = open(os.path.join(args['save_dir'], 'retrosynthesis_valid_result.txt'), 'w')
    f.write('====================\n')
    f.write(f'reaction model result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in reaction_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.write(f'synthon model result for fold {args["fold"]}\n')
    f.write('====================\n')
    f.close()

    # synthon model
    with open(f"{args['data_dir']}/" + valid_data_name, "r") as f:
        with open(f"{args['data_dir']}/" + test_data_name, "r") as f2:
            with open(f"{args['data_dir']}/" + data_name, "r") as f3:
                with open(f"{args['data_dir']}/tmp.csv", "w") as g:
                    vld_dataset = list(csv.reader(f))
                    tst_dataset = list(csv.reader(f2))[1:]
                    trn_dataset = list(csv.reader(f3))[1:]
                    len_vld = len(vld_dataset) - 1
                    len_tst = len(tst_dataset)
                    len_trn = len(trn_dataset)
                    datasets = vld_dataset + tst_dataset + trn_dataset
                    writer = csv.writer(g)
                    for data in tqdm(datasets, desc="preprocess"):
                        writer.writerow(data)

    synthon_dataset = USPTOText50k(f"{args['data_dir']}/",
                                file_name="tmp.csv",
                                atom_feature="synthon_completion", as_synthon=True,
                                with_hydrogen=False, kekulize=True, verbose=1)
    synthon_valid, synthon_test, synthon_train = synthon_dataset.split_new(ratios=None, nums=[len_vld, len_tst, len_trn])
    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256, 256, 256],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))
    # synthon_task = tasks.SynthonCompletion(synthon_model)
    synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=args['synthon_lr'])

    synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                                synthon_test, synthon_optimizer, gpus=[args['device']],
                                batch_size=args['synthon_batch_size'],
                                log_interval=args['log_interval'])
    best_metric = 0
    synthon_metric = None
    for e in range(args['synthon_epoch']):
        synthon_solver.train(num_epoch=1)
        try:
            # raise NotImplementedError
            val_metrics = synthon_solver.evaluate("valid")
        except:
            from IPython import embed; embed()
        if val_metrics[args['best_metric_synthon']] > best_metric:
            best_metric = val_metrics[args['best_metric_synthon']]
            synthon_metric = val_metrics
            synthon_solver.save(f"{args['save_dir']}/synthon_best.pth")
        wandb.log({"synthon best metrics": best_metric})
        wandb.log({f"{args['best_metric_synthon']}": val_metrics[args['best_metric_synthon']]})
        logger.info(f'Epoch {e} best metrics: {best_metric}')
    f = open(os.path.join(args['save_dir'], 'retrosynthesis_valid_result.txt'), 'w')
    f.write('====================\n')
    f.write(f'reaction model result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in reaction_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.write(f'synthon model result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in synthon_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.close()
    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                            num_synthon_beam=5, max_prediction=10)
    lengths = [len(reaction_valid) // 10,
            len(reaction_valid) - len(reaction_valid) // 10]
    reaction_valid_small = torch_data.random_split(reaction_valid, lengths)[0]
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = core.Engine(task, reaction_train, reaction_valid_small, reaction_test,
                        optimizer, batch_size=args['retrosynthesis_batch_size'], gpus=[args['device']])
    solver.load(f"{args['save_dir']}/reaction_best.pth", load_optimizer=False)
    solver.load(f"{args['save_dir']}/synthon_best.pth", load_optimizer=False)
    val_metrics = solver.evaluate("valid")

    f = open(os.path.join(args['save_dir'], 'retrosynthesis_valid_result.txt'), 'w')
    f.write('====================\n')
    f.write(f'reaction model result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in reaction_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.write(f'synthon model result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in synthon_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.write(f'retrosynthesis result for fold {args["fold"]}\n')
    f.write('====================\n')
    for key, value in val_metrics.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.write('====================\n')
    f.write('parametrs:\n')
    for k in args.keys():
        f.write(f'{k}: {args[k]}\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../ChemicalReaction/data')
    parser.add_argument('--data_name', type=str, default='data_ChatGPT')
    parser.add_argument('--PLM', type=str, default='prajjwal1/bert-small')
    parser.add_argument('--PLM_d', type=int, default=512)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--reaction_arch', type=str, default='CenterIdentificationTruncate')
    parser.add_argument('--reaction_epoch', type=int, default=50)
    parser.add_argument('--synthon_epoch', type=int, default=1)
    parser.add_argument('--reaction_warmups', type=int, default=20)
    parser.add_argument('--reaction_lr', type=float, default=1e-3)
    parser.add_argument('--synthon_lr', type=float, default=1e-4)
    parser.add_argument('--reaction_batch_size', type=int, default=8)
    parser.add_argument('--synthon_batch_size', type=int, default=8)
    parser.add_argument('--retrosynthesis_batch_size', type=int, default=5)
    parser.add_argument('--best_metric_reaction', type=str, default='accuracy')
    parser.add_argument('--best_metric_synthon', type=str, default='total accuracy')
    parser.add_argument('--save_dir', type=str, default='../chem_results/ChatGPT-check/')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--wandb_key', type=str, default="")
    if parser.parse_args().wandb_key:
        os.environ["WANDB_API_KEY"]= parser.parse_args().wandb_key
    args = parser.parse_args()
    args = args.__dict__

    wandb.init(project=f'TextRetrosynthesis_fold{args["fold"]}', config=args)

    main(args)




