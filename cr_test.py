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
        data_name = f'{args["data_name"]}.csv'
    reaction_dataset = USPTOText50k(f"{args['data_dir']}/",
                                              file_name=data_name,
                                              atom_feature="center_identification", as_synthon=False,
                                              with_hydrogen=False, kekulize=True, verbose=1)

    torch.manual_seed(args['fold'])
    reaction_train, reaction_valid, reaction_test = reaction_dataset.split(ratios=[1, 0, 0])
    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[256, 256, 256, 256, 256, 256],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)

    reaction_arch = get_reaction_task(args)
    reaction_task = reaction_arch(reaction_model,
                                  feature=("graph", "atom", "bond"),# need to change
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
    reaction_solver.load(f"{args['save_dir']}/reaction_best.pth")
    reaction_task = reaction_solver.model
    if args['save_model_for_reranking']:
        reaction_solver.model.save(f"pretrained_GCN/pretrained_GCN/reaction_model_fold{args['fold']}.pth")
    reaction_metric = reaction_solver.evaluate("train")

    synthon_dataset = USPTOText50k(f"{args['data_dir']}/",
                                    file_name=data_name,
                                    atom_feature="synthon_completion", as_synthon=True,
                                    with_hydrogen=False, kekulize=True, verbose=1)
    synthon_train, synthon_valid, synthon_test = synthon_dataset.split(ratios=[1, 0, 0])
    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256, 256, 256],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))

    synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=args['synthon_lr'])
    synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                                synthon_test, synthon_optimizer, gpus=[args['device']],
                                batch_size=args['synthon_batch_size'],
                                log_interval=args['log_interval'])
    synthon_solver.load(f"{args['save_dir']}/synthon_best.pth")
    synthon_task = synthon_solver.model
    synthon_metric = synthon_solver.evaluate("train")
    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                            num_synthon_beam=5, max_prediction=10)
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = core.Engine(task, reaction_train, reaction_valid, reaction_test,
                        optimizer, batch_size=args['retrosynthesis_batch_size'], gpus=[args['device']])
    solver.load(f"{args['save_dir']}/reaction_best.pth", load_optimizer=False)
    solver.load(f"{args['save_dir']}/synthon_best.pth", load_optimizer=False)
    val_metrics = solver.evaluate("train")
    f = open(os.path.join(args['save_dir'], 'retrosynthesis_valid_result_test.txt'), 'w')

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
    parser.add_argument('--data_name', type=str, default='data_ChatGPT_test')
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
    parser.add_argument('--synthon_lr', type=float, default=1e-3)
    parser.add_argument('--reaction_batch_size', type=int, default=8)
    parser.add_argument('--synthon_batch_size', type=int, default=8)
    parser.add_argument('--retrosynthesis_batch_size', type=int, default=5)
    parser.add_argument('--best_metric_reaction', type=str, default='accuracy')
    parser.add_argument('--best_metric_synthon', type=str, default='total accuracy')
    parser.add_argument('--save_dir', type=str, default='../chem_results/ChatGPT-check/')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--save_model_for_reranking', type=bool, default=False)
    parser.add_argument('--wandb_key', type=str, default="")
    if parser.parse_args().wandb_key:
        os.environ["WANDB_API_KEY"]= parser.parse_args().wandb_key
    args = parser.parse_args()
    args = args.__dict__

    wandb.init(project=f'TextRetrosynthesis_fold{args["fold"]}', config=args)

    main(args)




