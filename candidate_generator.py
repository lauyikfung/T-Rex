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
from torch.utils import data as torch_data
from transformers.optimization import get_linear_schedule_with_warmup
import json
import csv
from torch.nn import functional as F 
from torchdrug import core, models, tasks, utils, data
import pubchempy as pcp
def get_reaction_task(args):
    if args['reaction_arch'] == 'CenterIdentificationText':
        return CenterIdentificationText
    elif args['reaction_arch'] == 'CenterIdentificationTruncate':
        return CenterIdentificationTruncate
    else:
        raise ValueError(f"Unknown reaction task {args['reaction']}")

def smiles_to_name(smiles):
    try:
        compound_smiles = pcp.get_compounds(smiles, 'smiles')
        cpd_id = int(str(compound_smiles[0]).split("(")[-1][:-1])
        c = pcp.Compound.from_cid(cpd_id)
        if isinstance(c.iupac_name, str):
            return c.iupac_name
        else:
            return "None"
    except:
        return "None"
    
candidate_num = {"train": 3, "valid": 10, "test": 10}

def main(args):
    if not os.path.exists(args['save_dir']):
      os.makedirs(args['save_dir'])
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
    handler = logging.FileHandler('{}/train.log'.format(args['save_dir']))
    handler.setFormatter(format)
    logger = logging.getLogger("")
    logger.addHandler(handler)
    for candidate_split in ["train", "valid", "test"]:
        if args['debug']:
            data_name = f'{args["data_name"]}_debug.csv'
        else:
            data_name = f'{args["data_name"]}_{candidate_split}.csv'
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
        reaction_solver.load(f"{args['save_dir']}/reaction_best.pth")
        reaction_task = reaction_solver.model

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

        task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                                num_synthon_beam=5, max_prediction=10)
        lengths = [len(reaction_test) // 10,
                len(reaction_test) - len(reaction_test) // 10]
        reaction_test_small = torch_data.random_split(reaction_test, lengths)[0]
        optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
        solver = core.Engine(task, reaction_train, reaction_valid, reaction_test_small,
                            optimizer, batch_size=args['retrosynthesis_batch_size'], gpus=[args['device']])
        solver.load(f"{args['save_dir']}/reaction_best.pth", load_optimizer=False)
        solver.load(f"{args['save_dir']}/synthon_best.pth", load_optimizer=False)
        bsz = args['generation_batch_size']

        K = candidate_num[candidate_split]
        with open(f"{args['data_dir']}/data_ChatGPT-fold{args['fold']}_candidate_{candidate_split}.csv", "w") as f2:
          writer = csv.writer(f2)
          tmp = 0
          writer.writerow(["id", "reaction", "rxn_smiles", "label", "sample id", "prod_molt5", "prod_smiles", "prod_IUPAC", "prod_ChatGPT"])
          with torch.no_grad():
            test_set = solver.train_set
            sampler = torch_data.DistributedSampler(test_set, solver.world_size, solver.rank)
            dataloader = data.DataLoader(test_set, bsz, sampler=sampler, num_workers=solver.num_worker)
            model = solver.model
            cnt = 0
            model.eval()
            data_reactant = []
            for batch in tqdm(dataloader):
                cnt += 1
                try:
                    text_batch = {}
                    for t_k in solver.train_set.dataset.text_fields:
                        text_batch[t_k] = batch[t_k]
                    if solver.device.type == "cuda":
                        batch = utils.cuda(batch, device=solver.device)
                    for k in text_batch.keys():
                        batch[k] = text_batch[k]
                    data_reactant = []
                    pred, target = model.predict_and_target(batch)
                    cum_sum = (torch.cumsum(pred[1], dim=0) - pred[1]).tolist()
                    data_reactant.extend([[batch['reaction'][j].item(), 
                                pred[0][i + cum_sum[j]].to_smiles(isomeric=False, atom_map=False, canonical=True), 
                    pred[0][i + cum_sum[j]].to_smiles(isomeric=False, atom_map=False, canonical=True) == target[j].to_smiles(isomeric=False, atom_map=False, canonical=True),
                    batch['sample id'][j].item(), batch['MolT5'][j], batch['graph'][1][j].to_smiles(isomeric=False, atom_map=False, canonical=True),
                     smiles_to_name(batch['graph'][1][j].to_smiles(isomeric=False, atom_map=False, canonical=True)), batch['ChatGPT'][j]] for j in range(len(batch['reaction'])) for i in range(K)])
                    if candidate_split == "train":
                      data_reactant.extend([[batch['reaction'][j].item(),
                                            target[j].to_smiles(isomeric=False, atom_map=False, canonical=True), 
                                            True, batch['sample id'][j].item(), batch['MolT5'][j], batch['graph'][1][j].to_smiles(isomeric=False, atom_map=False, canonical=True),
                      smiles_to_name(batch['graph'][1][j].to_smiles(isomeric=False, atom_map=False, canonical=True)), batch['ChatGPT'][j]] for j in range(len(batch['reaction']))])

                    for i in range(len(data_reactant)):
                        writer.writerow([tmp] + data_reactant[i])
                        tmp += 1
                    data_reactant = []
                    f2.flush()
                except:
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/candidate_generation')
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
    parser.add_argument('--synthon_lr', type=float, default=1e-3)
    parser.add_argument('--reaction_batch_size', type=int, default=8)
    parser.add_argument('--synthon_batch_size', type=int, default=8)
    parser.add_argument('--retrosynthesis_batch_size', type=int, default=5)
    parser.add_argument('--best_metric_reaction', type=str, default='accuracy')
    parser.add_argument('--best_metric_synthon', type=str, default='total accuracy')
    parser.add_argument('--save_dir', type=str, default='../chem_results/ChatGPT-check/')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--wandb_key', type=str, default="")
    parser.add_argument('--generation_batch_size', type=int, default=32)

    if parser.parse_args().wandb_key:
        os.environ["WANDB_API_KEY"]= parser.parse_args().wandb_key
    args = parser.parse_args()
    args = args.__dict__

    wandb.init(project=f'TextRetrosynthesis_fold{args["fold"]}', config=args)

    main(args)




