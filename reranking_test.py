import sys
import numpy as np
import argparse
import datetime
import wandb

import os
import logging
import torch
import json
import pytorch_lightning as pl
import wandb
import torch
import csv
from torchdrug import data as tdrug_data
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer
from TextRetrosynthesis.classification_gpt.classification_gpt import ReactionClassification_GPT
def main(args):

    model = ReactionClassification_GPT(
        data_root=args['data_dir'],
        bert_name=args['PLM'],
        num_labels=args['num_labels'],
        dropout_rate=args['dropout'],
        lr=args['lr'],
        batch_size=args['batch_size'],
        max_epochs=args['max_epochs'],
        num_workers=args['num_workers'],
        alpha=args['alpha'],
        beta=0.0,
        graph_model_dir=args['graph_model_dir'],
        fold=args['fold'],
    )
    model.load_all(args['checkpoint'])
    model.eval()
    device = "cuda"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.bert_name)

    with open(f"{args['data_dir']}data_ChatGPT-reactant-fold{args['fold']}_test_processed_new.csv", "r") as f:
        reader = csv.reader(f)
        data_reactant = list(reader)[1:]
    device = "cuda"
    model.to(device)

    bsz = 10
    topns = [3, 5, 10]
    metrics = [[1], [3], [5]]
    results = [[0], [0], [0]]

    for idx in range(len(topns)):
      topn = topns[idx]
      right_number = torch.tensor([0 for _ in range(10)])[:topn]
      wrong_number = torch.tensor([0 for _ in range(10)])[:topn]
      for i in tqdm(range(len(data_reactant) // bsz)):
          small_batch = data_reactant[i * bsz:i * bsz + bsz][:topn]
          smiles_list = [row[2] for row in small_batch]
          
          text_list = [row[-1] for row in small_batch]
          label_list = [(row[3] == "True") + 0 for row in small_batch]
          label_list = torch.tensor(label_list)
          label_tensor = torch.tensor(label_list).to(device)
          text_input = tokenizer.batch_encode_plus(list(text_list), padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
          mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
          mol_list = tdrug_data.PackedMolecule.from_molecule(mol_list, 
                          atom_feature="center_identification", 
                          bond_feature="default", 
                          with_hydrogen=False, kekulize=True).to(device)
          pred = torch.softmax(model(text_input, mol_list), dim=1)
          right_number += torch.tensor(label_list)[pred[:,1].sort(descending=True).indices]
          wrong_number += 1 - torch.tensor(label_list)[pred[:,0].sort(descending=True).indices]

      results[idx] = [right_number[:metric].sum().item() / (len(data_reactant) // bsz) for metric in metrics[idx]]
    for idx in range(len(topns)):
      for metric in metrics[idx]:
        print(f"Top {metric} accuracy in {topns[idx]}: {results[idx][metrics[idx].index(metric)]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/candidate_generation/')
    parser.add_argument('--PLM', type=str, default='microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--graph_model_dir', type=str, default='pretrained_GCN/reaction_model_fold1.pth')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    args = args.__dict__

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])
    main(args)