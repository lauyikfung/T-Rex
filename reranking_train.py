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
from TextRetrosynthesis.classification_gpt.classification_gpt import ReactionClassification_GPT
def main(args):
    wandb.init(project=args['project_name'], name=args['exp_name'])

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

    logger = pl.loggers.WandbLogger(project=args['project_name'], name=args['exp_name'])
    
    logger.watch(model, log='all', log_freq=500)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid/macro_f1',
        dirpath='logs-ckpt',
        filename=args['save_dir']+'-{epoch:02d}',
        save_top_k=1,
        mode='max',
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        devices=1, 
        accelerator='gpu',
        max_epochs=args['max_epochs'],
        logger=logger,
        callbacks=[
            checkpoint_callback, 
            lr_monitor_callback
        ],
        log_every_n_steps=10,
        val_check_interval=5000,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/candidate_generation/')
    parser.add_argument('--PLM', type=str, default='microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL')
    parser.add_argument('--project_name', type=str, default='rreranking')
    parser.add_argument('--exp_name', type=str, default='reranking')
    parser.add_argument('--save_dir', type=str, default='reranking')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--graph_model_dir', type=str, default='pretrained_model/graph_model')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--wandb_key', type=str, default='')
    args = parser.parse_args()
    args = args.__dict__

    import os
    os.environ["WANDB_API_KEY"]=args['wandb_key']
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])
    main(args)