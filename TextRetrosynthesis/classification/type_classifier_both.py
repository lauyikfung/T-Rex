import sys
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
import csv
from tqdm import tqdm
from rdkit import Chem

module = sys.modules[__name__]
logger = logging.getLogger(__name__)
class TypeClassifierBoth(pl.LightningModule):
    def __init__(self, data_root, bert_name, num_labels = 10, 
                 dropout_rate=0.1, lr=1e-5, batch_size=128, 
                 max_epochs=1000, num_workers=4, bert_layer_train_num = 1, 
                 graph_model_dir = None, use_text = True):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = self.bert.config.hidden_size
        self.graph_model_dir = graph_model_dir
        self.use_text = use_text
        if self.graph_model_dir is not None:
            self.has_load = False
            self.hidden_size += 1536 * 2 # self.graph_model.output_dim
        if not self.use_text:
            self.hidden_size = 1536 * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels),
        )
        self.cst_mlp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.num_labels = num_labels
        self.data = {"train": TypeClassifierBothDataset(data_root, 'train'), 
                     "valid": TypeClassifierBothDataset(data_root, 'valid'), 
                     "test": TypeClassifierBothDataset(data_root, 'test')}
        self.bert_layer_train_num = bert_layer_train_num

        self.update_require_grad_state()

    def update_require_grad_state(self):
        if self.bert_layer_train_num is None:
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = False
            
            for i in range(self.bert_layer_train_num):
                for param in self.bert.encoder.layer[-1-i].parameters():
                    param.requires_grad = True

    def forward(self, text1_inputs, mol_list, rs_list):
        if self.graph_model_dir is not None and not self.has_load:
              self.load_graph_model()
        if self.use_text:
            bert1_output = self.bert(**text1_inputs)
            seq1_embedding = bert1_output.pooler_output
            if self.graph_model_dir is not None:
                graph_emb = self.graph_model(mol_list, mol_list.node_feature.float(), None, None)['graph_feature']
                graph_emb2 = self.graph_model(rs_list, rs_list.node_feature.float(), None, None)['graph_feature']
                seq1_embedding = torch.cat([seq1_embedding, graph_emb, graph_emb2], dim=-1)
            seq1_embedding = self.dropout(seq1_embedding)
            logits = self.classifier(seq1_embedding)
        else:
            graph_emb = self.graph_model(mol_list, mol_list.node_feature.float(), None, None)['graph_feature']
            graph_emb2 = self.graph_model(rs_list, rs_list.node_feature.float(), None, None)['graph_feature']
            graph_emb = torch.cat([graph_emb, graph_emb2], dim=-1)
            logits = self.classifier(graph_emb)
        return logits

    def load_graph_model(self):
        from torchdrug import models as tdrug_models
        self.tdrug_models = tdrug_models
        self.graph_model = self.tdrug_models.RGCN(input_dim=43,
            hidden_dims=[256, 256, 256, 256, 256, 256],
            num_relation=3,
            concat_hidden=True).to("cuda")
        self.graph_model.load_state_dict(torch.load(self.graph_model_dir, map_location=torch.device('cuda')))
        self.has_load = True


    def training_step(self, batch, batch_idx):
        text1_inputs, labels, mol_list, rs_list = batch
        logits = self(text1_inputs, mol_list, rs_list)
        loss = F.cross_entropy(logits, labels)
        self.log('train/loss', loss)
        self.log('train/accuracy', self._accuracy(logits.argmax(dim=-1), labels))
        self.log('train/macro_f1', self._macro_f1_score(logits.argmax(dim=-1), labels, num_labels=self.num_labels))
        return loss
    
    def validation_step(self, batch, batch_idx):
        text1_inputs, labels, mol_list, rs_list = batch
        logits = self(text1_inputs, mol_list, rs_list)
        self.log('valid/loss', F.cross_entropy(logits, labels))
        self.log('valid/accuracy', self._accuracy(logits.argmax(dim=-1), labels))
        self.log('valid/macro_f1', self._macro_f1_score(logits.argmax(dim=-1), labels, num_labels=self.num_labels))
    
    @staticmethod
    def _accuracy(pred, label):
        """pred is a 1D int tensor, label is a 1D int tensor"""
        return (pred == label).float().mean()
    
    @staticmethod
    def _macro_f1_score(pred, labels, num_labels):
        """pred is a 1D int tensor, label is a 1D int tensor"""
        pred_onehot = F.one_hot(pred, num_classes=num_labels)
        label_onehot = F.one_hot(labels, num_classes=num_labels)
        # print('pred_onehot', pred_onehot)
        # print('label_onehot', label_onehot)
        tp = (pred_onehot * label_onehot).sum(dim=0)
        fp = ((1 - label_onehot) * pred_onehot).sum(dim=0)
        fn = (label_onehot * (1 - pred_onehot)).sum(dim=0)
        # print('tp fp fn', tp, fp, fn)
        eps = 1e-8    # float32 precision
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        # print(f1)
        return f1.mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # slanted triangular scheduler (linear warmup followed by linear decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, 
            steps_per_epoch=len(self.train_dataloader()), epochs=self.hparams.max_epochs,
            pct_start=0.1, anneal_strategy='linear'
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data['train'],
            collate_fn=TypeClassifierBothCollateFn(tokenizer_name=self.hparams.bert_name),
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data['valid'],
            collate_fn=TypeClassifierBothCollateFn(tokenizer_name=self.hparams.bert_name),
            batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers,
        )
    
    def load_all(self, model_dir):
        import collections
        if not self.has_load:
            self.load_graph_model()
        state_dict = torch.load(model_dir)['state_dict']
        state_dicts = collections.OrderedDict()
        for k in state_dict.keys():
            if k.startswith("graph_model"):
                state_dicts[k.replace("graph_model.", "")] = state_dict[k]
        self.graph_model.load_state_dict(state_dicts)
        self.load_state_dict(state_dict, strict=False)

class TypeClassifierBothDataset(Dataset):
    def __init__(self, data_root, split_type):
        assert split_type in ['train', 'valid', 'test']
        # load "data/molecule-datasets/data_ChatGPT.csv"
        data_file = {"train": "data_ChatGPT_train.csv",
                     "valid": "data_ChatGPT_valid.csv",
                     "test": "data_ChatGPT_test.csv"}
        data_root = Path(data_root + data_file[split_type])

        self.data = self._load_data(data_root)

        self.data = [{'text': sample['text'], 'label': sample['label'], 'smiles': sample['smiles'], 'rs': sample['rs']} for sample in self.data]
            
    def _load_data(self, data_root):
        with open(data_root, "r") as f:
            datasets = list(csv.reader(f))[1:]
            data = []
            for row in tqdm(datasets):
                data.append({"text": row[-1], "label": int(row[2]) - 1, "smiles": row[4], "rs": row[5].split(">>")[0]})
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['text'], self.data[idx]['label'], self.data[idx]['smiles'], self.data[idx]['rs']

class TypeClassifierBothCollateFn:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = 512
        from torchdrug import data as tdrug_data
        self.tdrug_data = tdrug_data
        
    def __call__(self, batch):
        text_list, label_list, smiles_list, rs_list = zip(*batch)
        text_input = self.tokenizer.batch_encode_plus(list(text_list), padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
        label_input = torch.tensor(label_list).long()
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        rs_list = [Chem.MolFromSmiles(rs) for rs in rs_list]
        mol_list = self.tdrug_data.PackedMolecule.from_molecule(mol_list, 
                    atom_feature="center_identification", 
                    bond_feature="default", 
                    with_hydrogen=False, kekulize=True)
        rs_list = self.tdrug_data.PackedMolecule.from_molecule(rs_list,
                    atom_feature="center_identification",
                    bond_feature="default",
                    with_hydrogen=False, kekulize=True)
        return text_input, label_input, mol_list, rs_list
    