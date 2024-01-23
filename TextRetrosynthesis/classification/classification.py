import sys
import logging

from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
import csv
from tqdm import tqdm

module = sys.modules[__name__]
logger = logging.getLogger(__name__)
class ReactionClassification(pl.LightningModule):
    def __init__(self, data_root, bert_name, num_labels, 
                 num_rt = 0, dropout_rate=0.1, lr=1e-5, 
                 batch_size=128, max_epochs=1000, num_workers=4, 
                 bert_layer_train_num = 1, alpha = 0.0, beta = 0.0, 
                 graph_model_dir = None, fold=1):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_size = self.bert.config.hidden_size
        self.graph_model_dir = graph_model_dir

        if self.graph_model_dir is not None:
            self.has_load = False
            self.hidden_size += 1536

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels),
        )
        self.cst_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_labels = num_labels
        self.data = {"train": ClassificationDataset(data_root, 'train', fold=fold), 
                     "valid": ClassificationDataset(data_root, 'valid', fold=fold), 
                     "test": ClassificationDataset(data_root, 'test', fold=fold)}
        self.bert_layer_train_num = bert_layer_train_num
        self.alpha = alpha
        self.beta = beta
        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
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
    def forward(self, text1_inputs, mol_list, training = False):
        if self.graph_model_dir is not None and not self.has_load:
              self.load_graph_model()
        bert1_output = self.bert(**text1_inputs)
        seq1_embedding = bert1_output.pooler_output
        if self.graph_model_dir is not None:
            graph_emb = self.graph_model(mol_list, mol_list.node_feature.float(), None, None)['graph_feature']
            seq1_embedding = torch.cat([seq1_embedding, graph_emb], dim=-1)
        seq1_embedding = self.dropout(seq1_embedding)
        logits = self.classifier(seq1_embedding)
        if training:
            return logits, seq1_embedding
        else:
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
        text1_inputs, labels, mol_list = batch
        logits, text_emb = self(text1_inputs, mol_list, training = True)
        loss = F.cross_entropy(logits, labels)
        cst_loss = self.get_cosine_loss(text_emb, text_emb)
        pos_loss = self.get_pos_loss(text_emb, labels)
        loss = loss + self.alpha * cst_loss + self.beta * pos_loss
        self.log('train/loss', loss)
        self.log('train/accuracy', self._accuracy(logits.argmax(dim=-1), labels))
        self.log('train/accuracy_total', self._accuracy_total(logits.argmax(dim=-1), labels))
        self.log('train/macro_f1', self._macro_f1_score(logits.argmax(dim=-1), labels, num_labels=self.num_labels))
        return loss
    
    def validation_step(self, batch, batch_idx):
        text1_inputs, labels, mol_list = batch
        logits = self(text1_inputs, mol_list)
        self.log('valid/loss', F.cross_entropy(logits, labels))
        self.log('valid/accuracy', self._accuracy(logits.argmax(dim=-1), labels))
        self.log('train/accuracy_total', self._accuracy_total(logits.argmax(dim=-1), labels))
        self.log('valid/macro_f1', self._macro_f1_score(logits.argmax(dim=-1), labels, num_labels=self.num_labels))
    
    @staticmethod
    def _accuracy(pred, label):
        """pred is a 1D int tensor, label is a 1D int tensor"""
        # return (pred == label).float().mean()
        return ((pred == label) * label).float().mean()

    @staticmethod
    def _accuracy_total(pred, label):
        """pred is a 1D int tensor, label is a 1D int tensor"""
        return (pred == label).float().mean()
    
    @staticmethod
    def _macro_f1_score(pred, labels, num_labels):
        """pred is a 1D int tensor, label is a 1D int tensor"""
        pred_onehot = F.one_hot(pred, num_classes=num_labels)
        label_onehot = F.one_hot(labels, num_classes=num_labels)
        tp = (pred_onehot * label_onehot).sum(dim=0)
        fp = ((1 - label_onehot) * pred_onehot).sum(dim=0)
        fn = (label_onehot * (1 - pred_onehot)).sum(dim=0)
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1.mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, 
            steps_per_epoch=len(self.train_dataloader()), epochs=self.hparams.max_epochs,
            pct_start=0.1, anneal_strategy='linear'
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data['train'],
            collate_fn=CollateFn(tokenizer_name=self.hparams.bert_name),
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data['valid'],
            collate_fn=CollateFn(tokenizer_name=self.hparams.bert_name),
            batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers,
        )
    
    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss
    
    def get_pos_loss(self, logits, labels):
        # reward the positive pairs
        logits = torch.softmax(logits, dim=-1)
        logits = logits[:, 1]
        pos_loss = -torch.log(logits + 1e-8) * labels
        pos_loss = pos_loss.mean()
        return pos_loss
    
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

    def compute_top1_acc(self, data_reactant, tokenizer, device, bsz=3):
        right_number = 0
        wrong_number = 0
        from torchdrug import data as tdrug_data
        for i in tqdm(range(len(data_reactant) // bsz)):
            smiles_list = [row[2] for row in data_reactant[i * bsz:i * bsz + bsz]]
            text1_list = [row[5] for row in data_reactant[i * bsz:i * bsz + bsz]]
            text2_list = [row[6] for row in data_reactant[i * bsz:i * bsz + bsz]]
            text3_list = [row[7] if row[7] != "None" else row[6] for row in data_reactant[i * bsz:i * bsz + bsz]]
            text_list = [text1 + " [SEP] " + text2 + " [SEP] " + text3 for text1, text2, text3 in zip(text1_list, text2_list, text3_list)]
            label_list = [(row[3] == "True") + 0 for row in data_reactant[i * bsz:i * bsz + bsz]]
            label_tensor = torch.tensor(label_list).to(device)
            text_input = tokenizer.batch_encode_plus(list(text_list), padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            mol_list = tdrug_data.PackedMolecule.from_molecule(mol_list, 
                            atom_feature="center_identification", 
                            bond_feature="default", 
                            with_hydrogen=False, kekulize=True).to(device)
            pred = torch.softmax(self(text_input, mol_list), dim=1)
            right_number += label_list[torch.argmax(pred, dim=0)[1].item()]
            wrong_number += 1 - label_list[torch.argmax(pred, dim=0)[0].item()]
        return (right_number / (len(data_reactant) // bsz)), (wrong_number / (len(data_reactant) // bsz))
            

class ClassificationDataset(Dataset):
    def __init__(self, data_root, split_type, fold=1):
        assert split_type in ['train', 'valid', 'test']
        data_file = {"train": f"data_ChatGPT-reactant-fold{fold}_train_processed_new.csv",
                     "valid": f"data_ChatGPT-reactant-fold{fold}_valid_processed_new.csv",
                     "test": f"data_ChatGPT-reactant-fold{fold}_test_processed_new.csv"}
        data_root = Path(data_root + data_file[split_type])

        self.data = self._load_data(data_root)

        self.data = [{'text': sample['text'], 'label': sample['label'], 'smiles':sample['smiles']} for sample in self.data]
        print(f'Example: {self.data[0]}')
    
    def _load_data(self, data_root):
        with open(data_root, "r") as f:
            datasets = list(csv.reader(f))[1:]
            data = []
            print(len(datasets))
            for row in tqdm(datasets):
                if row[7] == "None":
                    data.append({"text": row[5] + " [SEP] " + row[6] + " [SEP] " + row[6], "label": (row[3] == "True") + 0, "smiles": row[2]})
                else:
                    data.append({"text": row[5] + " [SEP] " + row[6] + " [SEP] " + row[7], "label": (row[3] == "True") + 0, "smiles": row[2]})
                # data.append({"text": " [SEP] ", "label": (row[3] == "True") + 0, "smiles": row[2]})#temp
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['text'], self.data[idx]['label'], self.data[idx]['smiles']

class CollateFn:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = 512
        from torchdrug import data as tdrug_data
        self.tdrug_data = tdrug_data
        
    def __call__(self, batch):
        text_list, label_list, smiles_list = zip(*batch)
        text_input = self.tokenizer.batch_encode_plus(list(text_list), padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
        label_input = torch.tensor(label_list).long()
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        
        mol_list = self.tdrug_data.PackedMolecule.from_molecule(mol_list, 
                    atom_feature="center_identification", 
                    bond_feature="default", 
                    with_hydrogen=False, kekulize=True)
        return text_input, label_input, mol_list
