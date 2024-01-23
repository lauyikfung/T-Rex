import inspect
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_scatter import scatter_max, scatter_add
from torchdrug import core, tasks, data, metrics, transforms
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import layers
import logging
from typing import List
from transformers import BertTokenizer, AutoModel, AutoConfig


logger = logging.getLogger(__name__)


class CenterIdentificationTruncate(tasks.Task, core.Configurable):
    """
    Reaction center identification task.

    This class is a part of retrosynthesis prediction.

    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the product
            atom: original atom feature
            bond: original bond feature
        num_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = {"feature"}

    def __init__(self, model, feature=("reaction", "graph", "atom", "bond"), num_mlp_layer=2,
                 PLM='prajjwal1/bert-small',
                 #PLM='bert-base-uncased',
                 PLM_d=512,
                 max_len=512,
                 alpha=0.00, bert_layer_train_num = 1):
        super(CenterIdentificationTruncate, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.feature = feature
        self.max_len = max_len
        self.alpha = alpha
        self.tokenizer = BertTokenizer.from_pretrained(PLM)
        self.PLM = AutoModel.from_pretrained(PLM)
        self.PLM_d = PLM_d
        # self.text_fields = ['ChatGPT', 'MolT5']
        self.text_fields = ['ChatGPT']

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
        self.bert_layer_train_num = bert_layer_train_num
    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        bond_types = set()
        for sample in train_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                bond_types.update(graph.edge_list[:, 2].tolist())
        self.num_reaction = len(reaction_types)
        self.num_relation = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = train_set[0]["graph"][0].edge_feature.shape[-1]

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            elif _feature == "bond":
                edge_dim += edge_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)
        # add text dimension
        graph_dim += self.PLM_d
        
        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1])
        self.cst_mlp = layers.MLP(self.model.output_dim, self.PLM_d)

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        metric.update(self.evaluate(pred, target))

        target, size = target
        target = functional.variadic_max(target, size)[1]
        loss = functional.variadic_cross_entropy(pred, target, size)

        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        return loss, metric

    def _collate(self, edge_data, node_data, graph):
        new_data = torch.zeros(len(edge_data) + len(node_data), *edge_data.shape[1:],
                               dtype=edge_data.dtype, device=edge_data.device)
        num_cum_xs = graph.num_cum_edges + graph.num_cum_nodes
        num_xs = graph.num_edges + graph.num_nodes
        starts = num_cum_xs - num_xs
        ends = starts + graph.num_edges
        index = functional.multi_slice_mask(starts, ends, num_cum_xs[-1])
        new_data[index] = edge_data
        new_data[~index] = node_data
        return new_data

    def target(self, batch):
        reactant, product = batch["graph"]
        graph = product.directed()

        target = self._collate(graph.edge_label, graph.node_label, graph)
        size = graph.num_edges + graph.num_nodes
        return target, size

    def predict(self, batch, all_loss=None, metric=None):
        reactant, product = batch["graph"]
        output = self.model(product, product.node_feature.float(), all_loss, metric)

        text_emb = self.encode_text(batch)
        

        graph = product.directed()

        node_feature = [output["node_feature"]]
        edge_feature = []
        graph_feature = []

        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, batch["reaction"].unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature.float())
            elif _feature == "bond":
                edge_feature.append(graph.edge_feature.float())
            else:
                raise ValueError("Unknown feature `%s`" % _feature)
            
        graph_feature.append(text_emb)
        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)
        # inherit node features
        edge_feature.append(node_feature[graph.edge_list[:, :2]].flatten(1))
        edge_feature = torch.cat(edge_feature, dim=-1)

        edge_pred = self.edge_mlp(edge_feature).squeeze(-1)
        node_pred = self.node_mlp(node_feature).squeeze(-1)

        pred = self._collate(edge_pred, node_pred, graph)

        return pred

    def tokenize_and_cut(self, tokenizer, texts: List[str], max_length: int) -> List[List[str]]:
        tokenized_texts = []
        for text in texts:
            tokens = tokenizer.tokenize(text)
            token_chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
            tokenized_texts.append(token_chunks)
        return tokenized_texts

    def get_embeddings(self, model, text, tokenizer, device):
        embeddings = []

        input_ids = tokenizer.batch_encode_plus(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=self.max_len,
                        return_attention_mask=True,
                        return_tensors='pt')['input_ids']
        input_ids = input_ids.to(device)
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # embeddings = outputs.last_hidden_state.mean(dim=1)
        # embeddings = F.sigmoid(embeddings) # new added
        # embeddings = self.PLM_head(embeddings) # new added
        # embeddings = F.normalize(embeddings) # new added
        return embeddings

    def encode_text(self, batch):
        text = None 
        for k in self.text_fields:
            if text is None:
                text = batch[k]
            else:
                text = [t + ' ' + b for t, b in zip(text, batch[k])]
        embeddings = self.get_embeddings(self.PLM, \
                                         text, 
                                         self.tokenizer, self.device)
        return embeddings

    def variadic_top_precision(self, pred, target, size, k):
        index = functional.variadic_topk(pred, size, k, largest=True)[1]
        index = index.reshape([-1]) + (size.cumsum(0) - size).repeat_interleave(k)
        k_size = torch.tensor([k]*len(size)).to(size.device)
        precision = functional.variadic_sum(target[index], k_size)
        precision[size < k] = 0
        return torch.mean(precision.float())

    def evaluate(self, pred, target):
        target, size = target

        metric = {}
        target_vm = functional.variadic_max(target, size)[1]
        accuracy = metrics.variadic_accuracy(pred, target_vm, size).mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        for k in [1, 2, 3, 5]:
            p = self.variadic_top_precision(pred, target, size, k)
            name = tasks._get_metric_name(f"top:{k} precision")
            metric[name] = p

        return metric

    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss

    @ torch.no_grad()
    def predict_synthon(self, batch, k=1):
        """
        Predict top-k synthons from target molecules.

        Parameters:
            batch (dict): batch of target molecules
            k (int, optional): return top-k results

        Returns:
            list of dict: top k records.
                Each record is a batch dict of keys ``synthon``, ``num_synthon``, ``reaction_center``,
                ``log_likelihood`` and ``reaction``.
        """
        pred = self.predict(batch)
        target, size = self.target(batch)
        logp = functional.variadic_log_softmax(pred, size)

        reactant, product = batch["graph"]
        graph = product.directed()
        with graph.graph():
            graph.product_id = torch.arange(len(graph), device=self.device)

        graph = graph.repeat_interleave(k)
        reaction = batch["reaction"].repeat_interleave(k)
        with graph.graph():
            graph.split_id = torch.arange(k, device=self.device).repeat(len(graph) // k)

        logp, center_topk = functional.variadic_topk(logp, size, k)
        logp = logp.flatten()
        center_topk = center_topk.flatten()

        is_edge = center_topk < graph.num_edges
        node_index = center_topk + graph.num_cum_nodes - graph.num_nodes - graph.num_edges
        edge_index = center_topk + graph.num_cum_edges - graph.num_edges
        center_topk_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        center_topk[:-1]])
        product_id_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        graph.product_id[:-1]])
        is_duplicate = (center_topk == center_topk_shifted) & (graph.product_id == product_id_shifted)
        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)

        reaction_center = torch.zeros(len(graph), 2, dtype=torch.long, device=self.device)
        reaction_center[is_edge] = graph.atom_map[graph.edge_list[edge_index, :2]]
        reaction_center[~is_edge, 0] = graph.atom_map[node_index]

        # remove the edges from products
        graph = graph.edge_mask(edge_mask)
        graph = graph[~is_duplicate]
        reaction_center = reaction_center[~is_duplicate]
        logp = logp[~is_duplicate]
        reaction = reaction[~is_duplicate]
        synthon, num_synthon = graph.connected_components()
        synthon = synthon.undirected()  # (< num_graph * k)

        result = {
            "synthon": synthon,
            "num_synthon": num_synthon,
            "reaction_center": reaction_center,
            "log_likelihood": logp,
            "reaction": reaction,
        }

        return result
    
    @ torch.no_grad()
    def predict_synthon_3(self, batch, k=3):
        pred = self.predict(batch)
        target, size = self.target(batch)
        logp = functional.variadic_log_softmax(pred, size)
        reactant, product = batch["graph"]
        graph = product.directed()
        with graph.graph():
            graph.product_id = torch.arange(len(graph), device=self.device)
        graph = graph.repeat_interleave(k)
        reaction = batch["reaction"].repeat_interleave(k)
        with graph.graph():
            graph.split_id = torch.arange(k, device=self.device).repeat(len(graph) // k)
        logp, center_topk = functional.variadic_topk(logp, size, k)
        logp = logp.flatten()
        center_topk = center_topk.flatten()

        third_idx = torch.tensor([[1 + i * 3, 2 + i * 3, i * 3] for i in range(len(graph) // 3)], device=self.device)
        fourth_idx = torch.tensor([[2 + i * 3, i * 3, 1 + i * 3] for i in range(len(graph) // 3)], device=self.device)

        center_topk = center_topk[third_idx.flatten()]
        logp = logp[third_idx.flatten()]
        is_edge = center_topk < graph.num_edges
        node_index = center_topk + graph.num_cum_nodes - graph.num_nodes - graph.num_edges
        edge_index = center_topk + graph.num_cum_edges - graph.num_edges
        center_topk_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        center_topk[:-1]])
        product_id_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        graph.product_id[:-1]])
        is_duplicate = (center_topk == center_topk_shifted) & (graph.product_id == product_id_shifted)
        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)


        center_topk_1 = center_topk[fourth_idx.flatten()]
        logp += logp[fourth_idx.flatten()]
        is_edge_1 = center_topk_1 < graph.num_edges
        edge_index_1 = center_topk_1 + graph.num_cum_edges - graph.num_edges
        center_topk_shifted_1 = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        center_topk_1[:-1]])
        product_id_shifted_1 = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        graph.product_id[:-1]])
        is_duplicate = is_duplicate | ((center_topk_1 == center_topk_shifted_1) & (graph.product_id == product_id_shifted_1))
        edge_index_1 = edge_index_1[is_edge_1]
        edge_mask = edge_mask * (~functional.as_mask(edge_index_1, graph.num_edge))

        if k == 3:
            assert len(graph) % 3 == 0
            reaction_center = torch.zeros(len(graph), 4, dtype=torch.long, device=self.device)
            reaction_center[is_edge, :2] = graph.atom_map[graph.edge_list[edge_index, :2]]
            reaction_center[~is_edge, 0] = graph.atom_map[node_index]
            reaction_center[:, 2:] = reaction_center[:, :2][fourth_idx.flatten()]
            reaction_center = reaction_center[~is_duplicate]
            graph = graph.edge_mask(edge_mask)
            graph = graph[~is_duplicate]
            logp = logp[~is_duplicate]
            synthon, num_synthon = graph.connected_components()
            synthon = synthon.undirected()  # (< num_graph * k)
            result = {
                "synthon": synthon,
                "num_synthon": num_synthon,
                "reaction_center": reaction_center,
                "log_likelihood": logp,
                "reaction": reaction,
            }
            return result

class RandomBFSOrder(object):

    def __call__(self, item):
        assert hasattr(item["graph"][0], "reaction_center")
        reactant, synthon = item["graph"]

        edge_list = reactant.edge_list[:, :2].tolist()
        neighbor = [[] for _ in range(reactant.num_node)]
        for h, t in edge_list:
            neighbor[h].append(t)
        depth = [-1] * reactant.num_node

        # select a mapped atom as BFS root
        reactant2id = reactant.atom_map
        id2synthon = -torch.ones(synthon.atom_map.max() + 1, dtype=torch.long, device=synthon.device)
        id2synthon[synthon.atom_map] = torch.arange(synthon.num_node, device=synthon.device)
        reactant2synthon = id2synthon[reactant2id]

        candidate = (reactant2synthon != -1).nonzero().squeeze(-1)
        i = candidate[torch.randint(len(candidate), (1,))].item()

        queue = deque([i])
        depth[i] = 0
        order = []
        while queue:
            h = queue.popleft()
            order.append(h)
            for t in neighbor[h]:
                if depth[t] == -1:
                    depth[t] = depth[h] + 1
                    queue.append(t)

        reactant = reactant.subgraph(order)

        if reactant.num_edge > 0:
            node_index = reactant.edge_list[:, :2]
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small
            undirected_edge_id = undirected_edge_id * 2 + (node_index[:, 0] > node_index[:, 1])

            # rearrange edges into autoregressive order
            edge_order = undirected_edge_id.argsort()
            reactant = reactant.edge_mask(edge_order)

        assert hasattr(reactant, "reaction_center")

        item = item.copy()
        item["graph"] = (reactant, synthon)

        return item