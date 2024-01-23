import inspect

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
logger = logging.getLogger(__name__)
@R.register("tasks.RetrosynthesisText")
class RetrosynthesisText(tasks.Task, core.Configurable):
    """
    Retrosynthesis task.

    This class wraps pretrained center identification and synthon completion modeules into a pipeline.

    Parameters:
        center_identification (CenterIdentification): sub task of center identification
        synthon_completion (SynthonCompletion): sub task of synthon completion
        center_topk (int, optional): number of reaction centers to predict for each product
        num_synthon_beam (int, optional): size of beam search for each synthon
        max_prediction (int, optional): max number of final predictions for each product
        metric (str or list of str, optional): metric(s). Available metrics are ``top-K``.
    """

    _option_members = {"metric"}

    def __init__(self, center_identification, synthon_completion, center_topk=2, num_synthon_beam=10, max_prediction=20,
                 metric=("top-1", "top-3", "top-5", "top-10")):
        super(RetrosynthesisText, self).__init__()
        self.center_identification = center_identification
        self.synthon_completion = synthon_completion
        self.center_topk = center_topk
        self.num_synthon_beam = num_synthon_beam
        self.max_prediction = max_prediction
        self.metric = metric

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            raise ValueError("Retrosynthesis only supports load_state_dict() with strict=True")
        keys = set(state_dict.keys())
        for model in [self.center_identification, self.synthon_completion]:
            if set(model.state_dict().keys()) == keys:
                return model.load_state_dict(state_dict, strict)
        raise RuntimeError("Neither of sub modules matches with state_dict")

    def predict(self, batch, all_loss=None, metric=None):
        synthon_batch = self.center_identification.predict_synthon(batch, self.center_topk)
        
        synthon = synthon_batch["synthon"]
        num_synthon = synthon_batch["num_synthon"]
        assert (num_synthon >= 1).all() and (num_synthon <= 2).all()
        synthon2split = torch.repeat_interleave(num_synthon)
        with synthon.graph():
            synthon.reaction_center = synthon_batch["reaction_center"][synthon2split]
            synthon.split_logp = synthon_batch["log_likelihood"][synthon2split]

        reactant = self.synthon_completion.predict_reactant(synthon_batch, self.num_synthon_beam, self.max_prediction)

        logps = []
        reactant_ids = []
        product_ids = []

        # case 1: one synthon
        is_single = num_synthon[synthon2split[reactant.synthon_id]] == 1
        reactant_id = is_single.nonzero().squeeze(-1)
        logps.append(reactant.split_logp[reactant_id] + reactant.logp[reactant_id])
        product_ids.append(reactant.product_id[reactant_id])
        # pad -1
        reactant_ids.append(torch.stack([reactant_id, -torch.ones_like(reactant_id)], dim=-1))

        # case 2: two synthons
        # use proposal to avoid O(n^2) complexity
        reactant1 = torch.arange(len(reactant), device=self.device)
        reactant1 = reactant1.unsqueeze(-1).expand(-1, self.max_prediction * 2)
        reactant2 = reactant1 + torch.arange(self.max_prediction * 2, device=self.device)
        valid = reactant2 < len(reactant)
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        logps.append(reactant.split_logp[reactant1] + reactant.logp[reactant1] + reactant.logp[reactant2])
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(torch.stack([reactant1, reactant2], dim=-1))

        # combine case 1 & 2
        logps = torch.cat(logps)
        reactant_ids = torch.cat(reactant_ids)
        product_ids = torch.cat(product_ids)

        order = product_ids.argsort()
        logps = logps[order]
        reactant_ids = reactant_ids[order]
        num_prediction = product_ids.bincount()
        logps, topk = functional.variadic_topk(logps, num_prediction, self.max_prediction)
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index_shifted = torch.cat([-torch.ones(len(topk_index), 1, dtype=torch.long, device=self.device),
                                        topk_index[:, :-1]], dim=-1)
        is_duplicate = topk_index == topk_index_shifted
        reactant_id = reactant_ids[topk_index] # (num_graph, k, 2)

        # why we need to repeat the graph?
        # because reactant_id may be duplicated, which is not directly supported by graph indexing
        is_padding = reactant_id == -1
        num_synthon = (~is_padding).sum(dim=-1)
        num_synthon = num_synthon[~is_duplicate]
        logps = logps[~is_duplicate]
        offset = torch.arange(self.max_prediction, device=self.device) * len(reactant)
        reactant_id = reactant_id + offset.view(1, -1, 1)
        reactant_id = reactant_id[~(is_padding | is_duplicate.unsqueeze(-1))]
        reactant = reactant.repeat(self.max_prediction)
        reactant = reactant[reactant_id]
        assert num_synthon.sum() == len(reactant)
        synthon2graph = torch.repeat_interleave(num_synthon)
        first_synthon = num_synthon.cumsum(0) - num_synthon
        # inherit graph attributes from the first synthon
        data_dict = reactant.data_mask(graph_index=first_synthon, include="graph")[0]
        # merge synthon pairs from the same split into a single graph
        reactant = reactant.merge(synthon2graph)
        with reactant.graph():
            for k, v in data_dict.items():
                setattr(reactant, k, v)
            reactant.logps = logps

        num_prediction = reactant.product_id.bincount()

        return reactant, num_prediction # (num_graph * k)

    def predict_3(self, batch, all_loss=None, metric=None, only_3=False):
        # synthon_batch = self.center_identification.predict_synthon(batch, self.center_topk)
        synthon_batch = self.center_identification.predict_synthon_3(batch, 3)

        rewards = 0
        if only_3:
            rewards -= 100
        synthon = synthon_batch["synthon"]
        num_synthon = synthon_batch["num_synthon"]
        assert (num_synthon >= 1).all() and (num_synthon <= 3).all()
        synthon2split = torch.repeat_interleave(num_synthon)
        with synthon.graph():
            synthon.reaction_center = synthon_batch["reaction_center"][synthon2split]
            synthon.split_logp = synthon_batch["log_likelihood"][synthon2split]
            
        reactant = self.synthon_completion.predict_reactant(synthon_batch, self.num_synthon_beam, self.max_prediction)
        logps = []
        reactant_ids = []
        product_ids = []

        # case 1: one synthon
        is_single = num_synthon[synthon2split[reactant.synthon_id]] == 1
        reactant_id = is_single.nonzero().squeeze(-1)
        logps.append(reactant.split_logp[reactant_id] + reactant.logp[reactant_id] * 3 + rewards)
        product_ids.append(reactant.product_id[reactant_id])
        # pad -1
        reactant_ids.append(torch.stack([reactant_id, -torch.ones_like(reactant_id), -torch.ones_like(reactant_id)], dim=-1))

        # case 2: two synthons
        # use proposal to avoid O(n^2) complexity
        reactant1 = torch.arange(len(reactant), device=self.device)
        reactant1 = reactant1.unsqueeze(-1).expand(-1, self.max_prediction * 2)
        reactant2 = reactant1 + torch.arange(self.max_prediction * 2, device=self.device)
        valid = reactant2 < len(reactant)
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        logps.append(reactant.split_logp[reactant1] + (reactant.logp[reactant1] + reactant.logp[reactant2]) * 1.5 + rewards)
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(torch.stack([reactant1, reactant2, -torch.ones_like(reactant1)], dim=-1))

        # case 3: three synthons
        reactant1 = torch.arange(len(reactant), device=self.device)
        reactant1 = reactant1.unsqueeze(-1).unsqueeze(-1).expand(-1, self.max_prediction * 3, self.max_prediction * 3)
        reactant2 = reactant1 + torch.arange(self.max_prediction * 3, device=self.device)
        reactant3 = (reactant2.transpose(1, 2) + torch.arange(self.max_prediction * 3, device=self.device)).transpose(1, 2)
        valid = (reactant2 < len(reactant)) * (reactant3 < len(reactant))
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        reactant3 = reactant3[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        synthon3 = reactant.synthon_id[reactant3]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2]) & (synthon2 < synthon3) & (synthon2split[synthon2] == synthon2split[synthon3])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        reactant3 = reactant3[valid]
        logps.append(reactant.split_logp[reactant1] + reactant.logp[reactant1] + reactant.logp[reactant2] + reactant.logp[reactant3])
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(torch.stack([reactant1, reactant2, reactant3], dim=-1))


        # combine case 1 & 2
        logps = torch.cat(logps)
        reactant_ids = torch.cat(reactant_ids)
        product_ids = torch.cat(product_ids)

        order = product_ids.argsort()
        logps = logps[order]
        reactant_ids = reactant_ids[order]
        num_prediction = product_ids.bincount()
        logps, topk = functional.variadic_topk(logps, num_prediction, self.max_prediction)
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index_shifted = torch.cat([-torch.ones(len(topk_index), 1, dtype=torch.long, device=self.device),
                                        topk_index[:, :-1]], dim=-1)
        is_duplicate = topk_index == topk_index_shifted
        reactant_id = reactant_ids[topk_index] # (num_graph, k, 2)

        is_padding = reactant_id == -1
        num_synthon = (~is_padding).sum(dim=-1)
        num_synthon = num_synthon[~is_duplicate]
        logps = logps[~is_duplicate]
        offset = torch.arange(self.max_prediction, device=self.device) * len(reactant)
        reactant_id = reactant_id + offset.view(1, -1, 1)
        reactant_id = reactant_id[~(is_padding | is_duplicate.unsqueeze(-1))]
        reactant = reactant.repeat(self.max_prediction)
        reactant = reactant[reactant_id]
        assert num_synthon.sum() == len(reactant)
        synthon2graph = torch.repeat_interleave(num_synthon)
        first_synthon = num_synthon.cumsum(0) - num_synthon
        # inherit graph attributes from the first synthon
        data_dict = reactant.data_mask(graph_index=first_synthon, include="graph")[0]
        # merge synthon pairs from the same split into a single graph
        reactant = reactant.merge(synthon2graph)
        with reactant.graph():
            for k, v in data_dict.items():
                setattr(reactant, k, v)
            reactant.logps = logps

        num_prediction = reactant.product_id.bincount()

        return reactant, num_prediction # (num_graph * k)

    def target(self, batch):
        reactant, product = batch["graph"]
        reactant = reactant.ion_to_molecule()
        return reactant

    def evaluate(self, pred, target):
        pred, num_prediction = pred
        infinity = torch.iinfo(torch.long).max - 1

        metric = {}
        ranking = []
        # any better solution for parallel graph isomorphism?
        num_cum_prediction = num_prediction.cumsum(0)
        for i in range(len(target)):
            target_smiles = target[i].to_smiles(isomeric=False, atom_map=False, canonical=True)
            offset = (num_cum_prediction[i] - num_prediction[i]).item()
            for j in range(num_prediction[i]):
                pred_smiles = pred[offset + j].to_smiles(isomeric=False, atom_map=False, canonical=True)
                if pred_smiles == target_smiles:
                    break
            else:
                j = infinity
            ranking.append(j + 1)

        ranking = torch.tensor(ranking, device=self.device)
        for _metric in self.metric:
            if _metric.startswith("top-"):
                threshold = int(_metric[4:])
                score = (ranking <= threshold).float().mean()
                metric["top-%d accuracy" % threshold] = score
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

        return metric