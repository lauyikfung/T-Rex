from collections.abc import Sequence
from typing import List
import torch
from torch import nn

from torchdrug import core, layers
from .TextLayers import RelationalGraphConvText
from torchdrug.core import Registry as R

@R.register("models.TextRGCN")
class TextRGCN(nn.Module, core.Configurable):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, short_cut=False, batch_norm=False, with_text=False,
                 activation="relu", concat_hidden=False, readout="sum", PLM_d=512, text_layer=None):
        super(TextRGCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.PLM_d = PLM_d
        self.text_layer = text_layer
        self.layers = nn.ModuleList()
        if isinstance(with_text, bool):
            with_text = [with_text] * (len(self.dims) - 1)
        if isinstance(with_text, List):
            if all(isinstance(n, bool) for n in with_text):
                if len(with_text) == len(self.dims) - 1:
                    self.with_text = with_text
                else:
                    self.with_text = with_text + [False] * (len(self.dims) - 1 - len(with_text))
            else:
                self.with_text = [False] * (len(self.dims) - 1)
        else:
            self.with_text = [False] * (len(self.dims) - 1)
        for i in range(len(self.dims) - 1):
            self.layers.append(RelationalGraphConvText(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim,
                                                          batch_norm, activation, with_text=with_text[i], PLM_d=PLM_d, text_layer=text_layer))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None, text_emb=None):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for i, layer in enumerate(self.layers):
            if self.with_text[i] and text_emb is not None:
                hidden = layer(graph, layer_input, text_emb=text_emb)
            else:
                hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }