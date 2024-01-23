import torch
from torch import nn
from collections.abc import Sequence
from torch.nn import functional as F
class BranchedMultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, text_emb_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(BranchedMultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.text_emb_dim = text_emb_dim
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        self.text_layer = nn.Linear(text_emb_dim, text_emb_dim)
        for i in range(len(self.dims) - 1):
            if i != 1:
                self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            else:
                self.layers.append(nn.Linear(self.dims[i] + text_emb_dim, self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input, text_emb, repeat_list):
        """"""
        layer_input = input
        for i, layer in enumerate(self.layers):
            if i != 1:
                hidden = layer(layer_input)
            else:
                hidden = layer(torch.cat([layer_input, self.text_layer(text_emb.repeat_interleave(repeat_list, dim=0))], dim=-1))
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden
    

class CrossAttentionMultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, text_emb_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(CrossAttentionMultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.text_emb_dim = text_emb_dim
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        self.text_layer = nn.Linear(text_emb_dim, text_emb_dim)
        for i in range(len(self.dims) - 1):
            if i != 1:
                self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            else:
                self.layers.append(CrossAttentionBlock(self.dims[i], text_emb_dim, self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input, text_emb, repeat_list):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            if i != 1:
                hidden = layer(layer_input)
            else:
                hidden = layer(layer_input, text_emb, repeat_list)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, text_emb_dim, hidden_dim):
        super(CrossAttentionBlock, self).__init__()

        self.feature_dim = feature_dim
        self.text_emb_dim = text_emb_dim
        self.hidden_dim = hidden_dim

        self.query_matrix = nn.Linear(text_emb_dim, feature_dim)
        self.key_matrix = nn.Linear(feature_dim, feature_dim)
        self.value_matrix = nn.Linear(feature_dim, feature_dim)
        self.output_matrix = nn.Linear(feature_dim, hidden_dim)
        

    def forward(self, input, text_emb, repeat_list):
        pad_idx = 0.0
        splited_input, mask = self.pad_and_transform(input, repeat_list, pad_idx)
        query = self.query_matrix(text_emb)
        key = self.key_matrix(splited_input)
        value = self.value_matrix(splited_input)
        query = query.unsqueeze(1).repeat_interleave(key.shape[1], dim=1)
        key.masked_fill_(mask == pad_idx, 0.0)
        value.masked_fill_(mask == pad_idx, 0.0)
        attention = torch.matmul(query, key.transpose(-1, -2))
        attention_mask = torch.matmul(torch.ones_like(query), mask.transpose(-1, -2))
        attention.masked_fill_(attention_mask == pad_idx, -torch.inf)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention.transpose(-1, -2), value)
        output = torch.cat([output[i,:repeat_list[i],:] for i in range(len(repeat_list))])
        output = self.output_matrix(output)

        return output
    def pad_and_transform(self, input, repeat_list, pad_idx):
        # Pad sequence to max_len
        graph_list = input.split(repeat_list.tolist())
        seq = nn.utils.rnn.pad_sequence(graph_list, batch_first=True, padding_value=pad_idx) 
        mask = nn.utils.rnn.pad_sequence(torch.ones_like(input).split(repeat_list.tolist()), batch_first=True, padding_value=pad_idx)
        
        return seq, mask
