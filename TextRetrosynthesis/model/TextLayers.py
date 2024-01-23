import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from TextRetrosynthesis.model.Blocks import CrossAttentionBlock
from torchdrug import data, utils
    

class RelationalGraphConvText(nn.Module):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.

    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """
    eps = 1e-10
    gradient_checkpoint = False

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu", 
                 with_text=False, PLM_d=512, text_layer=None):
        super(RelationalGraphConvText, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim
        self.with_text = with_text
        self.PLM_d = PLM_d
        self.text_layer = text_layer
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if with_text and text_layer == "linear":
            self.self_loop = nn.Linear(input_dim + PLM_d, output_dim)
            self.projecting_layer = nn.Linear(PLM_d, PLM_d)
        elif with_text and text_layer == "cross_attention":
            self.self_loop = CrossAttentionBlock(input_dim, PLM_d, output_dim)
            self.projecting_layer = nn.Identity()
        elif with_text:
            self.self_loop = nn.Linear(input_dim + PLM_d, output_dim)
            self.projecting_layer = nn.Identity()
        else:
            self.self_loop = nn.Linear(input_dim, output_dim)
            self.projecting_layer = nn.Identity()
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def forward(self, graph, input, text_emb=None):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(graph, input)
        if self.with_text and text_emb is not None:
            output = self.combine(input, update, text_emb=text_emb, repeat_list = graph.num_atoms)
        else:
            output = self.combine(input, update)
        return output
  
    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update, text_emb=None, repeat_list=None):
        """
        Combine node input and node update.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        if self.with_text and text_emb is not None and repeat_list is not None:
            if self.text_layer == "cross_attention":
                output = self.linear(update) + self.self_loop(input, text_emb, repeat_list)
            else:
                output = self.linear(update) + self.self_loop(torch.cat([input, self.projecting_layer(text_emb).repeat_interleave(repeat_list, dim=0)], dim=-1))
        else:
            output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
