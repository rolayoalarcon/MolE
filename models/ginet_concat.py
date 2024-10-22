import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.BatchNorm1d(2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim),
            nn.ReLU()
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Graph Isomorphism Network implementation for graph-level representation learning.
    Representation is built with concatenation of each layer embeddings.

    Args:
        num_layer (int): Number of GIN layers.
        emb_dim (int): Dimension of node embeddings. The final representation will be num_layer * emb_dim.
        feat_dim (int): Dimension of the output embedding space. This is the input to the BT loss
        drop_ratio (float): Dropout ratio.
        pool (str): Pooling method for graph-level representations. Options: 'mean', 'max', 'add'.

    Attributes:
        num_layer (int): Number of GIN layers.
        emb_dim (int): Dimension of node embeddings.
        feat_dim (int): Dimension of the output embedding space.
        drop_ratio (float): Dropout ratio.
        x_embedding1 (nn.Embedding): Embedding layer for atom type.
        x_embedding2 (nn.Embedding): Embedding layer for atom chirality.
        gnns (nn.ModuleList): List of GIN layers.
        batch_norms (nn.ModuleList): List of batch normalization layers.
        pool (function): Pooling function for graph-level representations.
        feat_lin (nn.Linear): Linear layer for feature dimension reduction.
        out_lin (nn.Sequential): Output linear layer.
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.concat_dim = num_layer * emb_dim
        if self.concat_dim != self.feat_dim:
            print(f"Representation Dimension ({self.concat_dim}) - Embedding dimension ({self.feat_dim})")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.concat_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim), 
            nn.ReLU(inplace=True),

            nn.Linear(self.feat_dim, self.feat_dim), # Is not reduced to half size!
            nn.BatchNorm1d(self.feat_dim), 
            nn.ReLU(inplace=True),

            nn.Linear(self.feat_dim, self.feat_dim)
        )

    def forward(self, data):

        """
        Forward pass of the GINet model.

        Args:
            data (Data): Input graph data.

        Returns:
            Tensor: Graph-level representation.
            Tensor: Embedding vector.
        """

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h_init = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        
        # Perform the convolutions
        h_dict = {}

        for layer in range(self.num_layer):
            if layer == self.num_layer - 1:
                tmp_h = self.gnns[layer](h_dict[f"h_{layer - 1}"], edge_index, edge_attr)
                tmp_h = self.batch_norms[layer](tmp_h)
                h_dict[f"h_{layer}"] = F.dropout(tmp_h, self.drop_ratio, training=self.training)

            else: 
                if layer == 0:
                    tmp_h = self.gnns[layer](h_init, edge_index, edge_attr)
                    tmp_h = self.batch_norms[layer](tmp_h)
                    h_dict[f"h_{layer}"] = F.dropout(F.relu(tmp_h), self.drop_ratio, training=self.training)
                else:
                    tmp_h = self.gnns[layer](h_dict[f"h_{layer - 1}"], edge_index, edge_attr)
                    tmp_h = self.batch_norms[layer](tmp_h)
                    h_dict[f"h_{layer}"] = F.dropout(F.relu(tmp_h), self.drop_ratio, training=self.training)

        # Graph representation
        h_list_pooled = [self.pool(h_dict[f"h_{layer}"], data.batch) for layer in range(self.num_layer)]
        h_global_embedding = torch.cat(h_list_pooled, dim=1)

        assert h_global_embedding.shape[1] == self.concat_dim

        # Projection
        h_expansion = self.feat_lin(h_global_embedding) 
        out = self.out_lin(h_expansion)
        
        return h_global_embedding, out
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print(name)
            own_state[name].copy_(param)
