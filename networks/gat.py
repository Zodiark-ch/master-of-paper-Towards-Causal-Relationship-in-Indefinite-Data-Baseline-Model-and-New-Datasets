import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.gnn_utils import *





class GraphNN(nn.Module):
    def __init__(self, args,configs):
        super(GraphNN, self).__init__()
        in_dim = configs.emb_dim
        self.gnn_dims = [in_dim,configs.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [configs.multihead]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], 0.1)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h,b_inv = gnn_layer(doc_sents_h, adj)

        return doc_sents_h,b_inv