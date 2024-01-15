import torch
import torch.nn as nn
import torch .nn.functional as F
import torch.nn.init as init
import numpy as np

def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30

 
class GraphAttentionLayer(nn.Module):
    
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(torch.tanh(h), self.w_src)
        attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        adj = torch.FloatTensor(adj).cuda()
        mask = 1 - adj.unsqueeze(1)
        mask_one=torch.ones_like(mask)
        acyclic=torch.tril(mask_one,diagonal=-1)
        acyclic[:,:,0,:]=0
        acyclic[:,:,0,0]=1
        mask=torch.where(acyclic==1,mask,mask_one)
        # print(attn.size())
        # print(mask.size())
        attn.data.masked_fill_(mask.bool(), -999)

        attn = torch.softmax(attn, dim=-1)
    
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out,attn
    
    
    

class GATencoderlayer(nn.Module):
    
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GATencoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.strengh=nn.Linear(self.out_dim*2,1)
        self.sigmoid=nn.Sigmoid()
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)
        

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()
        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)
        H_src=h.expand(-1, adj.size()[1], -1, -1).permute(0,2,1,3)
        H_dst=h.expand(-1, adj.size()[1], -1, -1)
        H_pair=torch.cat((H_src,H_dst),dim=-1)
        attn=self.strengh(H_pair).squeeze(-1)#batch*N*N
        attn = self.sigmoid(attn).unsqueeze(1)
        attn_mask_one=torch.ones_like(adj)
        attn_mask=torch.tril(attn_mask_one.unsqueeze(1),diagonal=-1)
        attn=torch.where(attn_mask==1,attn,attn_mask)
        I_A=I-attn
        I_A_raw=I_A.clone()
        # "scale the attn to the matrix with attn_{i,i}=1"
        # scale=torch.ones(batch, self.att_head, N).cuda()
        # for i in range(batch):
        #     for j in range(self.att_head):
        #         attn_adj_diag=torch.diag(attn[i][j].detach())
        #         scale[i][j]=torch.div(scale[i][j],attn_adj_diag)
        # scale=scale.unsqueeze(-1)#[8,4,4,1]
        # attn_scale=torch.mul(I_A,scale)
        # attn_scale.data.masked_fill_(mask.bool(), 0)

        # attn_src = torch.matmul(torch.tanh(h), self.w_src)
        # attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
        # attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        # attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)
        
        
        #adj = torch.FloatTensor(adj)
        mask = 1 - adj.unsqueeze(1)
        # print(attn.size())
        # print(mask.size())
        # I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()#邻接矩阵为（I-B)，
        # I.data.masked_fill_(mask.bool(),0)
        mask_one=torch.ones_like(mask)
        acyclic=torch.tril(mask_one,diagonal=0)
        mask=torch.where(acyclic==1,mask,mask_one)
        #attn=torch.where(I==1,mask_zero,attn)
        
        I_A.data.masked_fill_(mask.bool(), -999999)
        #attn=I-attn
        
        I_A = torch.softmax(I_A, dim=-1)
        
        feat_out = torch.matmul(I_A, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)
        
        

        return feat_out,I_A_raw

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
class GATdecoderlayer(nn.Module):
    
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GATdecoderlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, IB,adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim
        #IB=IB.unsqueeze(1)
        #I=torch.eye(N).repeat(batch,self.att_head,1,1).cuda()#邻接矩阵为（I-B）的逆，
        #b_inv = torch.linalg.solve(I,IB)
        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)
        
        mask = 1 - adj.unsqueeze(1)
        # print(attn.size())
        # print(mask.size())
        mask_one=torch.ones_like(mask)
        acyclic=torch.tril(mask_one,diagonal=0)
        mask=torch.where(acyclic==1,mask,mask_one)
        IB.data.masked_fill_(mask.bool(), -999999)
        IB = torch.softmax(IB, dim=-1)

        feat_out = torch.matmul(IB, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
    
    
       