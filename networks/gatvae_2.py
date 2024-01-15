from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from networks.gnn_utils import *

class gat_vae(nn.Module):
    def __init__(self,args,config):
        super(gat_vae,self).__init__()
        self.args=args
        self.config=config
        self.encoder=GATVAEencoder(args,config)
        self.Udecoder=GATVAEdecoder(args,config)
        self.Sdecoder=Attentiondecoder(args,config)
        self.fce=nn.Linear(1,1)
        self.fcs=nn.Linear(1,1)
        self.noisex=nn.Linear(config.emb_dim,config.emb_dim)
        self.noisel=nn.Linear(config.emb_dim,config.emb_dim)
    
    def sampling(self,mu, log_var):
        # result=[]
        # result.append(mu[0])
        # for i in range(1,self.args.gnn_layers+1):
            
        #     std = torch.exp(0.5*log_var[i])
        #     eps = torch.randn_like(std)
        #     result.append(eps.mul(std).add_(mu[i]))
            
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        result=eps.mul(std).add_(mu)
        return result # return z sample
    
    def forward(self, doc_sents_h, doc_len, adj):
        
        #fU=BL+e, Ex=e, El=L
        fU,adjB_list=self.encoder(doc_sents_h,doc_len,adj)
        Ex=self.noisex(fU)
        El=self.noisel(fU)
        
        posL=torch.relu(El)#[B*head*len,dim]
        one=torch.ones_like(posL)
        biL=torch.where(posL==0,posL,one)
        rankp=torch.ones_like(doc_len)
        for i in range(doc_len.size()[0]):
            rankL=torch.linalg.matrix_rank(biL[i][:doc_len[i],:])#[B]
            rankp[i]=rankL/doc_len[i]
        
        S=self.Sdecoder(El,doc_sents_h)
        
        
        adj_e=torch.clone(adjB_list).view(-1,1)
        adj_s=torch.clone(adjB_list).view(-1,1)
        e=self.fce(adj_e).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],adjB_list.size()[3])
        s=self.fcs(adj_s).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],adjB_list.size()[3])
        
        e_bp=self.fce(adj_e.detach()).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],adjB_list.size()[3])
        s_bp=self.fcs(adj_s.detach()).view(adjB_list.size()[0],adjB_list.size()[1],adjB_list.size()[2],adjB_list.size()[3])
        
        z=self.sampling(e,s)
        z=torch.where(adjB_list==0,adjB_list,z)
        U,b_inv=self.Udecoder(Ex,doc_len,adj,z)

        return U,b_inv,e_bp,s_bp,rankp,S
        
class GATVAEencoder(nn.Module):
    def __init__(self, args,config):
        super(GATVAEencoder, self).__init__()
        in_dim = config.emb_dim
        self.gnn_dims = [in_dim,config.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [config.multihead]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GATencoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], 0.1)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h,W = gnn_layer(doc_sents_h, adj)

        return doc_sents_h,W



class GATVAEdecoder(nn.Module):
    def __init__(self, args,config):
        super(GATVAEdecoder, self).__init__()
        in_dim = config.emb_dim
        self.gnn_dims = [in_dim,config.gat_feat_dim]

        self.gnn_layers = 1
        self.att_heads = [config.multihead]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GATdecoderlayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], 0.1)
            )

    def forward(self, doc_sents_h, doc_len, adj,W):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h,b_inv = gnn_layer(doc_sents_h, W,adj)

        return doc_sents_h,b_inv
    
class Attentiondecoder(nn.Module):
    def __init__(self,args,config):
        super(Attentiondecoder,self).__init__()
        self.args=args
        self.config=config
        
        
    def forward(self,El,X):
        #X=[B,len,dim]   El=[B,len,dim] 单纯的估计，没有任何可训练参数，通过L和X估计S
        batch=X.size()[0]
        len=X.size()[1]
        
        inputX=X.detach()
        inputL=El.detach()
        px=torch.sigmoid(inputX) #value=[0,1]
        pl=torch.sigmoid(inputL) #value=[0,1]
        pxpl=px*pl #[B,len,dim] 
        
        attn=pxpl/torch.sum(pxpl,dim=-2,keepdim=True)#[B,len,dim]
        return attn*X
        
        
        


