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
        self.decoder=GATVAEdecoder(args,config)
        self.C_estimator=Attentiondecoder(args,config)
        self.fce=nn.Linear(1,1)
        self.fcs=nn.Linear(1,1)
        self.noisex=nn.Linear(config.emb_dim,config.emb_dim)
        self.noisel=nn.Linear(config.emb_dim,config.emb_dim)

        self.rankpool=nn.Linear(config.emb_dim,4)
    
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
        fU,adjB_list=self.encoder(doc_sents_h,doc_len,adj)#FU size: [batch,N,emb]
        # get x and l
        Ex=self.noisex(fU)
        El=self.noisel(fU)
        posL=torch.relu(self.rankpool(El))#[B*head*len,dim]
        one=torch.ones_like(posL)
        biL=torch.where(posL==0,posL,one)
        rankp=torch.ones_like(doc_len,dtype=torch.float32)
        for i in range(doc_len.size()[0]):
            rankL=torch.linalg.matrix_rank(biL[i][:doc_len[i],:])#only conclute the no-padding raws
            rankp[i]=rankL/doc_len[i]
            
        C=self.C_estimator(El,doc_sents_h)
        
        batch, N, in_dim = doc_sents_h.size()
        
        #get the inverse matrix of (I-A)
        I=torch.eye(N).repeat(batch,self.config.multihead,1,1).cuda()
        I_A=torch.inverse(adjB_list)
        
        #sample from (I-A)
        adj_e=torch.clone(I_A).reshape(-1,1)
        adj_s=torch.clone(I_A).reshape(-1,1)
        e=self.fce(adj_e).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        s=self.fcs(adj_s).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        z=self.sampling(e,s)
        z=torch.where(I_A==0,I_A,z)
        if self.args.confounding=="True":
            X=self.decoder(Ex,doc_len,adj,z)
        
        else:
            X=self.decoder(fU,doc_len,adj,z)

        return X,I-adjB_list,e,s,rankp,C
        
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
            doc_sents_h = gnn_layer(doc_sents_h, W,adj)

        return doc_sents_h
    
    
class Attentiondecoder(nn.Module):
    def __init__(self,args,config):
        super(Attentiondecoder,self).__init__()
        self.args=args
        self.config=config
        
        
    def forward(self,El,X):
        #X=[B,len,dim]   El=[B,len,dim] 单纯的估计，没有任何可训练参数，通过L和X估计S
        batch=X.size()[0]
        len=X.size()[1]
        
        inputX=X
        inputL=El
        px=torch.sigmoid(inputX) #value=[0,1]
        pl=torch.sigmoid(inputL) #value=[0,1]
        pxpl=px*pl #[B,len,dim] 
        
        attn=pxpl/torch.sum(pxpl,dim=-2,keepdim=True)#[B,len,dim]
        return attn*X


