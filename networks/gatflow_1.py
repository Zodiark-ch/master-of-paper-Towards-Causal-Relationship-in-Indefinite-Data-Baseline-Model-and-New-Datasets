from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from networks.gnn_utils import *
from networks.gat import *

class flow_gat(nn.Module):
    def __init__(self,args,config):
        super(flow_gat,self).__init__()
        self.args=args
        self.config=config
        self.F=GraphNN(args,config)
        self.G=GraphNN(args,config)
        self.y1u=nn.Linear(config.emb_dim,config.emb_dim)
        self.y1v=nn.Linear(config.emb_dim,config.emb_dim)
        self.y2u=nn.Linear(config.emb_dim,config.emb_dim)
        self.y2v=nn.Linear(config.emb_dim,config.emb_dim)
    
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
    
    def forward(self, doc_sents_h, doc_len, adj):#doc_sents_h:因果变量：batch*节点数量*维度   #doc_len:batch*1,代表每个batch里的节点数量，adj：邻接矩阵mask，即为下三角矩阵
        
        '''整个flow 模型的理念基于'''
        x2=doc_sents_h
        x1=torch.zeros_like(x2)
        fx2,fx2_adj=self.F(x2,doc_len,adj)
        y1=x1+fx2       
        
        gy1,gy1_adj=self.G(y1,doc_len,adj)
        y2=x2+gy1
        
        #sampling y2
        y2_u=self.y2u(y2)
        y2_v=self.y2v(y2)
        y2_sample=self.sampling(y2_u,y2_v)
        #sampling y2
        
        x2_hat=y2_sample-gy1
        x1_hat=y1-fx2    
        A=-gy1_adj*fx2_adj 
        
        return x1_hat,x2_hat,A,y2_u,y2_v
        
