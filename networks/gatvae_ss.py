from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from networks.gnn_utils import *

class gat_vae_do(nn.Module):
    def __init__(self,args,config):
        super(gat_vae_do,self).__init__()
        self.args=args
        self.config=config
        self.encoder=GATVAEencoder(args,config)
        self.decoder=GATVAEdecoder(args,config)
        self.fce=nn.Linear(1,1)
        self.fcs=nn.Linear(1,1)
    
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
        fU,I_A=self.encoder(doc_sents_h,doc_len,adj)
        batch, N, in_dim = doc_sents_h.size()
        I=torch.eye(N).repeat(batch,self.config.multihead,1,1).cuda()
        # I_A=torch.inverse(adjB_list)
        # adj_e=torch.clone(I_A).reshape(-1,1)
        # adj_s=torch.clone(I_A).reshape(-1,1)
        # e=self.fce(adj_e).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        # s=self.fcs(adj_s).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        # z=self.sampling(e,s)
        # z=torch.where(I_A==0,I_A,z)
        X_raw,X_do,correlation_label,e,s=self.decoder(fU,doc_len,adj,I_A)#z=W(the inverse of I-A),causal graph is A
        causal_graph=I-I_A
        
        return X_raw,X_do,correlation_label,causal_graph,e,s
        
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
        self.fce=nn.Linear(1,1)
        self.fcs=nn.Linear(1,1)
    
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

    def forward(self, doc_sents_h, doc_len, adj,I_A):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        I=torch.eye(max_doc_len).repeat(batch,1,1,1).cuda()
        causal_graph=I-I_A
        I_A_inverse=torch.inverse(I_A)
        adj_e=torch.clone(I_A_inverse).reshape(-1,1)
        adj_s=torch.clone(I_A_inverse).reshape(-1,1)
        e=self.fce(adj_e).view(I_A_inverse.size()[0],I_A_inverse.size()[1],I_A_inverse.size()[2],-1)
        s=self.fcs(adj_s).view(I_A_inverse.size()[0],I_A_inverse.size()[1],I_A_inverse.size()[2],-1)
        z=self.sampling(e,s)
        W=torch.where(I_A_inverse==0,I_A_inverse,z)
        
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h_raw = gnn_layer(doc_sents_h, W,adj)#batch,node_num,node_num
        
        correlation_label,W_do,do_num=self.computing(max_doc_len,causal_graph)
        causal_graph=causal_graph.squeeze(-3)
        doc_sents_h_do=doc_sents_h.unsqueeze(1).expand(-1,do_num,-1,-1).reshape(batch*do_num,max_doc_len,-1)
        adj=adj.unsqueeze(1).expand(-1,do_num,-1,-1).reshape(batch*do_num,max_doc_len,-1)
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h_do = gnn_layer(doc_sents_h_do, W_do,adj)#batch*do_num,node_num,node_num

        return doc_sents_h_raw,doc_sents_h_do,correlation_label,e,s#
    
    def computing(self,max_doc_len,causal_graph):
        causal_graph=causal_graph.squeeze(-3)
        batch=causal_graph.size()[0]
        node_num=causal_graph.size()[-1]
        do_num=int(0.5*(node_num*(node_num-1)))
        raw_do=[]
        for i in range(0,node_num-1):
            for j in range(i+1,node_num):
                raw_do.append([i,j])#number=do_num
        correlation_label=torch.zeros(batch,do_num,node_num,node_num).cuda()
        causal_graph_do=causal_graph.unsqueeze(1).expand(-1,do_num,-1,-1).clone()
        for bc in range(batch):
            for do in range(do_num):
                causal_graph_do[bc][do][raw_do[do][0]]=0
                causal_graph_do[bc][do][raw_do[do][1]]=0
                for i in range(node_num):
                    for j in range(0,i+1):
                        if i==j:
                            correlation_label[bc][do][i][j]=1
                        elif j==raw_do[do][0] and i!=raw_do[do][1]:
                            assert causal_graph[bc][i][j]==causal_graph_do[bc][do][i][j]
                            if causal_graph[bc][i][j]>0:
                                correlation_label[bc][do][i][j]=1
                        elif j==raw_do[do][1] and i!=raw_do[do][0]:
                            assert causal_graph[bc][i][j]==causal_graph_do[bc][do][i][j]
                            if causal_graph[bc][i][j]>0:
                                correlation_label[bc][do][i][j]=1
                        elif j!=raw_do[do][0] and j!=raw_do[do][1] and i!=raw_do[do][0] and i!=raw_do[do][1]:
                            pa_i=[]
                            pa_j=[]
                            for k in range(0,i):
                                assert causal_graph[bc][i][k]==causal_graph_do[bc][do][i][k]
                                if causal_graph[bc][i][k]>0:
                                    pa_i.append(k)
                            for k in range(0,j):
                                assert causal_graph[bc][j][k]==causal_graph_do[bc][do][j][k]
                                if causal_graph[bc][j][k]>0:
                                    pa_j.append(k)
                                    
                            for ki in range(len(pa_i)):
                                if pa_i[ki] in pa_j:
                                    correlation_label[bc][do][i][j]=1
                        else:continue
        correlation_label=correlation_label.reshape(batch*do_num,node_num,node_num)  #batch*do_num,node_num,node_num
        causal_graph_do=causal_graph_do.reshape(batch*do_num,node_num,node_num)   #batch*do_num,node_num,node_num 
        I=torch.eye(node_num).repeat(batch*do_num,1,1).cuda()     
        causal_graph_do=causal_graph_do+I         
        I_A=torch.inverse(causal_graph_do).unsqueeze(1)
        adj_e=torch.clone(I_A).reshape(-1,1)
        adj_s=torch.clone(I_A).reshape(-1,1)
        e=self.fce(adj_e).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        s=self.fcs(adj_s).view(I_A.size()[0],I_A.size()[1],I_A.size()[2],-1)
        z=self.sampling(e,s).squeeze(-1)
        z=torch.where(I_A==0,I_A,z)#batch*do_num,node_num,node_num
        return correlation_label,z,do_num
        
                            
       
        
                                


