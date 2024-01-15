import torch
import torch.nn as nn
from networks.rank import *

class Diclass(nn.Module):
    def __init__(self,args,config):
        super(Diclass,self).__init__()
        self.args=args
        self.config=config
        self.rankU=RankNN(args,config)
        self.rankS=RankNN(args,config)
        self.predU=classfication(args,config)
        self.predS=classfication(args,config)
        
    def forward(self,U,S):
        S_allU=self.diclass(U,S.detach())#rgiht label U
        U_allS=self.diclass(S.detach(),U)#fake label S
        
        list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos=[],[],[],[],[],[],[],[]
        for i in range (len(S_allU)):
            pred_UE,pred_UC=self.predU(S_allU[i])
            couples_pred, emo_cau_pos = self.rankU(S_allU[i])
            list_UE.append(pred_UE)
            list_UC.append(pred_UC)
            list_UP.append(couples_pred)
            list_Upos.append(emo_cau_pos)
        for i in range (len(U_allS)):
            pred_SE,pred_SC=self.predU(U_allS[i])
            couples_pred, emo_cau_pos = self.rankU(U_allS[i])
            list_SE.append(pred_SE)
            list_SC.append(pred_SC)
            list_SP.append(couples_pred)
            list_Spos.append(emo_cau_pos)
        return list_UE,list_UC,list_UP,list_Upos,list_SE,list_SC,list_SP,list_Spos
    
    def diclass(self,U,S):
        S_U=[]
        assert U.size()[0]==S.size()[0]
        batch=U.size()[0]
        for i in range(batch):
            S_i=S[i].unsqueeze(0).repeat(batch,1,1)# S_i=[batch,N,feature]
            SU=S_i+U
            S_U.append(SU) #list [batch*tensor],tensor=[batch,N,feature]
        return S_U
    
    
    
         
        
        
class classfication(nn.Module):
    def __init__(self, args,config):
        super(classfication, self).__init__()
        #self.feat_dim = int(args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim)#输入维度
        self.feat_dim=config.feat_dim
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
        