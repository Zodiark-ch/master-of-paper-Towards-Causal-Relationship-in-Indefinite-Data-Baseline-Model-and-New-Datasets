import torch
import torch.nn as nn
import torch .nn.functional as F
from transformers import RobertaModel
from networks.gatvae_ss import *
from networks.gatvae_1 import *

class baseline_model(nn.Module):
    def __init__(self,args,config):
        super(baseline_model,self).__init__()
        self.args=args
        self.config=config
        if self.args.dataset=='Causalogue':
            self.bert=RobertaModel.from_pretrained(self.config.roberta_pretrain_path)
            config.emb_dim=768
            config.feat_dim=768
            config.gat_feat_dim=768
            if args.bert_learning==False:
                for p in self.parameters():
                    p.requires_grad=False
        if self.args.dataset=="Causaction":
            self.oneDlayer=nn.Linear(300*2048,config.emb_dim)
        self.gnn=gat_vae(args,config)
        #layers=[nn.Linear(self.config.feat_dim*2,self.config.feat_dim),nn.Sigmoid(),nn.Dropout(0.5),nn.Linear(self.config.feat_dim,1),nn.Sigmoid(),nn.Dropout(0.5),nn.Linear(1,1)]
        #self.out_mlp = nn.Sequential(*layers)
        self.strengh=nn.Linear(self.config.feat_dim*2,1)
        # self.strengh2=nn.Linear(1,1)
        # self.pool=nn.Sigmoid()
       
       
    def forward(self,lengths,adj_mask,bert_token_b,bert_masks_b,bert_clause_b):
        if self.args.dataset=='Causalogue':
            bert_output=self.bert(input_ids=bert_token_b.cuda(),attention_mask=bert_masks_b.cuda())
            doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.cuda())
        if self.args.dataset=="Causaction":
            batch=lengths.size()[0]
            max_len=adj_mask.size()[1]
            doc_sents_h = bert_token_b.view(batch,max_len,-1)
            doc_sents_h=self.oneDlayer(doc_sents_h)
        
        H,A,e,s,rank,C=self.gnn(doc_sents_h,lengths,adj_mask)
        H_src=H.unsqueeze(-2).expand(-1, -1, doc_sents_h.size()[1], -1)
        H_dst=H.unsqueeze(-3).expand(-1, doc_sents_h.size()[1], -1, -1)
        H_pair=torch.cat((H_src,H_dst),dim=-1)
        pred_results=torch.relu(self.strengh(H_pair).squeeze(-1))
        assert not torch.any(torch.isnan(pred_results))
        assert not torch.any(torch.isnan(A))
        
        doc_sents_h_src=doc_sents_h.unsqueeze(-2).expand(-1, -1, doc_sents_h.size()[1], -1)
        doc_sents_h_dst=doc_sents_h.unsqueeze(-3).expand(-1, doc_sents_h.size()[1], -1, -1)
        doc_sents_h_pair=torch.cat((doc_sents_h_src,doc_sents_h_dst),dim=-1)
        doc_sents_h_pred_results=torch.relu(self.strengh(doc_sents_h_pair).squeeze(-1))
        assert not torch.any(torch.isnan(doc_sents_h_pred_results))
        
        
        return H,doc_sents_h,A.squeeze(1),e,s,rank,pred_results,doc_sents_h_pred_results,C
    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
    
    # def loss_ss(self,H_do,correlation_label,batch_label_mask):
    #     criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #     batch,do_num,max_doc_len,_=H_do.size()[1]
    #     batch_label_mask.expand(-1,do_num,-1,-1).reshape(batch*do_num,max_doc_len,-1)
    #     batch_label_mask=batch_label_mask.ge(0.5)
    #     H_do_raw=H_do.expand(-1,-1,max_doc_len,-1)
    #     H_do_arr=H_do.expand(-1,max_doc_len,-1,-1)
    #     similarity=torch.cosine_similarity(H_do_raw,H_do_arr,dim=-1)
        
        
    def loss_hl(self,causal_strengh,causal_graph,batch_label,batch_label_mask):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        batch_label_mask=batch_label_mask.ge(0.5)
        causal_strengh=torch.masked_select(causal_strengh,batch_label_mask)
        causal_graph=torch.masked_select(causal_graph,batch_label_mask)
        batch_label=torch.masked_select(batch_label,batch_label_mask)
        loss1=criterion(causal_strengh,batch_label)
        loss2=criterion(causal_graph,batch_label)
        return loss1+loss2

    
    def loss_KL(self,e,s):
        # batch=e[1].size()[0]
        # utt=e[1].size()[1]
        # num=batch*utt*utt
        # sum=0
        # for i in range(1,self.args.gnn_layers+1):
        #     KLD= -0.5 * torch.sum(1 + s[i] - e[i].pow(2) - s[i].exp())
        #     KLD=KLD/num
        #     sum+=KLD
        e=e.squeeze(1)
        s=s.squeeze(1)
        batch=e.size()[0]
        utt=e.size()[1]
        num=batch*utt*utt    
        # print(e,s)
        # print(e.pow(2),s.exp())
        KLD= -0.5 * torch.sum(1 + s - e.pow(2) - s.exp())
        sum=KLD/num
        if sum>2:
            return 1
        else:
            return sum
    
    def loss_reconstruction(self,X_hat,X,confounding,rank,batch_label_mask):
        batch_label_mask=batch_label_mask.ge(0.5)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        assert X_hat.size()==X.size()
        batch,N,emb=X_hat.size()
        xhat_withC=torch.add(X_hat,confounding)
        xhat_withC_src=xhat_withC.unsqueeze(-2).expand(-1, -1, N, -1)
        xhat_withC_dst=xhat_withC.unsqueeze(-3).expand(-1, N, -1, -1)
        xhat_src=X_hat.unsqueeze(-2).expand(-1, -1, N, -1)
        xhat_dst=X_hat.unsqueeze(-3).expand(-1, N, -1, -1)
        x_src=X.unsqueeze(-2).expand(-1, -1, N, -1)
        x_dst=X.unsqueeze(-3).expand(-1, N, -1, -1)
        
        # Sim_xhat_withC_pair=torch.cat((xhat_withC_src,xhat_withC_dst),dim=-1)
        # Sim_xhat_withC=torch.relu(self.strengh(Sim_xhat_withC_pair).squeeze(-1))
        # Sim_xhat_pair=torch.cat((xhat_src,xhat_dst),dim=-1)
        # Sim_xhat=torch.relu(self.strengh(Sim_xhat_pair).squeeze(-1))
        # Sim_x_pair=torch.cat((x_src,x_dst),dim=-1)
        # Sim_x=torch.relu(self.strengh(Sim_x_pair).squeeze(-1))
        
        
        Sim_xhat_withC=torch.cosine_similarity(xhat_withC_src,xhat_withC_dst,dim=-1)
        Sim_xhat=torch.cosine_similarity(xhat_src,xhat_dst,dim=-1)
        Sim_x=torch.cosine_similarity(x_src,x_dst,dim=-1)
        if self.args.confounding=="True":
            loss_all=0
            for i in range(batch):
                Sim_xhat_withC_batch=torch.masked_select(Sim_xhat_withC[i],batch_label_mask)
                Sim_xhat_batch=torch.masked_select(Sim_xhat[i],batch_label_mask)
                Sim_x_batch=torch.masked_select(Sim_x[i],batch_label_mask)
                
                
                loss_withC=criterion(Sim_xhat_withC_batch,Sim_x_batch)
                loss_woC=criterion(Sim_xhat_batch,Sim_x_batch)
                loss=rank[i]*loss_withC+(1-rank[i])*loss_woC
                loss_all+=loss
            return loss_all/batch
        else: 
            loss=criterion(Sim_xhat,Sim_x)
            return loss
    
    # def loss_reconstruction(self,causal_strength,causal_graph,causal_label_mask):
    #     batch_adj_mask=causal_label_mask.ge(0.5)
    #     causal_strength=torch.masked_select(causal_strength,batch_adj_mask)
    #     causal_graph=torch.masked_select(causal_graph,batch_adj_mask)
    #     crie=torch.nn.MSELoss()
    #     distance=crie(causal_strength,causal_graph)
    #     return distance
    
    
class SS_model(nn.Module):
    def __init__(self,args,config):
        super(SS_mdoel,self).__init__()
        self.args=args
        self.config=config
        self.bert=RobertaModel.from_pretrained(self.config.roberta_pretrain_path)
        self.gnn=gat_vae_do(args,config)
        #layers=[nn.Linear(self.config.feat_dim*2,self.config.feat_dim),nn.Sigmoid(),nn.Dropout(0.5),nn.Linear(self.config.feat_dim,1),nn.Sigmoid(),nn.Dropout(0.5),nn.Linear(1,1)]
        #self.out_mlp = nn.Sequential(*layers)
        self.strengh=nn.Linear(self.config.feat_dim*2,1)
        # self.strengh2=nn.Linear(1,1)
        # self.pool=nn.Sigmoid()
       
       
    def forward(self,lengths,adj_mask,bert_token_b,bert_masks_b,bert_clause_b):
        bert_output=self.bert(input_ids=bert_token_b.cuda(),attention_mask=bert_masks_b.cuda())
        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.cuda())
        H,doc_sents_h,A,e,s,rank=self.gnn(doc_sents_h,lengths,adj_mask)
        
        return H,A,e,s,rank
        
    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
    
    def loss_ss(self,H_do,correlation_label,batch_label_mask):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        batch=batch_label_mask.size()[0]
        all,max_doc_len,_=H_do.size()
        do_num=int(all/batch)
        batch_label_mask=batch_label_mask.unsqueeze(1).expand(-1,do_num,-1,-1).reshape(batch*do_num,max_doc_len,-1)
        batch_label_mask=batch_label_mask.ge(0.5)
        H_do_raw=H_do.unsqueeze(2).expand(-1,-1,max_doc_len,-1)
        H_do_arr=H_do.unsqueeze(1).expand(-1,max_doc_len,-1,-1)
        similarity=torch.cosine_similarity(H_do_raw,H_do_arr,dim=-1)#[all,max_doc_len,max_doc_len]
        correlation_label=torch.masked_select(correlation_label,batch_label_mask)
        similarity=torch.masked_select(similarity,batch_label_mask)
        loss=criterion(similarity,correlation_label)
        return loss
        
        
    def loss_hl(self,causal_strengh,causal_graph,batch_label,batch_adj_mask):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        batch_adj_mask=batch_adj_mask.ge(0.5)
        causal_strengh=torch.masked_select(causal_strengh,batch_adj_mask)
        causal_graph=torch.masked_select(causal_graph,batch_adj_mask)
        batch_label=torch.masked_select(batch_label,batch_adj_mask)
        loss1=criterion(causal_strengh,batch_label)
        loss2=criterion(causal_graph,batch_label)
        #loss=loss1+loss2
        if self.args.high_level_loss=='loss1':
            return loss1
        if self.args.high_level_loss=='loss2':
            return loss2
    
    def loss_KL(self,e,s):
        # batch=e[1].size()[0]
        # utt=e[1].size()[1]
        # num=batch*utt*utt
        # sum=0
        # for i in range(1,self.args.gnn_layers+1):
        #     KLD= -0.5 * torch.sum(1 + s[i] - e[i].pow(2) - s[i].exp())
        #     KLD=KLD/num
        #     sum+=KLD
        e=e.squeeze()
        s=s.squeeze()
        batch=e.size()[0]
        utt=e.size()[1]
        num=batch*utt*utt    
        # print(e,s)
        # print(e.pow(2),s.exp())
        KLD= -0.5 * torch.sum(1 + s - e.pow(2) - s.exp())
        sum=KLD/num
        if sum>2:
            return 1
        else:
            return sum
    
    def loss_reconstruction(self,X_rec,X):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        X_rec=X_rec.view(-1,768)
        X=X.view(-1,768)
        loss=criterion(X_rec,X)
        return loss
    
    # def loss_reconstruction(self,causal_strength,causal_graph,causal_label_mask):
    #     batch_adj_mask=causal_label_mask.ge(0.5)
    #     causal_strength=torch.masked_select(causal_strength,batch_adj_mask)
    #     causal_graph=torch.masked_select(causal_graph,batch_adj_mask)
    #     crie=torch.nn.MSELoss()
    #     distance=crie(causal_strength,causal_graph)
    #     return distance
        
        
        
        
