import torch
from torch.utils.data import Dataset
from utils import *
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
import numpy as np


class VideoDataset(Dataset):
    def __init__(self,fold_id,data_type,args,config):
        self.data_type=data_type
        self.args=args
        self.config=config
        self.data_dir='data/Causaction/fold%s/%s.json'%(fold_id,data_type)
        
        

        self.id_list,self.seg_length_list,self.file_list,self.seg_id_list,self.action_list,self.adj_list= self.read_data_file(self.data_dir)
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id,seg_length,file=self.id_list[idx],self.seg_length_list[idx],self.file_list[idx]
        seg_id,action,adj=self.seg_id_list[idx],self.action_list[idx],self.adj_list[idx]
        
        assert seg_length == len(seg_id)==len(action)
        return id,seg_length,file,seg_id,action,adj
    
    def read_data_file(self, data_dir):
        datafile=data_dir
        id_list = []
        seg_length_list=[]
        file_list=[]
        seg_id_list=[]
        action_list=[]
        adj_list=[]
        data=read_json(datafile)
        for doc in data:
            id=doc['id']#id
            segment_len=doc['segment_length'] #number of actions
            if segment_len<=3:
                print('id%s has %s segments so being removed'%(id,segment_len))
                continue
            adj=doc['adj']
            id_list.append(id)
            seg_length_list.append(segment_len)
            adj_np=np.array(adj)
            adj_np=np.where(adj_np>0.3,1,0)
            adj_np.astype(int)
            adj_list.append(adj_np)
            segment=doc['segment']
            action_id_sample=[]
            action_sample=[]
            file_sample=[]
            for i in range(len(segment)):
                action_id=segment[i]['action_id']
                action=segment[i]['action']
                file=segment[i]['representation_path']
                action_id_sample.append(action_id)
                action_sample.append(action)
                file_sample.append(file)
            seg_id_list.append(action_id_sample)
            action_list.append(action_id_sample)
            file_list.append(file_sample)
            
        return id_list,seg_length_list,file_list,seg_id_list,action_list,adj_list
    
    def get_adj(self, couples, max_dialog_len,len):
        '''
    
        '''
        adj = []
    
        for couple in couples:
            label = torch.zeros(max_dialog_len, max_dialog_len)
            for pair in couple:
                outcome=pair[0]-1
                cause=pair[1]-1
                label[outcome][cause]=1
            adj.append(label)
        return torch.stack(adj)
    

    def collate_fn(self,batch):
        '''
        :param data:
            id_list,seg_length_list,file_list,seg_id_list,action_list,adj_list
        :return:
            B:batch_size  N:batch_max_doc_len
            batch_ids:(B)
            batch_seg_len(B)
            batch_label:(B,N,N)
            batch_label_mask:(B,N,N)low trigle matrix with diagnal-1
            batch_cls:(B,N,T,D)
            batch_seg_id:(B,N)
            batch_action:(B,N)
            batch_adj_mask (B,N,N) low trigle matrix
        '''
        
        id,seg_length,file,seg_id,action,adj=zip(*batch)
        max_segment_len=max(int(length) for length in seg_length)
        #print(max_segment_len)
        batch_ids=torch.IntTensor([docid for docid in id])
        batch=batch_ids.size()[0]
        batch_doc_len=torch.LongTensor([doclen for doclen in seg_length]).cuda()
        batch_label=torch.zeros(batch,max_segment_len,max_segment_len).cuda()
        for i in range(batch):
            adj_pad=np.pad(adj[i],((0,max_segment_len-seg_length[i]),(0,max_segment_len-seg_length[i])),'constant',constant_values=(-1,-1))
            batch_label[i]=torch.from_numpy(adj_pad)
        batch_adj_mask=torch.ones(len(id),max_segment_len,max_segment_len).cuda()
        for i in range(len(seg_length)):
            batch_adj_mask[i]=torch.tril(batch_adj_mask[i])
        batch_label_mask=torch.ones(len(id),max_segment_len,max_segment_len).cuda()
        for i in range(len(seg_length)):
            batch_label_mask[i]=torch.tril(batch_label_mask[i],diagonal=-1)
        batch_label_mask=torch.where(batch_label==-1,0,batch_label_mask)
            
        batch_seg_id=[seg for seg in seg_id]
        batch_action=[ac for ac in action]
        
        
        batch_cls=torch.zeros(batch,max_segment_len,300,2048).cuda()
        for filename in range(len(file)):
            for file_name in range(len(file[filename])):
                cls=np.load('/home/data/dddd/data4/%s'%(file[filename][file_name]))
                batch_cls[filename][file_name]=torch.from_numpy(cls)
        
        return batch_ids,batch_doc_len,batch_label,batch_label_mask,batch_cls,batch_seg_id,batch_action,batch_adj_mask
        
                       
                