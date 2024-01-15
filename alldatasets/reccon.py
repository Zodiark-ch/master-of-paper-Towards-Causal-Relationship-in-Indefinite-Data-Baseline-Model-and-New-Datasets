import torch
from torch.utils.data import Dataset
from utils import *
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer



class RecconDataset(Dataset):
    def __init__(self,fold_id,data_type,args,config):
        self.data_type=data_type
        self.args=args
        self.config=config
        self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.config.roberta_pretrain_path,local_files_only=True)
        if config.iemocaptest == True and data_type=='test':
            self.data_dir='data/iemocap/iemocap_test_cls.json'
            self.speaker_vocab=pickle.load(open('data/iemocap/speaker_vocab.pkl', 'rb'))
            self.label_vocab=pickle.load(open('data/iemocap/label_vocab.pkl' , 'rb'))
        else:
            self.data_dir='data/dailydialog/fold%s/dailydialog_%s_cls.json'%(fold_id,data_type)
            self.speaker_vocab={'stoi': {'A': 0, 'B': 1}, 'itos': ['A', 'B']}
            self.label_vocab ={'stoi': {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}, 'itos': ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']}
        
        

        self.doc_id_list,self.doc_len_list,self.doc_couples_list,self.doc_speaker_list,self.doc_text_list,\
        self.bert_token_idx_list,self.bert_clause_idx_list,self.bert_segments_idx_list,self.bert_token_lens_list= self.read_data_file(self.data_dir)
    
    def __len__(self):
        return len(self.doc_id_list)
    
    def __getitem__(self, idx):
        doc_id,doc_len,doc_couples=self.doc_id_list[idx],self.doc_len_list[idx],self.doc_couples_list[idx]
        doc_speaker,doc_text=self.doc_speaker_list[idx],self.doc_text_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]

        if bert_token_lens > 512 and self.args.withbert==True:
            bert_token_idx, bert_clause_idx,bert_segments_idx, bert_token_lens,doc_couples, doc_len,doc_speaker = \
                self.token_trunk(bert_token_idx, bert_clause_idx,bert_segments_idx, bert_token_lens,
                                doc_couples, doc_len,doc_speaker)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        assert doc_len == len(doc_speaker)
        return doc_id,doc_len,doc_couples,doc_speaker,doc_text, \
            bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens
        
    def read_data_file(self, data_dir):
        datafile=data_dir
        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        doc_speaker_list=[]
        doc_text_list=[]
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        data=read_json(datafile)
        for doc in data:
            doc_id=doc['doc_id']#id
            doc_len=doc['doc_len']#number of utterances
            doc_couples = doc['pairs']#pairs of outcomes and causes
            doc_clauses = doc['clauses']
            doc_speaker=doc['speakers']
            doc_id_list.append(doc_id)###
            doc_len_list.append(doc_len)##
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)###
            doc_str = ''
            doc_text = []
            for i in range(doc_len):
                doc_speaker[i]=self.speaker_vocab['stoi'][doc_speaker[i]]               
                clause_id = doc_clauses[i]['clause_id']#id
                assert int(clause_id) == i + 1
                doc_text.append(doc_clauses[i]['clause'])
                doc_str+='<s>'+doc_clauses[i]['clause']+' </s> '
            doc_text_list.append(doc_text)###
            doc_speaker_list.append(doc_speaker)###
            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 0]
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 0]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)
            
        return doc_id_list,doc_len_list,doc_couples_list,doc_speaker_list,doc_text_list,\
            bert_token_idx_list,bert_clause_idx_list,bert_segments_idx_list,bert_token_lens_list
    
    
    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, doc_len,doc_speaker):
        # TODO: cannot handle some extreme cases now
    
        i = doc_len - 1
        while True:
            temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
            if len(temp_bert_token_idx) <= 512:
                cls_idx = bert_clause_idx[i]
                bert_token_idx = bert_token_idx[:cls_idx]
                bert_segments_idx = bert_segments_idx[:cls_idx]
                bert_clause_idx = bert_clause_idx[:i]
                y_emotions = y_emotions[:i]
                y_causes = y_causes[:i]
                doc_speaker=doc_speaker[:i]
                doc_emotion_category=doc_emotion_category[:i]
                doc_len = i
                break
            i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, doc_len,doc_speaker

    def get_adj(self, couples, max_dialog_len,len):
        '''
    
        '''
        adj = []
    
        for couple in couples:
            label = torch.ones(max_dialog_len, max_dialog_len)
            label[:len[couple],:len[couple]]=0
            label=torch.where(label==0,0,-1)
            for pair in couple:
                outcome=pair[0]-1
                cause=pair[1]-1
                label[outcome][cause]=1
            adj.append(label)
        return torch.stack(adj)
    

    def collate_fn(self,batch):
        '''
        :param data:
            doc_id,doc_len,doc_couples,doc_speaker,doc_text
        :return:
            B:batch_size  N:batch_max_doc_len
            batch_ids:(B)
            batch_doc_len(B)
            batch_doc_speaker:(B,N) padded[1,0,-1]
            batch_label:(B,N,N)
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            batch_utterances:  not a tensor
            batch_utterances_mask:(B,N) padded[1,0]
            batch_adj_mask (B,N,N) low trigle matrix
        '''
        
        doc_id,doc_len,doc_couples,doc_speaker,doc_text,bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b=zip(*batch)
        max_dialog_len=max(int(length) for length in doc_len)
        batch_ids=torch.IntTensor([docid for docid in doc_id])
        batch_doc_len=torch.LongTensor([doclen for doclen in doc_len])
        batch_doc_speaker=pad_sequence([torch.LongTensor(speaker) for speaker in doc_speaker], batch_first=True,padding_value=-1)
        batch_label=self.get_adj([couple for couple in doc_couples],max_dialog_len,[len for len in doc_len])
        batch_utterances=[utt for utt in doc_text]
        batch_utterances_mask=torch.zeros(len(doc_id),max_dialog_len)
        for i in range(len(doc_len)):
            batch_utterances_mask[i][:doc_len[i]]=1
        batch_adj_mask=torch.zeros(len(doc_id),max_dialog_len,max_dialog_len)
        for i in range(len(doc_len)):
            batch_adj_mask[i][:doc_len[i],:doc_len[i]]=1
            
        bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
        bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
        bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)
        bsz, max_len = bert_token_b.size()
        bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
        for index, seq_len in enumerate(bert_token_lens_b):
            bert_masks_b[index][:seq_len] = 1

        bert_masks_b = torch.FloatTensor(bert_masks_b)
        assert bert_segment_b.shape == bert_token_b.shape
        assert bert_segment_b.shape == bert_masks_b.shape
        
        return batch_ids,batch_doc_len,batch_doc_speaker,batch_label,batch_utterances, batch_utterances_mask,batch_adj_mask,\
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b
        
                       
                