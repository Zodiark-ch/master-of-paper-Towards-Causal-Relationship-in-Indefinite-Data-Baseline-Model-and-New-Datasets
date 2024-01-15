import pickle, json, decimal, math
import numpy as np
import torch
from sklearn.metrics import roc_curve,auc
def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))

def auroc(causal_strength,causal_graph,causal_input,causal_label,causal_label_mask):
    batch_adj_mask=causal_label_mask.ge(0.5)
    causal_strength=torch.masked_select(causal_strength,batch_adj_mask).detach().cpu().numpy()
    causal_strength_input=torch.masked_select(causal_input,batch_adj_mask).detach().cpu().numpy()
    causal_graph=torch.masked_select(causal_graph,batch_adj_mask).detach().cpu().numpy()
    causal_label=torch.masked_select(causal_label,batch_adj_mask).detach().cpu().numpy()
    fpr_s,tpr_s,thresholds_s=roc_curve(causal_label,causal_strength,pos_label=1)
    auc_s=auc(fpr_s,tpr_s)
    fpr_g,tpr_g,thresholds_g=roc_curve(causal_label,causal_graph,pos_label=1)
    auc_g=auc(fpr_g,tpr_g)
    fpr_in,tpr_in,thresholds_in=roc_curve(causal_label,causal_strength_input,pos_label=1)
    auc_in=auc(fpr_in,tpr_in)
    return auc_s,auc_g,auc_in

def mse(causal_strength,causal_graph,causal_input,causal_label,causal_label_mask):
    batch_adj_mask=causal_label_mask.ge(0.5)
    causal_strength=torch.where(causal_strength>0.5,1,0)
    causal_input=torch.where(causal_input>0.5,1,0)
    causal_strength=torch.masked_select(causal_strength,batch_adj_mask).detach()
    causal_strength_input=torch.masked_select(causal_input,batch_adj_mask).detach()
    causal_graph=torch.masked_select(causal_graph,batch_adj_mask).detach()
    causal_label=torch.masked_select(causal_label,batch_adj_mask).detach()
    crie=torch.nn.MSELoss()
    mse_s=crie(causal_strength,causal_label)
    mse_g=crie(causal_graph,causal_label)
    mse_in=crie(causal_strength_input,causal_label)
    return mse_s,mse_g,mse_in
    
def consistency(causal_strength,causal_graph,causal_label_mask): 
    crie=torch.nn.MSELoss()
    causal_label_mask=causal_label_mask.ge(0.5)
    batch,max_doc_len,_=causal_strength.size()
    H_do_raw=causal_strength.unsqueeze(2).expand(-1,-1,max_doc_len,-1)
    H_do_arr=causal_strength.unsqueeze(1).expand(-1,max_doc_len,-1,-1)
    similarity=torch.cosine_similarity(H_do_raw,H_do_arr,dim=-1)#[batch,max_doc_len,max_doc_len]
    
    correlation_label=torch.zeros(batch,max_doc_len,max_doc_len).cuda()
    for bc in range(batch):
        for i in range(max_doc_len):
            for j in range(0,i+1):
                pa_i=[]
                pa_j=[]
                for k in range(0,i):
                    if causal_graph[bc][i][k]>0:
                        pa_i.append(k)
                for k in range(0,j):
                    if causal_graph[bc][j][k]>0:
                        pa_j.append(k)
                        
                for ki in range(len(pa_i)):
                    if pa_i[ki] in pa_j:
                        correlation_label[bc][i][j]=1
    correlation_label=torch.masked_select(correlation_label,causal_label_mask)
    similarity=torch.masked_select(similarity,causal_label_mask)
    distance=crie(similarity,correlation_label)
    

    return distance
    
    
    
    
def eval_func(doc_couples_all, doc_couples_pred_all, y_causes_b_all):
    tmp_num = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}
    tmp_den_p = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}
    tmp_den_r = {'ec_p': 0, 'ec_n': 0,'e': 0, 'c': 0}

    for doc_couples, doc_couples_pred, y_causes_b in zip(doc_couples_all, doc_couples_pred_all, y_causes_b_all):
        doc_couples_total = []
        for i,o in enumerate(y_causes_b):
            if o == 1:
                doc_couples_total.extend([f'{i+1},{ii+1}' for ii in range(len(y_causes_b))])
        doc_couples = set([','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples])
        doc_couples_pred = set([','.join(list(map(lambda x: str(x), doc_couple))) for doc_couple in doc_couples_pred])

        tmp_num['ec_p'] += len(doc_couples & doc_couples_pred)
        tmp_den_p['ec_p'] += len(doc_couples_pred)
        tmp_den_r['ec_p'] += len(doc_couples)

        doc_couples_n = set(doc_couples_total) - set(doc_couples)#总候选减去预测的
        doc_couples_pred_n = set(doc_couples_total) - set(doc_couples_pred)
        tmp_num['ec_n'] += len(doc_couples_n & doc_couples_pred_n)
        tmp_den_p['ec_n'] += len(doc_couples_pred_n)
        tmp_den_r['ec_n'] += len(doc_couples_n)

        doc_emos = set([doc_couple.split(',')[0] for doc_couple in doc_couples])
        doc_emos_pred = set([doc_couple.split(',')[0] for doc_couple in doc_couples_pred])
        tmp_num['e'] += len(doc_emos & doc_emos_pred)
        tmp_den_p['e'] += len(doc_emos_pred)
        tmp_den_r['e'] += len(doc_emos)

        doc_caus = set([doc_couple.split(',')[1] for doc_couple in doc_couples])
        doc_caus_pred = set([doc_couple.split(',')[1] for doc_couple in doc_couples_pred])
        tmp_num['c'] += len(doc_caus & doc_caus_pred)
        tmp_den_p['c'] += len(doc_caus_pred)
        tmp_den_r['c'] += len(doc_caus)

    metrics = {}
    for task in ['ec_p','ec_n', 'e', 'c']:
        p = tmp_num[task] / (tmp_den_p[task] + 1e-8)
        r = tmp_num[task] / (tmp_den_r[task] + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        metrics[task] = (p, r, f)
    metrics['ec_avg'] = (np.array(metrics['ec_p']) + np.array(metrics['ec_n'])) / 2

    return metrics['ec_p'], metrics['ec_n'], metrics['ec_avg'], metrics['e'], metrics['c']


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js
