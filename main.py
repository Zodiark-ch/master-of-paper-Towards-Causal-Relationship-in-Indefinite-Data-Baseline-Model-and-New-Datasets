import os,time,argparse,random
import numpy as np
import torch
import logging
import wandb
from data_loader import *
from configs import dataset_config,model_config
from probabilistic_model import baseline_model
from transformers import AdamW,get_cosine_schedule_with_warmup
from train_test import train_eval

import warnings
warnings.filterwarnings('ignore')


parser=argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0',  help='id of gpus')
parser.add_argument('--dataset',default='Causalogue',type=str,help='Causaction or Causalogue')
parser.add_argument('--fold',type=int,default=5)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--wandb',default=True, help='')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
parser.add_argument('--epoch', type=int, default=50, metavar='E', help='number of epochs')
parser.add_argument('--baseline', default='True', help='')
parser.add_argument('--model_name', default='basemodel', type= str, help='')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='how many batchiszes to bp.')
parser.add_argument('--warmup_proportion', type=float, default=0.06, help='the lr up phase in the warmup.')
parser.add_argument('--earlystop', type=int, default=30,  help='id of gpus')
parser.add_argument('--high_level_loss', type=str, default='loss1',  help='id of gpus')
parser.add_argument('--confounding', default='True', help='')
parser.add_argument('--bert_learning',default=True, help='')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#torch.autograd.set_detect_anomaly(True)

if args.dataset=='Causaction':
    args.batch_size=4


if args.wandb==False:
    os.environ["WANDB_DISABLED"] = "true"
if args.wandb==True:
    os.environ["WANDB_DISABLED"] = "false"

def seed_everything(seed=args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(seed)
seed_everything()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #同时输出到屏幕
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

today = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
logger = get_logger('saved_models/' +args.dataset+'_'+ args.model_name +'_'+str(args.lr)+'_'+str(args.epoch)+'_'+str(args.batch_size) +str(today)+'_logging.log')
logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
logger.info(args)

cuda=torch.cuda.is_available()


def main(fold_id):
    train_loader=build_train_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='train',args=args,config=dataset_config)
    valid_loader = build_inference_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='valid',args=args,config=dataset_config)
    test_loader = build_inference_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='test',args=args,config=dataset_config)
    if args.baseline=='True':
        model=baseline_model(args,config=model_config).cuda()
        
    if args.baseline=='False':
        model=SS_mdoel(args,config=model_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    wandb.watch(model, log="all")
    num_steps_all = len(train_loader) // args.gradient_accumulation_steps * args.epoch 
    warmup_steps = int(num_steps_all * args.warmup_proportion)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
    model.zero_grad()
    print('Data and model load finished')    
    
    max_valid_auroc_s,max_valid_auroc_g,max_valid_auroc_in=0,0,0
    max_valid_mse_s,max_valid_mse_g,max_valid_mse_in=0,0,0
    max_test_auroc_s,max_test_auroc_g,max_test_auroc_in=0,0,0
    max_test_mse_s,max_test_mse_g,max_test_mse_in=0,0,0
    for epoch in range(1,int(args.epoch)+1):
        train_auroc_s,train_auroc_g,train_auroc_in,train_mse_s,train_mse_g,train_mse_in,train_distance=train_eval(model,train_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=True)
        logger.info('TRAIN#: fold: {} epoch: {}, auroc_s: {}, auroc_g: {}, auroc_in: {}, mse_s: {}, mse_g: {}, mse_in: {} \n'. \
                format(fold_id,   epoch,      train_auroc_s,             train_auroc_g,             train_auroc_in, train_mse_s,train_mse_g,train_mse_in))
        valid_auroc_s,valid_auroc_g,valid_auroc_in,valid_mse_s,valid_mse_g,valid_mse_in,valid_distance=train_eval(model,valid_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=False)
        logger.info('VALID#: fold: {} epoch: {}, auroc_s: {}, auroc_g: {}, auroc_in: {}, mse_s: {}, mse_g: {}, mse_in: {} \n'. \
                format(fold_id,   epoch,      valid_auroc_s,             valid_auroc_g,             valid_auroc_in, valid_mse_s,valid_mse_g,valid_mse_in))
        test_auroc_s,test_auroc_g,test_auroc_in,test_mse_s,test_mse_g,test_mse_in,test_distance=train_eval(model,test_loader, fold_id,epoch,args,optimizer,scheduler,logger,train=False)
        logger.info('TEST#: fold: {} epoch: {}, auroc_s: {}, auroc_g: {}, auroc_in: {}, mse_s: {}, mse_g: {}, mse_in: {} \n'. \
                format(fold_id,   epoch,      test_auroc_s,             test_auroc_g,             test_auroc_in, test_mse_s,test_mse_g,test_mse_in))
        
        print('fold:{}  epoch:{}      valid_auroc_s:{}, valid_auroc_g:{}, valid_auroc_in:{}, \
            test_auroc_s:{}, test_auroc_g:{}, test_auroc_in:{}, \n fold:{}  epoch:{}      valid_mse_s:{}, valid_mse_g:{}, valid_mse_in:{},\
                test_mse_s:{}, test_mse_g:{}, test_mse_in:{} '.format(fold_id, epoch, valid_auroc_s,valid_auroc_g,valid_auroc_in,test_auroc_s,test_auroc_g,test_auroc_in,fold_id, epoch,valid_mse_s,valid_mse_g,valid_mse_in,test_mse_s,test_mse_g,test_mse_in))
        wandb.log({'epoch': epoch,  'valid_auroc_s':valid_auroc_s,'valid_auroc_g':valid_auroc_g,'valid_auroc_in':valid_auroc_in,\
               'test_auroc_s':test_auroc_s,'test_auroc_g':test_auroc_g,'test_auroc_in':test_auroc_in ,'valid_mse_s':valid_mse_s,'valid_mse_g':valid_mse_g,'valid_mse_in':valid_mse_in,\
               'test_mse_s':test_mse_s,'test_mse_g':test_mse_g,'test_mse_in':test_mse_in })
        #early_stop_flag = 1
        if args.high_level_loss=='loss1' and valid_auroc_s>max_valid_auroc_s:
            early_stop_flag = 1
            max_valid_auroc_s,max_valid_auroc_g,max_valid_auroc_in=valid_auroc_s,valid_auroc_g,valid_auroc_in
            max_test_auroc_s,max_test_auroc_g,max_test_auroc_in=test_auroc_s,test_auroc_g,test_auroc_in
            max_valid_mse_s,max_valid_mse_g,max_valid_mse_in=valid_mse_s,valid_mse_g,valid_mse_in
            max_test_mse_s,max_test_mse_g,max_test_mse_in=test_mse_s,test_mse_g,test_mse_in
        elif args.high_level_loss=='loss2' and valid_auroc_g>max_valid_auroc_g:
            early_stop_flag = 1
            max_valid_auroc_s,max_valid_auroc_g,max_valid_auroc_in=valid_auroc_s,valid_auroc_g,valid_auroc_in
            max_test_auroc_s,max_test_auroc_g,max_test_auroc_in=test_auroc_s,test_auroc_g,test_auroc_in
            max_valid_mse_s,max_valid_mse_g,max_valid_mse_in=valid_mse_s,valid_mse_g,valid_mse_in
            max_test_mse_s,max_test_mse_g,max_test_mse_in=test_mse_s,test_mse_g,test_mse_in
        else:
            early_stop_flag += 1
        if  early_stop_flag >= args.earlystop:
                break
            
    return max_test_auroc_s,max_test_auroc_g,max_test_auroc_in,max_test_mse_s,max_test_mse_g,max_test_mse_in
if __name__=='__main__':
    fold_id=args.fold
    if args.baseline=='False':
        args.model_name='SSmodel'
    if args.baseline=='True':
        args.model_name='basemodel'
    max_test_auroc_s_all,max_test_auroc_g_all,max_test_auroc_in_all=0,0,0
    max_test_mse_s_all,max_test_mse_g_all,max_test_mse_in_all=0,0,0
    #for fold_id in range(2,10):
    wandb_config=dict(lr=args.lr,batch_size=args.batch_size,fold=fold_id)
    wandb.init(config=wandb_config,reinit=True,project='Indefinite_baseline_Results',name='{}_bert_{}_lr_{}_batch_{}_fold_{}'.format(args.dataset,args.bert_learning,args.lr,args.batch_size,fold_id))
    print('===== fold {} ====='.format(fold_id))
    max_test_auroc_s,max_test_auroc_g,max_test_auroc_in,max_test_mse_s,max_test_mse_g,max_test_mse_in=main(fold_id=fold_id)
    print('max_test_auroc_s: {},max_test_auroc_g: {},max_test_auroc_in: {},max_test_mse_s: {},max_test_mse_g: {},max_test_mse_in: {}'.format(max_test_auroc_s,max_test_auroc_g,max_test_auroc_in,max_test_mse_s,max_test_mse_g,max_test_mse_in))
    wandb.log({'max_test_auroc_s':max_test_auroc_s,'max_test_auroc_g':max_test_auroc_g,'max_test_auroc_in':max_test_auroc_in,\
        'max_test_mse_s':max_test_mse_s,'max_test_mse_g':max_test_mse_g,'max_test_mse_in':max_test_mse_in})
    
    max_test_auroc_s_all+=max_test_auroc_s
    max_test_auroc_g_all+=max_test_auroc_g
    max_test_auroc_in_all+=max_test_auroc_in
    max_test_mse_s_all+=max_test_mse_s
    max_test_mse_g_all+=max_test_mse_g
    max_test_mse_in_all+=max_test_mse_in
    print('======== all ========')
    max_test_auroc_s_all=max_test_auroc_s_all/1
    max_test_auroc_g_all=max_test_auroc_g_all/1
    max_test_auroc_in_all=max_test_auroc_in_all/1
    max_test_mse_s_all=max_test_mse_s_all/1
    max_test_mse_g_all=max_test_mse_g_all/1
    max_test_mse_in_all=max_test_mse_in_all/1
    print('max_test_auroc_s_all: {},max_test_auroc_g_all: {},max_test_auroc_in_all: {},max_test_mse_s_all: {},max_test_mse_g_all: {},max_test_mse_in_all: {}'.format(max_test_auroc_s_all,max_test_auroc_g_all,max_test_auroc_in_all,max_test_mse_s_all,max_test_mse_g_all,max_test_mse_in_all))
    today = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    file=open('saved_models/' + args.model_name +'_'+str(args.lr)+'_'+str(args.epoch)+'_'+str(args.batch_size) +str(today)+'.txt','w')
    results='max_test_auroc_s_all: {},max_test_auroc_g_all: {},max_test_auroc_in_all: {},max_test_mse_s_all: {},max_test_mse_g_all: {},max_test_mse_in_all: {}'.format(max_test_auroc_s_all,max_test_auroc_g_all,max_test_auroc_in_all,max_test_mse_s_all,max_test_mse_g_all,max_test_mse_in_all)
    file.close()