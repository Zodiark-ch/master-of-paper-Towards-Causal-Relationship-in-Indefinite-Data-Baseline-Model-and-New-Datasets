import torch
from utils import *
import wandb
from configs import dataset_config,model_config
import math
from plot import *
import matplotlib.pyplot as plt

def train_eval(model,dataloader,fold,epoch,args,optimizer,scheduler,logger,train=False):
    assert not model or dataloader or optimizer or scheduler!= None
    # if epoch==6:
    #     print(epoch)
    if args.dataset=='Causalogue':
        if dataset_config.large==False:
            test_sample=math.ceil(200/args.batch_size)
            valid_sample=math.ceil(100/args.batch_size)
        else:
            test_sample=math.ceil(600/args.batch_size)
            valid_sample=math.ceil(300/args.batch_size)
    if args.dataset=='Causaction':
        test_sample=math.ceil(218/args.batch_size)
        valid_sample=math.ceil(100/args.batch_size)
    if train:
        model.train()
        logger.info('########################Training######################')
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        logger.info('########################Evaling#######################')
    
    trainstep=0
    evalstep=0
    teststep=0
    s_AUROC_all_train=0
    s_AUROC_all_eval=0
    s_AUROC_all_test=0
    g_AUROC_all_train=0
    g_AUROC_all_eval=0
    g_AUROC_all_test=0
    in_AUROC_all_train=0
    in_AUROC_all_eval=0
    in_AUROC_all_test=0
    
    s_MSE_all_train=0
    s_MSE_all_eval=0
    s_MSE_all_test=0
    g_MSE_all_train=0
    g_MSE_all_eval=0
    g_MSE_all_test=0
    in_MSE_all_train=0
    in_MSE_all_eval=0
    in_MSE_all_test=0
    
    
    distance_all_train=0
    distance_all_eval=0
    distance_all_test=0
    for train_step, batch in enumerate(dataloader, 1):
        if args.dataset=='Causalogue':
            batch_ids,batch_doc_len,batch_doc_speaker,batch_label,batch_label_mask,batch_utterances, batch_utterances_mask,batch_adj_mask,\
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b=batch
        if args.dataset=='Causaction':
            batch_ids,batch_doc_len,batch_label,batch_label_mask,batch_cls,batch_seg_id,batch_action,batch_adj_mask=batch
            
        if train and len(batch_ids)!=args.batch_size:
            continue
        
        if train:
            trainstep+=1
        else:
            #print(len(dataloader))
            if len(dataloader)==test_sample:
                teststep+=1
            else:
                evalstep+=1
        
            
       
    
        if train:
            if args.dataset=='Causalogue':
                X_hat,X,A,e,s,rank,pred_results,pred_results_input,confounding=model(batch_doc_len,batch_adj_mask,bert_token_b,bert_masks_b,bert_clause_b)
            if args.dataset=='Causaction':
                X_hat,X,A,e,s,rank,pred_results,pred_results_input,confounding=model(batch_doc_len,batch_adj_mask,batch_cls,None,None)
        else:
            with torch.no_grad():
                if args.dataset=='Causalogue':
                    X_hat,X,A,e,s,rank,pred_results,pred_results_input,confounding=model(batch_doc_len,batch_adj_mask,bert_token_b,bert_masks_b,bert_clause_b)
                if args.dataset=='Causaction':
                    X_hat,X,A,e,s,rank,pred_results,pred_results_input,confounding=model(batch_doc_len,batch_adj_mask,batch_cls,None,None)
                # if epoch==40: 
                #     for i in range(len(batch_doc_len)):
                #         #if batch_doc_len[i]==7 or batch_doc_len[i]==8:
                #         img,im=plot_cka_matrix(causal_graph[i],batch_doc_len[i])
                #         texts = annotate_heatmap(im, valfmt="{x:.2f}")
                #         img.savefig('savefig/sslmodel_{}/{}sslmodel_{}.jpg'.format(args.high_level_loss,batch_ids[i],args.high_level_loss))
                #     plt.show()
                # loc=torch.where(causal_strengh!=causal_strengh)
                # causal_strengh[loc]=0
        loss_KL=model.loss_KL(e,s)
        loss_reconsctruction=model.loss_reconstruction(X_hat,X,confounding,rank,batch_label_mask)
        loss_high_level=model.loss_hl(pred_results,A,batch_label,batch_label_mask)
        #loss_ss=model.loss_ss(H_do,correlation_label,batch_label_mask)
        #loss_KL=1
        #loss_re=1
        # loss_high_level=1
        # loss_KL=model.loss_KL(e,s)
        # loss_re=model.loss_reconstruction(X_rec,X)
        loss = loss_high_level + loss_KL + loss_reconsctruction
        if train:
            logger.info('TRAIN# fold: {}, epoch: {}, iter: {},  loss_high_level: {},  loss_KL: {},  loss_re:{}'. \
                        format(fold,   epoch,    trainstep, loss_high_level,     loss_KL,     loss_reconsctruction))
            
            wandb.log({'epoch': epoch,  'trainstep':trainstep+len(dataloader)*epoch,'loss_all_train':loss,'loss_high_level_train':loss_high_level,'loss_KL_train':loss_KL,'loss_re_train':loss_reconsctruction})
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if args.dataset=='Causalogue':
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10,norm_type=2)
            if args.dataset=='Causaction':
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,norm_type=2)
            if train_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            s_AUROC,g_AUROC,input_AUROC=auroc(pred_results,A,pred_results_input,batch_label,batch_label_mask)
            s_MSE,g_MSE,input_MSE=mse(pred_results,A,pred_results_input,batch_label,batch_label_mask)
            s_AUROC_all_train+=s_AUROC
            g_AUROC_all_train+=g_AUROC
            in_AUROC_all_train+=input_AUROC
            s_AUROC=0
            g_AUROC=0
            input_AUROC=0
            
            s_MSE_all_train+=s_MSE
            g_MSE_all_train+=g_MSE
            in_MSE_all_train+=input_MSE
            s_MSE=0
            g_MSE=0
            input_MSE=0
            # distance=consistency(X_rec,causal_graph,batch_label_mask)
            # distance_all_train+=distance
            # distance=0
            
            
            
        else:
            if len(dataloader)==test_sample:
                logger.info('TEST# fold: {}, epoch: {}, iter: {},  loss_high_level: {},  loss_KL: {},  loss_re:{}'. \
                                format(fold,   epoch,    train_step, loss_high_level,     loss_KL,     loss_reconsctruction))
                
                wandb.log({'epoch': epoch,  'teststep':teststep+len(dataloader)*epoch,'loss_all_test':loss,'loss_high_level_test':loss_high_level,'loss_KL_test':loss_KL,'loss_re_test':loss_reconsctruction})
                s_AUROC,g_AUROC,input_AUROC=auroc(pred_results,A,pred_results_input,batch_label,batch_label_mask)
                s_MSE,g_MSE,input_MSE=mse(pred_results,A,pred_results_input,batch_label,batch_label_mask)
                s_AUROC_all_test+=s_AUROC
                g_AUROC_all_test+=g_AUROC
                in_AUROC_all_test+=input_AUROC
                s_AUROC=0
                g_AUROC=0
                input_AUROC=0
                
                s_MSE_all_test+=s_MSE
                g_MSE_all_test+=g_MSE
                in_MSE_all_test+=input_MSE
                s_MSE=0
                g_MSE=0
                input_MSE=0
                # distance=consistency(X_rec,causal_graph,batch_label_mask)
                # distance_all_test+=distance
                # distance=0
            else:
                logger.info('VALID# fold: {}, epoch: {}, iter: {},  loss_high_level: {},  loss_KL: {},  loss_re:{}'. \
                                format(fold,   epoch,    train_step, loss_high_level,     loss_KL,     loss_reconsctruction))
        
                wandb.log({'epoch': epoch,  'evalstep':evalstep+len(dataloader)*epoch,'loss_all_valid':loss,'loss_high_level_valid':loss_high_level,'loss_KL_valid':loss_KL,'loss_re_valid':loss_reconsctruction})
                s_AUROC,g_AUROC,input_AUROC=auroc(pred_results,A,pred_results_input,batch_label,batch_label_mask)
                s_MSE,g_MSE,input_MSE=mse(pred_results,A,pred_results_input,batch_label,batch_label_mask)
                s_AUROC_all_eval+=s_AUROC
                g_AUROC_all_eval+=g_AUROC
                in_AUROC_all_eval+=input_AUROC
                s_AUROC=0
                g_AUROC=0
                input_AUROC=0
                
                s_MSE_all_eval+=s_MSE
                g_MSE_all_eval+=g_MSE
                in_MSE_all_eval+=input_MSE
                s_MSE=0
                g_MSE=0
                input_MSE=0
                # distance=consistency(X_rec,causal_graph,batch_label_mask)
                # distance_all_eval+=distance
                # distance=0
    if train:
        s_AUROC_all_train=s_AUROC_all_train/trainstep
        g_AUROC_all_train=g_AUROC_all_train/trainstep
        in_AUROC_all_train=in_AUROC_all_train/trainstep
        s_MSE_all_train=s_MSE_all_train/trainstep
        g_MSE_all_train=g_MSE_all_train/trainstep
        in_MSE_all_train=in_MSE_all_train/trainstep
        distance_all_train=distance_all_train/trainstep
        return s_AUROC_all_train,g_AUROC_all_train,in_AUROC_all_train,s_MSE_all_train,g_MSE_all_train,in_MSE_all_train,distance_all_train
    elif len(dataloader)==valid_sample:
            s_AUROC_all_eval=s_AUROC_all_eval/evalstep
            g_AUROC_all_eval=g_AUROC_all_eval/evalstep
            in_AUROC_all_eval=in_AUROC_all_eval/evalstep
            s_MSE_all_eval=s_MSE_all_eval/evalstep
            g_MSE_all_eval=g_MSE_all_eval/evalstep
            in_MSE_all_eval=in_MSE_all_eval/evalstep
            distance_all_eval=distance_all_eval/evalstep
            return s_AUROC_all_eval,g_AUROC_all_eval,in_AUROC_all_eval,s_MSE_all_eval,g_MSE_all_eval,in_MSE_all_eval,distance_all_eval
    else: 
            s_AUROC_all_test=s_AUROC_all_test/teststep
            g_AUROC_all_test=g_AUROC_all_test/teststep
            in_AUROC_all_test=in_AUROC_all_test/teststep
            s_MSE_all_test=s_MSE_all_test/teststep
            g_MSE_all_test=g_MSE_all_test/teststep
            in_MSE_all_test=in_MSE_all_test/teststep
            distance_all_test=distance_all_test/teststep
            return s_AUROC_all_test,g_AUROC_all_test,in_AUROC_all_test,s_MSE_all_test,g_MSE_all_test,in_MSE_all_test,distance_all_test