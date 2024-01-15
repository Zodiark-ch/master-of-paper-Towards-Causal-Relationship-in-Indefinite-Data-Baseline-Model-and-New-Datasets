import torch
from alldatasets.reccon import RecconDataset
from alldatasets.gptdialog import GPTDialogDataset
from alldatasets.videoaction import VideoDataset

def build_train_data(datasetname,fold_id, batch_size,data_type,args,config,shuffle=True):
    if datasetname=='dailydialog':
        train_dataset = RecconDataset(fold_id, data_type=data_type,args=args,config=config)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                collate_fn=train_dataset.collate_fn,shuffle=shuffle)
        return train_loader
    
    if datasetname=='Causalogue':
        train_dataset = GPTDialogDataset(fold_id, data_type=data_type,args=args,config=config)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                collate_fn=train_dataset.collate_fn,shuffle=shuffle)
        return train_loader
    
    if datasetname=='Causaction':
        train_dataset = VideoDataset(fold_id, data_type=data_type,args=args,config=config)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                collate_fn=train_dataset.collate_fn,shuffle=shuffle)
        return train_loader

def build_inference_data(datasetname,fold_id, batch_size,args,config,data_type):
    if datasetname=='dailydialog':
        inference_dataset = RecconDataset( fold_id, data_type=data_type,args=args,config=config)
        data_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=batch_size,
                                                collate_fn=inference_dataset.collate_fn,shuffle=False)
        return data_loader
    
    if datasetname=='Causalogue':
        inference_dataset = GPTDialogDataset( fold_id, data_type=data_type,args=args,config=config)
        data_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=batch_size,
                                                collate_fn=inference_dataset.collate_fn,shuffle=False)
        return data_loader
    
    if datasetname=='Causaction':
        inference_dataset = VideoDataset( fold_id, data_type=data_type,args=args,config=config)
        data_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=batch_size,
                                                collate_fn=inference_dataset.collate_fn,shuffle=False)
        return data_loader