# Our Papers
Pytorch code for Towards-Causal-Relationship-in-Indefinite-Data-Baseline-Model-and-New-Datasets

## New Datasets
Causalogue: 
A text dataset with 1638 dialogue samples, labeled with full causal relationships.   
For the representation, we recommend the pre-trained model RoBERTa (https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)

Causaction: 
A video dataset with 1118 video samples, labeled with full causal relationships between two segments.   
For the representation, we recommend the representation from I3D (https://zenodo.org/records/3625992#.Xiv9jGhKhPY) 

In this repo， you can use follows to run the baseline model with two new datasets：  
```
python main.py --config
```
To access the new datasets, you can look into: 
```
\data\causaction\breakfast2.json
\data\causalogue\all_data_small.json
```
To load these two new datasets, you can: 
```
from data_loader import *

train_loader=build_train_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='train',args=args,config=dataset_config)
valid_loader = build_inference_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='valid',args=args,config=dataset_config)
test_loader = build_inference_data(datasetname=args.dataset,fold_id=fold_id,batch_size=wandb.config.batch_size,data_type='test',args=args,config=dataset_config)

```


# Reference 
@misc{chen2023causal,  
      title={Towards Causal Representation Learning and Deconfounding from Indefinite Data},   
      author={Hang Chen and Xinyu Yang and Qing Yang},  
      year={2023},  
      eprint={2305.02640},  
      archivePrefix={arXiv},  
      primaryClass={cs.LG}  
}
