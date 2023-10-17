# MVP
Hello this is MVP. We directly use pre-training outputs for prompt tuning. You can tune the pretraining phrase by modifying "pretrain.py".

## cora
python node_clas.py --dataset Cora --task_num 10 --activation tanh
python linkpre.py --dataset Cora --aplha 0.1 --pretrain_lr 0.0005 --hidden_channels 256 --out_channels 128 --num_proj_hidden 256

## citeseer
python node_clas.py --dataset CiteSeer --activation tanh --task_num 10 --hidden_channels 128 --out_channels 128 --num_proj_hidden 128
python linkpre.py --dataset CiteSeer --activation tanh --pretrain_lr 0.001 --hidden_channels 128 --out_channels 128 --num_proj_hidden 128 

## pubmed
python node_clas.py --dataset PubMed --task_num 10  --activation tanh --hidden_channels 256 --out_channels 128 --num_proj_hidden 256 --delta 2
python linkpre.py --dataset PubMed --seed 0  --activation tanh --pretrain_lr 0.0005 --hidden_channels 256 --out_channels 128 --num_proj_hidden 256 

## Wisconsin
python node_clas.py --lr 0.01 --dataset wisconsin --seed 0 --alpha 0.5

## Cornell
python node_clas.py --dataset cornell --seed 0 --alpha 0.1 

## Chameleon
python node_clas.py  --lr 0.001 --dataset chameleon --seed 0 --alpha 0.01 

## Texas
python node_clas.py --lr 0.01 --dataset texas  --seed 0 --alpha 0.5

## PROTEINS 
python downstream.py --dataset PROTEINS --hidden-dim 16 --aplha 10 --act

## BZR
python downstream.py --dataset BZR --hidden-dim 16 --aplha 10

## COX2
python downstream.py --dataset COX2 --hidden-dim 32 --aplha 0.5 --l2

## ENZYMES
python downstream.py --dataset ENZYMES --hidden-dim 32 --aplha 0.0001 
