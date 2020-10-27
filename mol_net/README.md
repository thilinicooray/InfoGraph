# GraphSAD: Learning Graph Representations with Structure-Attribute Disentanglement

## Introduction

This project is an implementation of ``GraphSAD: Learning Graph Representations with Structure-Attribute Disentanglement'' in PyTorch. 
We provide the codes for Input-SAD, Embed-SAD and SAD-Metric evaluation. 

### Prerequisites

We develop this project with `Python3.6` and  following Python packages:

```
Pytorch                   1.5.0
torch-cluster             1.5.5                    
torch-geometric           1.5.0                   
torch-scatter             2.0.5                    
torch-sparse              0.6.5                    
torch-spline-conv         1.2.0 
rdkit                     2020.03.1
```
**P.S.** In our project, these packages can be successfully installed and work together under `CUDA/9.0` and `cuDNN/7.0.5`.


### Datasets

For graph classification tasks on MoleculeNet, you can download the datasets in [this pervious work](http://snap.stanford.edu/gnn-pretrain).

For the datasets of node classification, they will be automatically downloaded by torch-geometric.

**P.S.** These datasets are suggested to be stored under the same path, e.g. `./data/`.

### Graph Classification

**Baseline.** To train the baseline model, simply run:
```
python train_baseline.py --dataset $dataset_name$ --filename $save_file$ \
                         --gnn_type $GNN_name$ --eval_train
```

**Input-SAD.** To train the Input-SAD model, simply run:
```
python train_input_SAD.py --dataset $dataset_name$ --filename $save_file$ \
                          --gnn_type $GNN_name$ --eval_train 
```

**Embed-SAD.** To train the Embed-SAD model, simply run:
```
python train_embed_SAD.py --dataset $dataset_name$ --filename $save_file$ \
                          --gnn_type $GNN_name$ --eval_train 
```

## Node Classification

**Baseline.** To train the baseline model, simply run:
```
python multiple_runs.py --gpu_id $device$ --dataset $dataset_name$ --filename $save_file$ \
                        --result_file $save_result$ --num_run $default: 100$
```

**Input-SAD.** To train the Input-SAD model, simply run:
```
python multiple_runs.py --gpu_id $device$ --dataset $dataset_name$ --filename $save_file$ \
                        --result_file $save_result$ --num_run $default: 100$ --use_input_disen 
```

**Embed-SAD.** To train the Embed-SAD model, simply run:
```
python multiple_runs.py --gpu_id $device$ --dataset $dataset_name$ --filename $save_file$ \
                        --result_file $save_result$ --num_run $default: 100$ --use_embed_disen 
```

## SAD-Metric Evaluation

We provide the evaluation procedure for Embed-SAD model, which contains two steps.

**First step.** Train and save the model on a graph classification dataset:
```
python train_embed_SAD.py --dataset $dataset_name$ --filename $save_file$ \
                          --gnn_type $GNN_name$ --eval_train --output_model_file $model_name$
```

**Second step.** Evaluating the SAD-Metric based on an additional classifier:

```
python SAD_metric.py --dataset $dataset_name$ --gnn_type $GNN_name$ \
--eval_train --output_model_file $model_name$
```