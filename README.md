# Learning to Augment (L2A)


This is a PyTorch implementation of the Learning to Augment (L2A), and the code includes the following modules:

* Dataset Loader (Cora, Citeseer, BlagCatalog, Texas, Cornell, Wisconsin, Actor, and Syn-Cora)

* Various Architectures (GCN, SAGE, GAT, and GNN Classifier used in this paper)

* Training paradigm pre-training and fine-tuning on 8 datasets

* Visualization and evaluation metrics 

  

## Main Requirements

* networkx==2.5
* numpy==1.19.2
* scipy==1.5.2
* torch==1.6.0
* pyro_ppl==1.3.0



## Description

* main.py  
  * pretrain_EdgePredictor() -- Pretrain Graph Augmentor
  * pretrain_Classifier() -- Pretrain GNN Classifier
  * main() -- Train the model for node classification task on the *Cora, Citeseer, BlagCatalog, Texas, Cornell, Wisconsin, Actor, and Syn-Cora* datasets
* model.py  
  
  * GCNLayer() -- GCN Layer
  * SageConv() -- SAGE Layer
  * GATLayer() -- GAT Layer
  * EdgePredictor() -- Learn parameterized augmentation distribution
  * EdgeSampler() -- Perform gumbel-softmax sampling
  * EdgeLearning() -- Learn weighted graph
  * Classifier() -- Classify nodes based on the learned weighted graph
* graphSSL.py  
  * DistanceCluster() -- Perform self-supervised Global-Path Prediction
  * ContextLabel() -- Perform self-supervised Local Label Distribution Preservation
* dataset.py  

  * load_data() -- Load synthetic and real-world datasets
* utils.py  
  * evaluation() -- Calculate classification accuracy



## Running the code

1. Install the required dependency packages

3. To get the results on a specific *dataset* with a specific *GNN architecture*, please run with proper hyperparameters:

  ```
python main.py --dataset data_name --model architecture
  ```

where the *data_name* is one of the 8 datasets (Cora, Citeseer, BlagCatalog, Texas, Cornell, Wisconsin, Actor, and Syn-Cora) and *architecture* is one of the 4 GNN architectures (GCN, SAGE, GAT, and GNN Classifier used in this paper). Use *GCN* on the *Cora* dataset an example: 

```
python main.py --dataset cora --model GCN
```



## Acknowledgement

This project borrows the architecture design and part of the code from [GAUG](https://github.com/zhao-tong/GAug).



## License

Learning to Augment (L2A) is released under the MIT license.