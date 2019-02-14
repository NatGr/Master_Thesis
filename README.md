# Master Thesis repository: Machine learning under resource constraints
## Under the supervision of Prof. Pierre Geurts and Jean-Michel Begon, ULiège 2019

### Objectives:
The objective of this master thesis is to explore the different techniques used to reduce the resource consumptions of CNNs at inference. 
Due to GPU-ressource limitations, we limited ourselves to use the CIFAR-10 dataset in order to compare the different networks.
The techniques investigated were:
    1. Base CNN architecture (WideResnet, DenseNet) and their related parameters (depth, width, bottlenecks,...)
    2. Number of channels per layers
    
    
### Repository structure:
	- pytorch-prunes: modification of the code of the paper "[Pruning neural networks: is it time to nip it in the bud?][https://arxiv.org/abs/1810.04622]"