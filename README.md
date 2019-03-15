# Master Thesis repository: Machine learning under resource constraints
## Under the supervision of Prof. Pierre Geurts and Jean-Michel Begon, ULiège 2019

### Objectives:
The objective of this master thesis is to explore the different techniques used to reduce the resource consumptions of CNNs at inference. 
Due to GPU-ressource limitations, we limited ourselves to use the CIFAR-10 dataset in order to compare the different networks.
The techniques investigated were:
    1. Base CNN architecture (WideResnet, DenseNet, EffNet, CondenseNet, MobileNet,...) and their related parameters (depth, width, bottlenecks,...)
    2. Number of channels per layers (fisher pruning and morphnet)
    
    
### Repository structure:
	- pytorch-prunes: modification of the code of the paper "[Pruning neural networks: is it time to nip it in the bud?][https://arxiv.org/abs/1810.04622]"
	- morph_net: modification of the code of the paper "[MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks][http://arxiv.org/abs/1711.06798]"
	- NetAdapt: self-made implementation of the paper "[NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications][http://arxiv.org/abs/1804.03230]"