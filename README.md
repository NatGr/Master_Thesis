# Master Thesis repository: Machine learning under resource constraints
## Under the supervision of Prof. Pierre Geurts and Jean-Michel Begon, ULiège 2019

### Objectives:
The objective of this master thesis is to explore the different techniques used to reduce the resource consumptions of CNNs at inference. 
Due to GPU-ressource limitations, we limited ourselves to use the CIFAR-10 dataset in order to compare the different networks.
The techniques investigated were:
    1. Base CNN architecture (WideResnet, ShuffleNet, EffNet, CondenseNet, MobileNet,...) and their related parameters (depth, width, bottlenecks,...)
    2. Number of channels per layers (fisher pruning, morphnet and NetAdapt algorithms)
    
    
### Repository structure:
	- morph_net: modification of the code of the paper "[MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks][http://arxiv.org/abs/1711.06798]"
	- NetAdapt: self-made implementation of the paper "[NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications][http://arxiv.org/abs/1804.03230]", several modifications were made (the main ones are: fisher pruning instead of weights-norm pruning and training from scratch rather than long term fine-tuning)
	- pytorch-prunes: modification of the code of the paper "[Pruning neural networks: is it time to nip it in the bud?][https://arxiv.org/abs/1810.04622]"
	- schemes: a folder containing the .odf file I used to create custom schemes to illustrate the different neural networks architectures
	- training_from_scratch: a repo containing the code that trains neural networks in tensorflow to then load them as .tflite models on the rasberry-pi
	- The cross-compiled tflite 1.13.1 binary as well as the "benchmark" binary that are used for benchmarking.