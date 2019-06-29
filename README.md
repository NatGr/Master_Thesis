# Master Thesis repository: Machine learning under resource constraints
## Under the supervision of Prof. Pierre Geurts and Jean-Michel Begon, ULiège 2019

### Objectives:
The objective of this master thesis is to explore the different techniques used to reduce the resource consumptions of CNNs at inference.   
Due to GPU-ressource limitations, we limited ourselves to use the CIFAR-10 dataset in order to compare the different networks.
The techniques investigated were:  
1.  Base CNN architecture (WideResnet, ShuffleNets, EffNet, CondenseNet, MobileNets,...) and their related parameters (depth, width, bottlenecks,...)  
1.  We tried to use pruning as an architecture search algorithm, i.e. starting from a wider network, using some pruning algorithm to prune certain channels, obtaining a thinner network and retraining it from scratch.  
    We tested the NetAdapt, MorphNet and Fisher pruning. Interestingly, NetAdapt was later used for the same purpose in MobileNetsv3 (that where not public when we started this thesis).   
    Unfortunately, all the algorithms we tested did not show better results than mere random pruning on a WideResNet-40-2 on CIFAR-10 using a raspberry-pi 3B to perform inference. Wether it is also the case for other architectures/hardwares/datasets remains to be determined.  
1.  We tried to use knowledge distillation, this did show interesting improvements but only for MobileNetv1 (We don't know why -- seems like something worth exploring in details). We also tried to use tensorflow's built-in quantization capabilities and managed to use them on MobileNetv1/v2 without SE blocks. We could only make posttraining quantization work and this was quite tricky. For the moment, quantization is not so well integrated into tensorflow. Posttraining quantization results were not integrated to the report since tensorflow released posttraining quantization between the report and presentation deadlines. See last slide of the final presentation for results.

    
### Repository structure:
	- CondenseNet: clone of https://github.com/ShichenLiu/CondenseNet with some minor modifications to make it run
	- morph_net: modification of the code of the paper "[MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks][http://arxiv.org/abs/1711.06798]"
	- NetAdapt: self-made implementation of the paper "[NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications][http://arxiv.org/abs/1804.03230]", several modifications were made (the main ones are: fisher pruning instead of weights-norm pruning and training from scratch rather than long term fine-tuning)
	- pytorch-prunes: modification of the code of the paper "[Pruning neural networks: is it time to nip it in the bud?][https://arxiv.org/abs/1810.04622]"
	- schemes: a folder containing the .odf file I used to create custom schemes to illustrate the different neural networks architectures
	- training_from_scratch: a repo containing the code that trains neural networks in tensorflow to then load them as .tflite models on the rasberry-pi. This repo also handles knowledge distillation and quantization.
	- The cross-compiled tflite binary as well as the "benchmark" binary that are used for benchmarking. Both for tf 1.13.1 and tf-nightly
