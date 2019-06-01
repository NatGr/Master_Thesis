# My implementation of [NetAdapt](http://arxiv.org/abs/1804.03230)

This code is based on pytorch (version 1.0.1). 
However, since pytorch does not handle 32bits architectures and that the goal I had in mind when implementing the paper was to do efficient inference on a rasberry-pi, 
I used tensorflow lite to compute the tables instead (compute_table.py). 
This means that two environments (one with pytorch and scikit learn and one with tensorflow 1.13.1 (or one environment with both) are needed to run the project.
The tensorflow environment used on the rasberry-pi comes from [this page](https://github.com/PINTO0309/Tensorflow-bin)
Initially I cross compiled it myself but I suffered several bugs doing so that the author of the previous repo corrected.
I crosscompiled the tf-lite binaries used on the raspberry-pi. They are available in the root repository.

First, the base network needs to be built, 
```
python train.py --net='res' --depth=40 --width=2.0 --save_file='res-40-2'

```
would train a base wideresnet-40-2 for example (assuming CIFAR-10 location is in ~/Documents/CIFAR-10)

Then, the network can be pruned, for example
```
python prune.py --save_file='res-40-2-pruned' --pruning_fact=0.4 --base_model='res-40-2' --perf_table='res-40-2'

```
would prune 40% of the res-40-2 network using res-40-2 (assumed to lay in folder perf_tables) as a performance table

To build a table, one would need to use
```
python compute_table.py --save_file='res-40-2-tf-lite-2-times' --eval_method='tf-lite-2-times' --mode='load' --output_folder='/media/pi/Elements/models'

```
on the targeted device, compute_table uses a res-40-2 by default


Unfortunately, the network objects used in this script are non trivial to implement (see model/wideresnet.py) for example.