# training_from_scratch

This repo contains code to train neural networks in tensorflow 1.13.1 and then save them in .tflite models to perform inference on a rasberry-pi.  

Due to the poor image augmentation capabilities of tensorflow.keras, the [albumentations library](https://github.com/albu/albumentations) was used as well.

The main file here is train_save.py that trains a network and saves it as a .tflite file (if --get_tf_lite flag is set). It contains many different networks (effnet, squeezenext, wideresnet, shufflenetv1/2, mobilenetv1/2, mnasnet, nasnet) as well as parameters to run each of those. Among those, Squeeze and exitation layers can be added to wideresnets, mnasnets, shufflenetv2 and mobilenets.  
These networks can be trained with different optimizers (adam, rmsprop, multistep SGD, cosine SGD).   
The code allows to use nowledge distillation (the predictions of the teacher are memorized in a ~200MB numpy file).  
You can have a look at the flags of train\_save.py for more details. To give a few examples:
```
python train_save.py --save_file=file --lr_type='cosine' --weight_decay=1e-4 --net='mobilenetv2' --depth=12 --expansion_rate=6 --width=24 --use_dropout --se_factor=2 --get_tf_lite --train_val_set

```
Trains a mobilenetv2 of 12 blocks (depths indicates number of blocks and not of layers except for wideresnets) width an expansion rate of 6 and a width of [24 (when tensors are 32\*32), 48 (when tensors are 16\*16), 96 (when tensors are 8\*8)]. Dopout is used in the last layer and SE blocks are added at the end of each mobilenetv2 block (when no stride is applied). At the end of the training, a tf-lite file is produced.  

```
python train_save.py --save_file=res_16 --lr_type='cosine' --weight_decay=1e-4 --net='resnet' --depth=10 --width=16 --save_pred_table

python train_save.py --save_file=res_10 --lr_type='cosine' --weight_decay=1e-4 --net='resnet' --depth=10 --width=16 --save_pred_table --lambda_KD=0.2 --temperature=2 --teacher_pred='res_16'
```
First trains a resnet of depth 16 and saves its predictions. These are later used to train a resnet of depth 10 with knowledge distillation (weight of 0.2 for the classical crossentropy loss and temperature of 2).  


quantize\_model is used to quantize a pretrained model (both weights and activations). At the time I'm writing this, tf-nighlty is needed to run this code, it however crashes after having compiled the tf-lite file. The tf-lite file compiled is extremy suspicious (heavyer than the base one and much slower) so it's of no use for the moment.
posttrain_quantize can be used to quantize a saved keras model (automatically produced after training when the "--get_tf_lite" flag is set). It needs tf-nightly as well. It however does not work with SE blocks, due to some bugs in tensorflow-nightly that will probably be solved in the near future. The quantized files do not show so good results, they are however promising enough so that we think train-time quantization to be promizing, once it will be available in tensorflow. We need the tf-lite binaries associated with tf-nightly to run. Those use a (at the time of writing this) very recent version of glibc and we thus had to install unstable packages [as shown here](https://raspberrypi.stackexchange.com/a/24032)

benchmarks\_tf\_lite\_file.sh is a bash script (I should have done it in python, sorry :( ) used to run tflite files and compute inference time on the raspberry-pi (the path of the folder containing the binaries is harcoded dor my raspberry-pi) 