ImageNet2012 Classification
========
* Train and test ImageNet2012 dataset using pytorch.    
* Adapted from [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).


### Download ImageNet2012
```
$ wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
$ wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
$ wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar
```
* move validation images to labeled subfolders use the following [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).
* no available labels for test dataset, thus evaluation would be conducted on val dataset.

### Training detail
* SGD optimizer with momentum (0.9) and weight decay (5.0e-4)
* Initial learning rate (0.1) and batch size (64)
* Training 90 epochs, learning rate decay exponentially by 0.1X per 30 epochs
