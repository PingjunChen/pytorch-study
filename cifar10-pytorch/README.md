cifar10-pytorch
========
* Train and test *convnets* on *cifar10* dataset using pytorch.  
* Adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| AlexNet           | 78.06%      |
| VGG16             | 93.93%      |
| VGG19             | 93.69%      |
| ResNet50          | 95.21%      |
| ResNet101         | 95.42%      |
| DenseNet121       | 95.13%      |
| DenseNet201       | 94.90%      |


## Training detail
* SGD optimizer with momentum (0.9) and weight decay (5.0e-4)
* Initial learning rate (0.1) and batch size (64)
* Training 300 epochs, learning rate decay exponentially by 0.1X per 100 epochs

## Todo
* AlexNet using default training settings gets low classification accuracy. Adjustment is needed.
