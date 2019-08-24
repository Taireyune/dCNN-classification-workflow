# dCNN-classification-workflow
dCNN-classification-workflow encompass my journey to learn image classification
using deep convolutional neural networks. Elements of this is been used as a 
discriminator in my current project.

## Image augmentation 
Although keras library contains the basic image augmentation tools,
[I wrote one myself](https://github.com/Taireyune/dCNN-classification-workflow/blob/master/basic_workflow/ImageAugmentation.py)
to better understand the image manipulations and parallel processing performance.

## Model architecture
The [model](https://github.com/Taireyune/dCNN-classification-workflow/blob/master/basic_workflow/ModelArch.py)
contains elements of batch normalization, residual networks, inception networks,
and squeeze/excite. Note that in reality, training a network involve using
the simplest architecture that works or iterate with increasing complexity until
it works. Here I am just familiarizing with the available tools that have shown
to work in the literature.

### Graph
Here is a section of the graph using tensorboard.

<img 
src="https://github.com/Taireyune/dCNN-classification-workflow/blob/master/images/graph.png" 
width="600" height="800" alt="graph">

### Accuracy and loss
Performance of a simple cat and dog classification without optimizing the model.

<img 
src="https://github.com/Taireyune/dCNN-classification-workflow/blob/master/images/accuracy.png" 
width="600" height="395" alt="accuracy">

<img 
src="https://github.com/Taireyune/dCNN-classification-workflow/blob/master/images/loss.png" 
width="600" height="395" alt="loss">


## GPU and performance
The code runs in pip installed tf-gpu environment and in Nvidia docker 
tensorflow environment. With the Nvidia tensorflow docker, 
it is possible to use both fp32 and fp16.
