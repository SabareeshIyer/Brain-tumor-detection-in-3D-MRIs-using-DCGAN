# Brain tumor detection in 3D MRIs using DCGAN
## Implementing DCGAN in Tensorflow to perform tumor segmentation in brain image scans

Semantic segmentation constitutes an integral part of medical image analysis for which breakthroughs in the field of deep learning were of high relevance. Attributing the pixels of an input image to a certain category is an important and well-studied problem in computer vision.
Currently, the most frequently used approach to address the visual attribution problem is training a neural network classifier to predict the categories of a set of images and then following one of two strategies: analysing the gradients of the prediction with respect to an input image or analysing the activations of the feature maps for the image to determine which part of the image was responsible for making the associated prediction. 

The approach that I am using here, called [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) attempts to segment images by training [Generative Adversarial Networks (GANs)](https://arxiv.org/pdf/1406.2661) and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks. The learning process and lack of heuristic cost function in GANs make them attractive for representation learning.

### Introduction
Manual segmentation of lesions/tumors from Magnetic Resonances Images (MRIs) is a time-sink for medical professionals whose time can be better put to use in non-repetitive tasks. Further, it is tedious and cumbersome for a large number of images. Judging by the repetitive nature of the task with no special challenges encountered with every new image, it would appear that such an activity could be automated; but the intricacies involved in the task make it extremely difficult to codify a certain set of rules by which to do so. However, in the last few years, deep learning has provided many a solution that was previously beyond our reach.
Here, I use [DCGANs](https://arxiv.org/abs/1511.06434.pdf) for segmentation of medical images.

### Model Architecture
I have modified the architecture in the DCGAN paper to fit the needs of the project.
![dcgan arch](https://user-images.githubusercontent.com/20294710/40440237-84af0c46-5e8b-11e8-9edf-548e1df365ef.PNG)

As seen in the figure, the compressed flair volume of brain multimodal scan images is provided as input to generator, which in turn generates a 28x28 segmented data that is fed to the discriminator. The discriminator also takes the annotated segmented data i.e.
groundtruth and tries to identify which is real and generated one. The discriminator optimizes itself to correctly distinguish between the two. The generator is also optimized according to the output of the discriminator and thus, tries to generate a segmented image
that closely resembles the groundtruth, thus, tries to fool the discriminator. Once the model is trained, we can feed the generator with test data and obtain its segmented result. 

#### Discriminator Network
![image](https://user-images.githubusercontent.com/20294710/40440366-daa24af0-5e8b-11e8-8214-5eaa02f58271.png)
The discriminator is the “art critic” who tries to distinguish between real and generated segmented images. This is a convolutional neural network for image classification. The discriminator network consists of three convolutional layers. For every layer of the network, we are going to perform a convolution, then we are going to perform batch normalization to make the network faster and more accurate, and, finally, we are going to perform a Leaky ReLu to further speed up the training. At the end, we flatten the output of the last layer and use the sigmoid activation function to get a classification.

#### Generator Network
![image](https://user-images.githubusercontent.com/20294710/40440453-1c40de2c-5e8c-11e8-8cbb-49d06c675c28.png)
The generator goes the other way: It is the “artist” who is trying to fool the discriminator. The generator makes use of deconvolutional layers. They are the exact opposite of a convolutional layers: Instead of performing convolutions until the image is transformed
into simple numerical data, such as a classification, we perform deconvolutions to transform numerical data into an image. In this case, instead of feeding a random noise vector to the generator network, we are feeding the brain multimodal scan volumes as
input. First, we take our input, called Z which is the Flair data, and feed it into our first deconvolutional layer. Each deconvolutional layer performs a deconvolution and then performs batch normalization and a leaky ReLu as well. Then, we return the tanh activation function.

### Experiment
The objective being medical image segmentation, I have used the BRATS-2 dataset for the project. The dataset comprises of clinically-acquired 3T multimodal MRI scans and all the ground truth labels have been manually-revised by expert board-certified
neuroradiologists.

For the purposes of training the GAN, we use all the training images from HG and LG resized from 256 x 256 to 28 x 28 by cropping and resizing the area needed for segmentation.

