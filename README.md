# One_Class_MNIST

This is an exporation of classifying the 0 images in MNIST using only the 0 images as training data

Much of the idea came from this paper: https://arxiv.org/pdf/1801.05365.pdf (Learning Deep Features for One-Class Classification by Pramuditha Perera etal.)

The objective to prepare the reference network is switched to a autoencoder. The motivation behind this is that the tasks which require a one class classification approach often have much difficulty acquiring data of other classes. 

Due to the amount of data available, the pretraining of the autoencoder did not improve on the performance on classification task. 

Additionally, as opposed to reducing the variance of g, the variance of the last layer of the encoder is reduced instead.

Oh, and create dc_img folder if you want to make the autoencoder. The autoencoder model is taken from https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
