# Natural_images_deep_learning
Classification of natural images using deep learning
Dataset: https://www.kaggle.com/datasets/prasunroy/natural-images
The dataset was downloaded from the above-mentioned link. It was in a compressed format which was later extracted to obtain the images in the local PC. The extracted files were saved in individual subfolders with the file name of each individual object class. Thus, each class has one subfolder with the file name same as its class name. And all the subfolders were saved in a main folder named Natural_Images.

# Section1: Network Architecture
For training the network, Stochastic Gradient Descent with momentum was used in the model. The learning rate was set at 0.01 and number of epochs were set at 40 to get the maximum testing and validation accuracies. After every epoch the dataset was shuffled to improve the randomness of the data. For each iteration a minibatch of data is taken and the model is trained with these minibatches and the process is repeated until all the training data is used which is termed as an epoch. Splitting the data into smaller batches while training the model helps in decreasing the overfitting of the model.
The dataset was divided into three parts 70 percent for training 15 each for testing and validation. This was done looking at the class with least number of data points. The class dog has the least number of images which is 702. Hence 491 images were taken for training which is 70 percent of the data, 105 images were used for validation purpose and remaining 106 images were taken for the testing. Likewise for every other class we followed taking a similar number for the other classes for training and validation and remaining pictures were taken for the purpose of testing. Hence there will be variations in the number of datapoints in the testing set which is as per the variations in the number of images in the class. We took this approach to maintain a consistency in the number of training data in the model.
Image augmenter function in the MATLAB was used to boost the network performance and increase the accuracy as required by the project. We used imageDataAugmenter functions such as RandXReflection to make reflection of an input image along the X-axis randomly, RandRotation was used to rotate the images 20 degrees towards left or right of the image and, RandXTranslation is used to move the images upto 11 pixels left or right randomly. With the help of imageDataAugmenter more variations of the existing input images were obtained to get the best predictions by avoiding overfitting during the training period.
To feed the images to the network, augmentedImageDatastore function was used to create images of the size 50*100*3 to train the model.
Once all these were done to each individual image, we feed it into the network for training purpose. The inputs are of the dimension 50*100*3.
 
The first layer has 8, 3*3 filters to apply a convolution operation on the input image to make a feature map and a padding is given to the output to keep the dimensions equal to the input image. The batchNormalizationLayer is used to improve the learning speed of the network and to provide regularization to the network which will help in avoiding overfitting. 
 
Then swishLayer is added as the activation function to the network. Activation function adds the non-linearity to the model which in turn helps to learn and perform more complex tasks. 
 
Then the output is fed to a pooling layer, here max pooling is used with a window of 2 by 2 with a stride of 2. 
 
Then this output is fed to the next convolutional layer with 16 filters of the size 3 by 3 and the process is repeated and finally it is fed to a final convolutional layer with 32 filters of 3 by 3 size and the output after batch normalization and swishLayer output is flattened, which means a vector is formed and is the input for the fully connected layer. The fully connected layer is a feed forward neural network. Here, fully connected layer has 8 outputs and these outputs go through a softmaxLayer before the final classification is done.
The total epoch was kept at 40, an epoch is when a full training cycle is done using the entire training data. Training data is shuffled after every epoch. The accuracy of the model is monitored with help of validation data at every 10 iterations.

# Section 2: Results Obtained
Accuracies obtained:
epochs	Training	Validation	Testing
20	94.55%	91.43%	92.96%
40	96.44 %	92.62%	93.15%
The training accuracy is 96.44 percent after 40 epochs similarly we got 92.62 percent for the validation and 93.15 percent accuracy for the testing set. We received the required accuracy above 90 percent at 20 epochs but to improve the predictions we trained the model further and it has been explained further in the critical analysis part towards the end of this report.
