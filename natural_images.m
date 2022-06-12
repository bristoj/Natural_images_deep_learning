% Bristo Joemon 12/06/2022
% Create a neural network in MATLAB

close all
% Close the workspace
clear
% Clear the command wondow
clc

%%
% data was saved in MATLAB drive
% the path to the data is saved below
imdspath = fullfile(matlabdrive, 'natural_images');

%%
% each class has one subfolder with the class name as the foldername
% below code helps to segregate the images with the foldername for training and testing purpose 
imageds = imageDatastore(imdspath, 'IncludeSubfolders', true,'LabelSource','foldernames')

%%
% the below code helps in creating a sample collage of images in the given
% data
figure;
perm = randperm(100,20); for i = 1:10
subplot(5,2,i); imshow(imageds.Files{perm(i)});
end

%%
% To count the number of images in each class
labelCount = countEachLabel(imageds)

%%
img = readimage(imageds,520); size(img)

%%
% Spliting the data into Training and testing
% 491 images are taken for training
numTest=491;
[imdsTrain,imdsValidation] = splitEachLabel(imageds,numTest,'randomize');

%%
% several classes had varrying number of images
% below code is used to split the testing data into test and validation
% with 105 images in validation for each class
numTest=105;
[imdsValidation2,imdsTest] = splitEachLabel(imdsValidation,numTest,'randomize');

%%
% the neural network has following layers
layers = [
imageInputLayer([50 100 3])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
% swish layer gave the best accuracy
% other layers are kept for future references
%leakyReluLayer
%reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
%leakyReluLayer
%reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
%leakyReluLayer
%reluLayer
fullyConnectedLayer(8)
% there are 8 classes, hence final layer has 8 output
softmaxLayer
classificationLayer];
%%
% augmentation is used to bring uniformity to the images. With this all
% images becomes the size 50*100*3 since they are color images
imdsValidation3 = augmentedImageDatastore([50 100],imdsValidation2)

%%
% training puropse sgdm gave the best output
options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',20, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation3, ...
 'ValidationFrequency',8, ...
 'Verbose',false, ...
 'Plots','training-progress');

%%
% to avoid overfitting the below augmenter is used
imageAugmenter = imageDataAugmenter('RandXReflection',1, ...
    ...'RandYReflection',1,...
    ...'FillValue',[200 200 3],...
    ...'RandXShear',[-90,90],...
    ...'RandYShear',[-90,90],...
    ...'RandScale',[0.6 1.2],...
    'RandRotation',[-20,20],...
    'RandXTranslation',[-11 11]...
    ...'RandYTranslation',[-11 11]...
    )

%%
%auds = augmentedImageDatastore([100 100],imdsTrain)
auds = augmentedImageDatastore([50 100],imdsTrain, 'DataAugmentation',imageAugmenter);
%%
% training the network
net = trainNetwork(auds,layers,options);

%%
% doing prediciton with the trained model
YPred = classify(net,imdsValidation3);
YValidation = imdsValidation2.Labels;
accuracyvalidation = sum(YPred == YValidation)/numel(YValidation);

%%
display(accuracyvalidation)

%%
YPred1 = classify(net,auds);
YValidation1 = imdsTrain.Labels;
accuracytrain = sum(YPred1 == YValidation1)/numel(YValidation1);

%%
display(accuracytrain)

%%
confusionchart(YValidation,YPred)

%%
imdsTest2 = augmentedImageDatastore([50 100],imdsTest)
%%
YPred2 = classify(net,imdsTest2);
YValidation2 = imdsTest.Labels;
accuracyTest = sum(YPred2 == YValidation2)/numel(YValidation2);

%%

display(accuracyTest)

%%
confusionchart(YValidation2,YPred2)