function  outputIDNew=FaceRecognitionNew(trainImgSet, trainPersonID, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainImgSet: contains all the given training face images
%    trainPersonID: indicate the corresponding ID of all the training images
%    testImgSet: constains all the test face images
%    outputID - predicted ID for all tested images 

%% Pretrained network link : 
%https://uniofnottm-my.sharepoint.com/:f:/g/personal/hcyja1_nottingham_ac_uk/Ek6G4HdUV3JPotWsSQY30LEB5POH2Z9gYrMSrnXgkqOqjA?e=nIj6dv
protofile = '.\vgg_face_caffe\VGG_FACE_deploy.prototxt';
datafile = '.\vgg_face_caffe\VGG_FACE.caffemodel';
net = importCaffeNetwork(protofile,datafile); %Import pretrained models
analyzeNetwork(net)
layer_fc7 = 'fc7'; %layer before FCN
inputSize = net.Layers(1).InputSize; %Size of input layer

%% Extract features from the training images: 
trainImgSet = imresize(trainImgSet,[inputSize(1:2)]); %Resize to CNN input size 
trainTmpSet = zeros(inputSize(1),inputSize(2),3,size(trainImgSet,4));
%% Viola jones
violj_detect = vision.CascadeObjectDetector;
violj_detect.MergeThreshold = 8;

%Normalization 
for i=1:size(trainImgSet,4)
    tmpI= rgb2gray(trainImgSet(:,:,:,i));
    gray_3d = cat(3,tmpI,tmpI,tmpI);
   % imshow(tmpI);
    %bboxes_train = violj_detect(gray_3d);
    %if(~isempty(bboxes_train))
    %    gray_3d = imcrop(gray_3d,bboxes_train(1,:));
    %    gray_3d = imresize(gray_3d,[inputSize(1:2)]);
        %imshow(gray_3d)
    %else
    %    gray_3d=gray_3d;
    %end
    tmpI = double(gray_3d)/255;
    trainTmpSet(:,:,:,i) = tmpI; 
end
featuresTrain = activations(net,trainTmpSet,layer_fc7,OutputAs="rows"); %Extract features 

%% Using SVM 
%t = templateSVM('KernelFunction','gaussian');
%options = statset('UseParallel',true);
%classifier = fitcecoc(featuresTrain,trainPersonID,'Coding','onevsone','Learners',t,'Prior','uniform','Options',options);

%% Face recognition for all the test images

outputIDNew=[];
testImgNames=dir([testPath,'*.jpg']);

for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:).name]);%load one of the test images
    % perform the same pre-processing as the training images
    tmpI = imresize(testImg,[inputSize(1:2)]);
    tmpI = rgb2gray(tmpI);
    tmpI = cat(3,tmpI,tmpI,tmpI);
    tmpI = double(tmpI)/255;
    release(violj_detect);
    %bboxes_test = violj_detect(tmpI);
    %if(~isempty(bboxes_test))
    %   tmpI = imcrop(tmpI,bboxes_test(1,:));
    %    tmpI = imresize(tmpI,[inputSize(1:2)]);
    %else
    %     tmpI = tmpI;
    %end
    featuresTest = activations(net,tmpI,layer_fc7,OutputAs="rows"); %Extract features
    
    %% Prediction using siamese network
    distance_layer = pdist2(featuresTest,featuresTrain,'euclidean');
    distance_dlarr = dlarray(distance_layer,'SSCB');
    activation_layer = sigmoid(distance_dlarr);
    [maxVal,yPred] = min(activation_layer);
    yPred = trainPersonID(yPred(1),:);
    disp(yPred)
    %% Prediction using SVM 
    %yPred=predict(classifier,featuresTest);  % Use SVM Classifier to predict 
    
    outputIDNew=[outputIDNew; yPred];   % store the ID for each of the test images
end




