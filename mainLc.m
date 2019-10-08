clear;
load("Yeast_cold.mat");
validationTimes = 10;
[trainFeatures,trainLabels,testFeatures,testLabels] = holdOutValidation(features,labels,validationTimes);
resultLc = table;

for i = 1:validationTimes
    resultLc = [resultLc;ldlLc(trainFeatures{i},trainLabels{i},testFeatures{i},testLabels{i})];
end

clear i validationTimes S'%'1;