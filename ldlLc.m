function [result,preLabels] = ldlLc(trainFeatures,trainLabels,testFeatures,testLabels)
% use LC to get preLabels
item =eye(size(trainFeatures,2),size(trainLabels,2));
[weights,~] = lcLdlTrain(item,trainFeatures,trainLabels);
preLabels = bfgsPredict(weights,testFeatures);

% get the evaluating indicators
[row,col]=size(testLabels);
lcKlDistance = zeros(row,1);
lcEuclideanDistance = zeros(row,1);
lcMSE = zeros(row,1); 

for j=1: row
    lcKlDistance(j) = kldist(testLabels(j,:), preLabels(j,:));
    lcEuclideanDistance(j) = norm(testLabels(j,:) - preLabels(j,:));
    lcMSE(j) = sum((testLabels(j,:)- preLabels(j,:)).^2)/col;
end

lcMeanKlDistance = mean(lcKlDistance);
lcMeanEuclideanDistance = mean(lcEuclideanDistance);
lcMeanMSE = mean(lcMSE);
lcSortLoss = sortLoss(testLabels,preLabels);

lcChebyshev = chebyshev(testLabels,preLabels);
lcClark = clark(testLabels,preLabels);
lcCanberra = canberra(testLabels,preLabels);
lcCosine = cosine(testLabels,preLabels);
lcIntersection = intersection(testLabels,preLabels);

% return result table
result = array2table([lcMeanKlDistance,lcMeanEuclideanDistance,lcMeanMSE,lcChebyshev,lcClark,lcCanberra,lcCosine,lcIntersection,lcSortLoss],'VariableNames',{'KlDistance','EuclideanDistance','MSE','Chebyshev','Clark','Canberra','Cosine','Intersection','sortLoss'} );
