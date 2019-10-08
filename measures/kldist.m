function distance = kldist(realDistribution, preDistribution)
%KLDIST	  Calculate the average Kullback-Leibler divergence between the predicted label
%         distribution and the real label distribution.
%
%	Description
%   DISTANCE = KLDIST(RD, PD) calculate the average Kullback-Leibler divergence 
%   between the predicted label distribution and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Kullback-Leibler divergence
%	
temp = 0;
for i =1:length(realDistribution)
    if preDistribution(i) == 0
        preDistribution(i) = 1.0e-11;
    end
    if realDistribution(i) ~= 0   
        temp= temp + realDistribution(i) * (log(realDistribution(i)/preDistribution(i)));
    end
end
distance=temp;
end
