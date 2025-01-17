function [weights,fval,exitFlag,output,grad] = lcLdlTrain(xInit,trainFeatures,trainLabels,optim)

fprintf('Begin training of BFGS-LC. \n');
% Read Optimalisation Parameters
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);
[weights,fval,exitFlag,output,grad] = fminunc(@lcProgress,xInit,options);

    function [target,gradient] = lcProgress(weights)
    c1=0.1;
    c2=c1*0.1;
    [row,cow]=size(weights);
    modProb = exp(trainFeatures * weights);  
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
    %%损失函数第一项                                      
    costfir=-sum(sum(trainLabels.*log(modProb)));

    %%损失函数第二项
    costsec=norm(weights,'fro')*norm(weights,'fro');

    %%损失函数第三项 theta中的不同列
    weightssize=size(weights,2);
    
    % weights
    relevance=0;
    for i=1:weightssize-1
        for j=i+1 :weightssize
            distance =euclideandist(weights(:,i), weights(:,j));
            s=corrcoef([weights(:,i), weights(:,j)]);
            relevance=relevance+s(1,2)*distance;
        end
    end

    % Target function.
    target =costfir+c1*costsec+c2*relevance;

    % The gradient.第一项是原始模型，第二项是向量F范数求和的形式，第三项是theta相关性；
    grad1=trainFeatures'*(modProb - trainLabels);

    grad2=0;
    for i=1:row
        for j=1:cow
            grad2(i,j)=2*sign(weights(i,j));
        end   
    end

    % euclideandist
    for i=1:row
        for j=1:cow
            temp=0;
            for k=1:cow
                s=corrcoef([weights(:,j), weights(:,k)]);
                temp1=abs(weights(i,j)-weights(i,k));
                temp2=sqrt(sum((weights(:,j)-weights(:,k)).^2));
                temp=temp + sign(s(1,2))* temp1./(temp2+0.00001);           
            end
                grad3(i,j)=temp;
        end
    end

    % sorensendist距离
    %      for i=1:row 
    %       for j=1:cow
    %           temp=0;newtemp=0;                                           
    %           for k=1:cow
    %               s=corrcoef([weights(:,j), weights(:,k)]);
    %               temp1=sign(weights(i,j)-weights(i,k));
    %               temp2=sign(weights(i,j)+weights(i,k));
    %               temp=temp + temp1;
    %               newtemp=newtemp+temp2;              
    %               res=sign(s(1,2))*temp/(newtemp+0.00001);         
    %           end
    %           grad3(i,j)=res;
    %       end 
    %  end

    % squaredchord距离
    % for i=1:row
    %     for j=1:cow
    %         temp=0;
    %         for k=1:cow
    %             s=corrcoef([weights(:,j), weights(:,k)]);
    %             temp1=2*weights(i,j)^2-2*weights(i,k)^2-(weights(i,j)^2-weights(i,k)^2);
    %             temp2=(weights(i,j)+weights(i,k))^2;
    %             temp=temp +sign(s(1,2))* temp1./(temp2+0.00001);
    %         end
    %         grad3(i,j)=temp;      
    %     end
    % end

    % kdlsist距离
    % for i=1:row
    %     for j=1:cow
    %         temp=0;
    %         temp2=abs(weights(i,j));
    %         for k=1:cow
    %                 s=corrcoef([weights(:,j), weights(:,k)]);
    %             temp1=-weights(i,k);
    %             temp=temp +sign(s(1,2))*  temp1./(temp2+0.00001);
    %         end
    %         grad3(i,j)=temp;
    %     end
    % end

    % intersecion距离
    %    for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               s=corrcoef([weights(:,j), weights(:,k)]);
    %               if weights(i,j)<weights(i,k)
    %               temp=temp +sign(s(1,2))* 1;
    %               end
    %           end
    %           grad3(i,j)=temp;
    %       end
    %    end

    % fldelity相似度偏导
    %  for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               %theat ik， ij可能乘机是负数，这里改为绝对值
    %                s=corrcoef([weights(:,j), weights(:,k)]);
    %               temp1=sqrt(abs(weights(i,k)));
    %               temp2=2*sqrt(abs(weights(i,j)));
    %               temp=temp + sign(s(1,2))* temp1./(temp2+0.00001);
    %           end
    %           grad3(i,j)=temp;
    %       end
    %  end

    % clark距离偏导
    %     for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               s=corrcoef([weights(:,j), weights(:,k)]);
    %               temp1=weights(i,k)^2-weights(i,j)^2-(weights(i,j)-weights(i,k))^2;
    %               temp2=(weights(i,j)-weights(i,k))^2*abs(weights(i,k)-weights(i,j));
    %               temp=temp +sign(s(1,2))* temp1./(temp2+0.00001);
    %           end
    %           grad3(i,j)=temp;
    %       end
    %  end

    % innerproduct距离偏导
    %   for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               %theat ik
    %                 s=corrcoef([weights(:,j), weights(:,k)]);
    %                temp=temp + sign(s(1,2))* weights(i,k);
    %           end
    %           grad3(i,j)=temp;
    %       end
    %   end

    % cosine距离
    %   for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               s=corrcoef([weights(:,j), weights(:,k)]);
    %               temp1=weights(i,k)*(sqrt(sum(weights(:,j).^2))+sqrt(sum(weights(:,k).^2)));
    %               temp2=weights(i,j)*sum(weights(:,j).*weights(:,k))/sqrt(sum(weights(:,j).^2));
    %               temp3=sqrt(sum(weights(:,j).^2))+sqrt(sum(weights(:,k).^2));
    %               temp=temp +  sign(s(1,2))* (temp1-temp2)./(temp3^2+0.00001);
    %           end
    %           grad3(i,j)=temp;
    %       end
    %   end

    % canberra距离
    %    for i=1:row
    %       for j=1:cow
    %           temp=0;
    %           for k=1:cow
    %               temp1=sign(weights(i,j))*(abs(weights(i,j))+abs(weights(i,k)));
    %               temp2=sign(weights(i,j))*(abs(weights(i,j)-weights(i,k)));
    %               temp3=(abs(weights(i,j))+abs(weights(i,k)))^2;
    %               temp=temp + (temp1-temp2)./(temp3+0.00001);
    %           end
    %           grad3(i,j)=temp;
    %       end
    %    end
    
    gradient =grad1+c1.*grad2+c2.*grad3;
    end
end

