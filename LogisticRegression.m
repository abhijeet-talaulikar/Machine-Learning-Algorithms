%   Binary classifier (0/1)
%   Using Log-Loss objective: Cost = t(-logy) + (1-t)(-log(1-y))
%   Stopping criteria is number of iterations
%   Batch Gradient Descent (can be made stochastic with a little change)

classdef LogisticRegression
    properties
        nSamples
        nFeatures
        Train
        x_Train
        y_Train
        W
        Threshold
        LearningRate
    end
    methods
        function clf = LogisticRegression(x_train,y_train,thresh,LearningRate,n_iter)            
            [clf.nSamples,clf.nFeatures] = size(x_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.Threshold = thresh;
            clf.LearningRate = LearningRate;
            clf.W = zeros(1+clf.nFeatures,1);
            
            %Train the classifier
            for n = 1:n_iter
                upd = zeros(1+clf.nFeatures,1);
                for i = 1:clf.nSamples
                    y = sigmf([1.0 clf.x_Train(i,:)]*clf.W,[1 0]);
                    t = clf.y_Train(i);
                    upd = upd + (y-t)*[1.0 clf.x_Train(i,:)]';
                end
                
                %Update the weights
                clf.W = clf.W - clf.LearningRate*upd;
            end
        end
        function [y_pred,y_prob] = Predict(clf,x_test)
            y_pred = zeros(size(x_test,1),1);
            y_prob = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                y_prob(i) = sigmf([1.0 x_test(i,:)]*clf.W,[1 0]);
                y_pred(i) = sigmf([1.0 x_test(i,:)]*clf.W,[1 0]) >= clf.Threshold;
            end
        end
    end
end
