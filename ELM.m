%   Binary classifier
%   Takes input weights from user

classdef ELM
    properties
        nSamples
        nFeatures
        Classes
        Train
        lambda
        x_Train
        y_Train
        W
        G
        V
    end
    methods
        function clf = ELM(x_train,y_train,w)            
            [clf.nSamples,clf.nFeatures] = size(x_train);
            clf.Classes = unique(y_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.W = w; %Dimension should be (number of Samples x number of Features)
            
            %Create the intermediate matrix G
            for i = 1:clf.nSamples
                for j = 1:clf.nSamples
                    clf.G(i,j) = clf.x_Train(i,:) * clf.W(j,:)';
                end
            end
            
            %Set the output weights V
            clf.V = pinv(clf.G) * clf.y_Train;
        end
        function y_pred = Predict(clf, x_test)
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                Gi = zeros(1,clf.nSamples);
                for j = 1:clf.nSamples
                    Gi(j) = x_test(i,:) * clf.W(j,:)';
                end
                y_pred(i) = Gi * clf.V;
                y_pred(i) = ((y_pred(i)-1)^2)<=(y_pred(i)^2);
            end
        end
    end
end
