%   Takes real-valued variables as input
%   Assumes each real-valued variable is normally distributed

classdef GaussianNaiveBayes
    properties
        nSamples
        nFeatures
        Classes
        x_Train
        y_Train
        Py
    end
    methods
        function clf = GaussianNaiveBayes(x_train,y_train)            
            [clf.nSamples,clf.nFeatures] = size(x_train);
            clf.Classes = unique(y_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            for i = 1:numel(clf.Classes)
                clf.Py(i,1) = sum(clf.y_Train == clf.Classes(i)) / clf.nSamples;
            end
        end
        function y_pred = Predict(clf, x_test)
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                Class_Layer = zeros(size(clf.Classes,1),1);
                for j = 1:numel(clf.Classes)
                    Class_Layer(j) = clf.Py(j);
                    classSet = clf.x_Train(clf.y_Train == clf.Classes(j),:);
                    for k = 1:clf.nFeatures
                        mu = mean(classSet(:,k));
                        sigma = std(classSet(:,k));
                        Class_Layer(j) = Class_Layer(j) * normpdf(x_test(i,k),mu,sigma);
                    end
                end
                [~,id] = max(Class_Layer);
                y_pred(i) = clf.Classes(id);
            end
        end
    end
end
