classdef KNearestNeighbors
    properties
        nSamples
        nFeatures
        Classes
        Train
        x_Train
        y_Train
        K
    end
    methods
        function clf = KNearestNeighbors(x_train,y_train,k)
            [clf.nSamples,clf.nFeatures] = size(x_train);
            clf.Classes = unique(y_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.K = k;
        end
        function y_pred = Predict(clf, x_test)
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                distances = zeros(clf.nSamples,2);
                for j = 1:clf.nSamples
                    distances(j,:) = [pdist2(x_test(i,:),clf.x_Train(j,:)), clf.y_Train(j)];
                end
                distances = sortrows(distances,1);
                y_pred(i) = mode(distances(1:clf.K,2));
            end
        end
    end
end