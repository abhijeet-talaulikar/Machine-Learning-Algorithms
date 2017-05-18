classdef RBFNN
    properties
        Size
        Classes
        Train
        lambda
        x_Train
        y_Train
        RBFM
        W
    end
    methods
        function clf = RBFNN(x_train,y_train,lambda)            
            clf.Size = size(x_train,1);
            clf.Classes = unique(y_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.lambda = lambda;
            
            %Create the RBF Matrix
            clf.RBFM = ones(clf.Size,clf.Size);
            for i = 1:clf.Size
                for j = i+1:clf.Size
                    clf.RBFM(i,j) = exp(-(pdist2(clf.x_Train(i),clf.x_Train(j))^2)/(2*clf.lambda^2));
                    clf.RBFM(j,i) = clf.RBFM(i,j);
                end
            end
            
            %Find output weights W in one pass
            clf.W = zeros(size(clf.Classes,1),clf.Size);
            for j = 1:numel(clf.Classes)
                T = (clf.y_Train == clf.Classes(j));
                clf.W(j,:) = (pinv(clf.RBFM) * T)';
            end
        end
        function y_pred = Predict(clf, x_test)
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                Class_Layer = zeros(size(clf.Classes,1),1);
                for j = 1:numel(clf.Classes)
                    distances = zeros(size(clf.Classes,1),1);
                    for k = 1:clf.Size
                        distances(k) = exp(-(pdist2(x_test(i),clf.x_Train(k))^2)/(2*clf.lambda^2));
                    end
                    Class_Layer(j) = sum(distances.*clf.W(j));
                end
                [~,id] = max(Class_Layer);
                y_pred(i) = clf.Classes(id);
            end
        end
    end
end