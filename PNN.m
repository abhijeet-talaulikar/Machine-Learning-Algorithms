classdef PNN
    properties
        Size
        Classes
        Train
        lambda
        x_Train
        y_Train
    end
    methods
        function clf = PNN(x_train,y_train,lambda)
            %Mean center and normalize the data
            x_train = x_train - repmat(mean(x_train),size(x_train,1),1);
            for i = 1:size(x_train,1)
                x_train(i,:) = x_train(i,:)/norm(x_train(i,:));
            end
            
            clf.Size = size(x_train,1);
            clf.Classes = unique(y_train);
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.lambda = lambda;
        end
        function y_pred = Predict(clf, x_test)
            %Normalize the data
            for i = 1:size(x_test,1)
                x_test(i,:) = x_test(i,:)/norm(x_test(i,:));
            end
            
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                Class_Layer = zeros(size(clf.Classes,1),1);
                for j = 1:numel(clf.Classes)
                    classSet = clf.x_Train(clf.y_Train == clf.Classes(j),:);
                    for k = 1:size(classSet,1)
                        Class_Layer(j) = Class_Layer(j) + exp((sum(x_test(i).*classSet(k))-1)/(2*clf.lambda));
                    end
                    Class_Layer(j) = Class_Layer(j)/size(classSet,1);
                end
                [~,id] = max(Class_Layer);
                y_pred(i) = clf.Classes(id);
            end
        end
    end
end