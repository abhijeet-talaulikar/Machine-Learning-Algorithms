%   Log loss objective function
%   Non-adaptive learning rate
%   Always scale your inputs to [0, 1] range when using sigmoid function
%   It is advisable to use a low learning rate (for e.g 0.01) with high number of epochs/iterations
%   Making a deep network can lead to vanishing gradients problem with the sigmoid function

classdef MLP
    properties
        nSamples
        nFeatures
        Classes
        x_Train
        y_Train
        hiddenLayers
        W
        nIter
        learningRate
    end
    methods
        function clf = MLP(x_train,y_train,hiddenLayers,nIter,learningRate)            
            [clf.nSamples,clf.nFeatures] = size(x_train);
            clf.Classes = unique(y_train);
            if numel(clf.Classes) == 2; clf.Classes = [1];end %Binary class problem has only one output node
            clf.x_Train = x_train;
            clf.y_Train = y_train;
            clf.nIter = nIter;
            clf.learningRate = learningRate;
            
            %Create the architecture with randomly initialized weights
            clf.hiddenLayers = hiddenLayers;
            clf.W = cell(1,numel(clf.hiddenLayers)+1);
            prev = clf.nFeatures;
            for i = 1:numel(clf.hiddenLayers)
                clf.W{i} = 0.001*randi(999,clf.hiddenLayers(i),prev) .* ((-1 * ones(clf.hiddenLayers(i),prev)).^randi(2,clf.hiddenLayers(i),prev));
                clf.W{i} = [ones(clf.hiddenLayers(i),1) clf.W{i}];
                prev = clf.hiddenLayers(i);
            end
            clf.W{numel(clf.hiddenLayers)+1} = 0.001*randi(999,numel(clf.Classes),prev) .* ((-1 * ones(numel(clf.Classes),prev)).^randi(2,numel(clf.Classes),prev));
            clf.W{numel(clf.hiddenLayers)+1} = [ones(numel(clf.Classes),1) clf.W{numel(clf.hiddenLayers)+1}];
            
            %Train the network
            for i = 1:clf.nIter
                for j = 1:clf.nSamples
                    %Feed forward through the network
                    val = cell(1,numel(clf.hiddenLayers)+1);
                    prev = clf.x_Train(j,:);
                    for k = 1:numel(clf.hiddenLayers)
                        val{k} = zeros(clf.hiddenLayers(k),1);
                        for l = 1:clf.hiddenLayers(k)
                            val{k}(l) = clf.sigmoid([1.0 prev] * clf.W{k}(l,:)');
                        end
                        prev = val{k}(:)';
                    end
                    val{numel(clf.hiddenLayers)+1} = zeros(numel(clf.Classes),1);
                    for k = 1:numel(clf.Classes)
                        val{numel(clf.hiddenLayers)+1}(k) = clf.sigmoid([1.0 prev] * clf.W{numel(clf.hiddenLayers)+1}(k,:)');
                    end
                    
                    %Backpropagate through the network
                    %Compute delta values
                    delta = cell(1,numel(clf.hiddenLayers)+1);
                    delta{numel(clf.hiddenLayers)+1} = zeros(numel(clf.Classes),1);
                    for l = 1:numel(clf.Classes)
                        expected = clf.y_Train(j) == clf.Classes(l);
                        output = val{numel(clf.hiddenLayers)+1}(l);
                        delta{numel(clf.hiddenLayers)+1}(l) = expected - output;
                    end
                    for k = numel(clf.hiddenLayers):-1:1
                        delta{k} = zeros(clf.hiddenLayers(k),1);
                        for l = 1:clf.hiddenLayers(k)
                            weightedSum = clf.W{k+1}(:,l)' * delta{k+1}(:);
                            delta{k}(l) = clf.derivative(val{k}(l)) * weightedSum;
                        end
                    end
                    
                    %Update weights
                    for k = 1:numel(clf.hiddenLayers)+1
                        if k == 1; prev = [1.0 clf.x_Train(j,:)]';
                        else prev = [1.0;val{k-1}(:)];
                        end
                        if k == numel(clf.hiddenLayers)+1; nodes = numel(clf.Classes);
                        else nodes = clf.hiddenLayers(k);
                        end
                        for l = 1:nodes
                            for m = 1:size(prev,1)
                                clf.W{k}(l,m) = clf.W{k}(l,m) + clf.learningRate * delta{k}(l) * prev(m);
                            end
                        end
                    end
                end
            end
        end
        function val = sigmoid(clf, x)
            val = 1 / (1+exp(-x));
        end
        function val = derivative(clf,x)
            val = x * (1 - x);
        end
        function y_pred = Predict(clf, x_test)
            y_pred = zeros(size(x_test,1),1);
            for i = 1:size(x_test,1)
                %Feed forward through the network
                val = cell(1,numel(clf.hiddenLayers)+1);
                prev = x_test(i,:);
                for j = 1:numel(clf.hiddenLayers)
                    val{j} = zeros(clf.hiddenLayers(j),1);
                    for k = 1:clf.hiddenLayers(j)
                        val{j}(k) = clf.sigmoid([1.0 prev] * clf.W{j}(k,:)');
                    end
                    prev = val{j}(:)';
                end
                val{numel(clf.hiddenLayers)+1} = zeros(numel(clf.Classes),1);
                for k = 1:numel(clf.Classes)
                    val{numel(clf.hiddenLayers)+1}(k) = clf.sigmoid([1.0 prev] * clf.W{numel(clf.hiddenLayers)+1}(k,:)');
                end
                if numel(clf.Classes) == 1
                    y_pred(i) = (val{numel(clf.hiddenLayers)+1} >= 0.5);
                else
                    [~,id] = max(val{numel(clf.hiddenLayers)+1});
                    y_pred(i) = clf.Classes(id);
                end
            end
        end
    end
end
