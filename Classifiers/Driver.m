clc

data = csvread('IRIS.csv');
data = data(randperm(end),:);

x_train = data(1:90,1:end-1);
y_train = data(1:90,end);

x_test = data(91:end,1:end-1);
y_test = data(91:end,end);

clf = IAPNN(x_train,y_train,1);
y_pred = clf.Predict(x_test)
sum(y_pred == y_test)*100/size(y_pred,1)