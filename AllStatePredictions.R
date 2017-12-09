# Importing the dataset
as.data = read.csv('train.csv')
as.test=read.csv('test.csv')

# Splitting the dataset into the Training set and Validation set
library(caTools)
set.seed(414)
split = sample.split(as.data, SplitRatio = 0.7)
as.train = subset(as.data, split == TRUE)
as.val = subset(as.data, split == FALSE)
library(Matrix)
sparse_train = sparse.model.matrix(loss ~ . -loss, data=as.train)
sparse_val = sparse.model.matrix(loss ~ . -loss, data=as.val)
hist(as.train$loss, breaks=10000)

#Transforming Target
loss1=log(as.train$loss)
hist(loss1, breaks=10000)

# Fitting XGBoost to the Training set
#install.packages('xgboost')
#install.packages('readr')
#install.packages('stringr')
#install.packages('caret')
#install.packages('car')
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

Sevierity = xgboost(data = sparse_train, label = loss1, booster= "gbtree",
                    objective="reg:linear", eta = 0.1, gamma = 0, nround=200,
                    max_depth = 7, subsample = 1, colsample_bytree = 1, verbose = 0,
                    min_child_weight=1.5, lambda=1.5, alpha=0,
                    eval_metric = "mae")
ptrain = predict(Sevierity, sparse_train)
error.train=MAE(exp(ptrain),as.train$loss)
pvalid = predict(Sevierity, sparse_val)
error.val=MAE(exp(pvalid),as.val$loss)
error.val

#Refitting with entire dataset
sparse_data = sparse.model.matrix(loss~ . -loss, data=as.data)
loss2=log(as.data$loss)
Sevierity1 = xgboost(data = sparse_data, label = loss2, booster= "gbtree",
                     objective="reg:linear", eta = 0.1, gamma = 0, nround=200,
                     max_depth = 7, subsample = 1, colsample_bytree = 1, verbose = 0,
                     min_child_weight=1.5, lambda=1.5, alpha=0,
                     eval_metric = "mae")
pdata = predict(Sevierity1, sparse_data)
error.data=MAE(exp(pdata),as.data$loss)

#We manually appended fictitious data (101 for id variable, "B" in all categorical variables, continuous variables can have any numeric data) to the test data so that a sparse data frame could be created for variables with only one level. We later removed this fictitious prediction in the final predicted output#

sparse_test = sparse.model.matrix(~ ., data=as.test)
ptest= predict(Sevierity1, sparse_test)
loss= exp(ptest)
write.csv(loss, file = "predictions.csv")
