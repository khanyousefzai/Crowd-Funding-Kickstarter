#importing Liabraries
#Importing Libraries
#Custoemr Churn Prediction whether a customer is loose left the telecom company or not
library(ggplot2)
library(MASS)
library(groupdata2)
library(gbm)
library(liquidSVM)
library(caret)
library(caTools)
library("pROC")
library(ggcorrplot)
library(tidyverse)
library(randomForest)
library(dplyr)
library(magrittr)
library(keras)
library(lubridate)
library(fastDummies)
library(cowplot)
library(ggpubr)
library(corrplot)
library(mlbench)
library(ElemStatLearn)
library(ROCR)
library(mlr)
library(keras)
library(tictoc)
library(corpus)
library(tidyverse)
library(ggpubr)
library(arm)
library(glmnet)

set.seed(123)
datf <- read.csv("downsample.csv", header=TRUE)
datf <- datf[,-c(1)]
# making two portions of dataset for trianing and testing
ind <- sample(2,nrow(datf),replace = TRUE, prob = c(0.8, 0.2))
traindata <- datf[ind == 1,] 
testdata  <- datf[ind == 2,]

#feature selection using gradient boosting machines
gbmmodel <- gbm(formula =  state ~.,
                   #   + MultipleLines
                   #  + OnlineSecurity 
                   #   + TechSupport 
                   # StreamingTV + 
                   #  StreamingMovies ,
                   distribution = "multinomial" ,
                   n.trees = 3000 ,
                   train.fraction = 0.5 ,
                   # distribution = "gausian" ,
                   n.minobsinnode = 10 ,
                   # nTrain = round(Churn*0.8),
                   verbose = TRUE ,   
                   shrinkage = 0.01 ,
                   interaction.depth = 4,
                   data = traindata)
summary(gbmmodel)
gbm.perf(gbmmodel)
#tesing on the test set
pred <- predict(gbmmodel, newdata = testdata, n.trees = 3000, type = 'response')
predgbm <- apply(pred, 1, which.max)
pmgbm <- table(Actual=predgbm, Predicted=testdata$state)
#We got the accuracy 87.5 percent on the test data 88 percent on the training

#radnom Forest
rfmodel <- randomForest(state ~.,
                        data = traindata,
                        ntree = 30,
                        mtry = 15,
                        importance = TRUE)

#tesing on the test set
predrf <- predict(rfmodel, newdata = testdata, n.trees = 3000, type = 'response')
prerf <- apply(predrf, 1, which.max)
pmrf <- table(Actual=prerf, Predicted=testdata$state)
  