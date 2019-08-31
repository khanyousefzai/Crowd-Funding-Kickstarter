#importing Liabraries
#Importing Libraries
#STate contribution using machine learning algortihms
######################Thanks to my supervisor Prof. Omair Shafiq for
#######################his valuable time and support at every corner############################
##################Date: 4/15/2019###############3
##################Name: Saad Hasan###################
####Implementation of ANN on full dataset for multi classifcation#################
###########Course: ITEC:5920##############
#######Course Name: Design and Development of Data Intensive Applications######
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


#reading file and understanding data
#Project on Business Inteligence
tic()
set.seed(123)
dataframe <- read.csv("ks-projects-201801.csv", header=TRUE)
head(dataframe)
dim(dataframe)
dataframe[1,1]
summary(dataframe$currency)
summary(dataframe$country)
toc()

#Data Cleaning
# we just omit the missing values rows because it consttitues only 1 percent of the data
dataframe1 <- na.omit(dataframe)
summary(dataframe1)
dim(dataframe1)
summary(dataframe1$main_category)
summary(dataframe1$state)
sub <- function(x){
  a <- x[10]
  value <- if(a == 'suspended' | a == 'undefined'){
    a <- NA
  } else a   
  return(value)
}
dataframe1$state <- as.character(apply(dataframe1, 1, sub))
dataframe1 <- na.omit(dataframe1)
dim(dataframe1)
#eliminating unnecessary column to classify
dataframe2 <- dataframe1[,-c(1,2,5,7,3,9,13)]
table(dataframe2$main_category)
table(dataframe2$state)
#dataframe1$new <- is.factor(dataframe2)
#head(dataframe1)
#sapply(dataframe1, class)
dataframe2$state <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$state, 
      levels = c("failed", "successful", "live", "canceled"), 
      labels = c("0", "1", "2", "3"))))
head(dataframe2)
summary(dataframe2)
head(dataframe2)
#Random froest classificaton
#Gradient Boosted Machine  
#Hours <- format(as.POSIXct(strptime((date1$launched,"%d/%m/%Y %H:%M",tz=""))) ,format = "%H:%M")
#head(Hours)
#date_from <- as.Date('01/01/1999  12:00:00 AM',
#   format = "%m/%d/%Y %I:%M:%S %p")
#output

dataframe2$launchedate <- format(as.POSIXct(strptime(dataframe2$launched,"%m/%d/%Y %H:%M",tz=""))
                                 ,format = "%m/%d/%Y")
dataframe2$date_diff <- as.Date(as.character(dataframe2$deadline), format="%m/%d/%Y")-
  as.Date(as.character(dataframe2$launchedate), format="%m/%d/%Y")
#dataframe2 <- dummy_cols(dataframe2, select_columns = c("main_category", "country"))
dataframe2$date_diff <- as.numeric(gsub(" days","", dataframe2$date_diff))

dataframe2 <- dummy_cols(dataframe2, select_columns = c("main_category", "country"))
dataframe2$date_diff <- as.numeric(gsub(" days","", dataframe2$date_diff))
summary(dataframe2$country)
dataframe2 <- na.omit(dataframe2)
summary(dataframe2$date_diff)
dataframe3 <- dataframe2[,-c(1,2,3,6,9)]
head(dataframe3)
dataframe3$country <- as.numeric( 
  as.character( 
    factor( 
      dataframe3$country, 
      levels = c("AT", "AU", "BE", "CA", "CH", "DE", "DK", "ES", "FR", "GB", "HK",
                 "IE", "IT", "JP", "LU", "MX", "NL", "NO", "NZ", "SE", "SG", "US"), 
      labels = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                 "13", "14", "15", "16", "17", "18", "19", "20", "21", "22")))) 
dim(dataframe3)
sapply(dataframe3, typeof)
pakarmy <- function(x){ (x - min(x))/(max(x) - min(x)) }
dataframe3[, 2:6] <- data.frame(lapply(dataframe3[, 2:6], pakarmy))
#dividing datset into test and traindata
head(dataframe3)
dataframe3 <- omit.na(dataframe3$state)


set.seed(3456)
Index <- createDataPartition(dataframe3$state, p = .1, 
                                  list = FALSE, 
                                  times = 1)
downsam <- dataframe3[ Index,]
dataf <- dataf[,-c(1)]
write.csv(downsam,'downsamp.csv')

ind <- sample(2,nrow(dataframe3),replace = TRUE, prob = c(0.8, 0.2))
traindata <- dataframe3[ind == 1,] 
testdata  <- dataframe3[ind == 2,]
#Logistic Regression to find accuracy
head(traindata)
head(testdata)

# neural network using keras
train_x <- traindata[,-c(1,3)]
trainiy <- traindata[,c(1)]
test_x <- testdata[,-c(1,3)]
testiy <- testdata[,c(1)]
#train_x <- as.matrix(train_x)
#trainiy <- as.matrix(trainiy)
#test_x <- as.matrix(test_x)
#testiy <- as.matrix(testiy)
#train_y <- dummy_cols(trainiy)
#est_y <- dummy_cols(testiy)

head(train_y)
train_y <- to_categorical(trainiy)
test_y <- to_categorical(testiy)

train_x <- as.matrix.data.frame(train_x)
train_y <- as.matrix.data.frame(train_y)
test_x <- as.matrix.data.frame(test_x)
test_y <- as.matrix.data.frame(test_y)
head(train_y)
# create model
model <- keras_model_sequential()
# define and compile the model
model %>% 
         layer_dense(units = 264, activation = 'relu', input_shape = c(40)) %>% 
         layer_dropout(rate = 0.5) %>% 
         layer_dense(units = 20, activation = 'relu') %>% 
         layer_dropout(rate = 0.5) %>% 
         layer_dense(units = 18, activation = 'relu') %>% 
         layer_dropout(rate = 0.5) %>% 
         layer_dense(units = 4, activation = 'softmax')

model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics ='accuracy')

summary(model)
# training model
    model %>% fit(train_x,
                          train_y,
                          epochs = 100,
                          batch_size = 32,
                          validation_split = 0.10)

# evaluate 
model %>% evaluate(test_x, test_y)
prob <- model %>%
  predict_proba(test_x)
pred <- model %>% predict_classes(test_x)
prob <- as.matrix(prob)
y_pp <- predict_classes(model, test_x, verbose = 1)
test.ct2 <- table(Actual=testiy, Predicted=y_pp)
test.ct2


#alternate scenerios
#kut <- lapply(proba, function(x) ifelse(nrow(proba[[x]])< 0.5, 0, 1))
names(test_y)[1]<-paste("failed")
names(test_y)[2]<-paste("successful")
names(test_y)[3]<-paste("live") 
names(test_y)[4]<-paste("canceled")

prob1 <- prob[,1]
failed <- ifelse(prob1 > 0.5, 1, 0)
prob2 <- prob[,2]
successful <- ifelse(prob2 > 0.5, 1, 0)
prob3 <- prob[,3]
live <- ifelse(prob3 > 0.01, 1, 0)
prob4 <- prob[,4]
canceled <- ifelse(prob4 > 0.2, 1, 0)
lol <- cbind(failed,successful)
lo <- cbind(live,canceled)
lopo <- cbind(lol,lo)
df2 <- as.data.frame(lapply(lopo, unlist))
NewFactor <- factor(apply(lopo, 1, function(x) which(x == 1)), 
                    labels = colnames(lopo)) 

#df.new <- lapply(prob, function(x) ifelse(prob[,x] > 0.5, 1, 0))
#preda <- model %>%
 # predict_classes(test_x)
lol$observations <- factor( cross.m$Observations, 
                                levels=c("Underweight","Normal","Overweight") )
cross.m$Predicted<- factor( cross.m$Predicted, 
                            levels=c("Underweight","Normal","Overweight") )
conf <- table(lopo, test_y)

confusionMatrix(factor(lopo),
                factor(test_y),
                mode = "everything")

f.conf <- confusionMatrix(conf)
f.conf
conf
y <- performance(prediction(prob,test_y),'tpr','fpr')
plot(y)
as.numeric(lopo)


m <- dataframe2[,c(4,5,7,8,10)]
c <- cor(m)
corrplot::corrplot(c, type = "upper")
res2 <- cor.test(m$usd_pledged_real, m$backers,  method="kendall")
res2


test_prediction <- matrix(lopo, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))


ggplot(m, aes(x=date_diff, y=usd_pledged_real)) + geom_point()
# Change the point size, and shape
ggplot(m, aes(x=date_diff, y=usd_goal_real)) +
  geom_point(size=2, shape=23)
m$

corrplot()


#LDA
m <- dataframe2[,c(4,5,7,8,10)]
m[,2:5] <- data.frame(lapply(m[,2:5], pakarmy))
ind1 <- sample(2,nrow(m),replace = TRUE, prob = c(0.5, 0.5))
m1 <- m[ind1 == 1,] 
m2  <- m[ind1 == 2,]

r <- lda(formula = state ~ ., 
         data = m)

r3 <- lda(state ~ ., # training model
          m1)
plda = predict(object = r, # predictions
               newdata = m2)

con <- table(plda$class, m2)
3
p1 <- ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = species,
                                       shape = species), size = 2.5) + 
  labs(x = paste("LD1 (", percent(prop.lda[1]), ")", sep=""),
       y = paste("LD2 (", percent(prop.lda[2]), ")", sep=""))



# Plotting
set.seed(23423423)
# Set color by cond
ggplot(m, aes(x=date_diff, y=usd_pledged_real, color=usd_goal_real)) + geom_point(shape=1)

# Same, but with different colors and add regression lines
ggplot(m, aes(x=date_diff, y=usd_pledged_real, color=usd_goal_real)) +
  geom_point(shape=1) +
  scale_colour_hue(l=50) + # Use a slightly darker palette than normal
  geom_smooth(method=lm,   # Add linear regression lines
              se=FALSE)    # Don't add shaded confidence region

# Extend the regression lines beyond the domain of the data
ggplot(m, aes(x=date_diff, y=usd_pledged_real, color=usd_goal_real)) + geom_point(shape=1) +
  scale_colour_hue(l=50) + # Use a slightly darker palette than normal
  geom_smooth(method=lm,   # Add linear regression lines
              se=FALSE,    # Don't add shaded confidence region
              fullrange=TRUE) # Extend regression lines


# Set shape by cond
ggplot(m, aes(x=date_diff, y=usd_pledged_real, shape=usd_goal_real)) + geom_point()

# Same, but with different shapes
ggplot(m, aes(x=date_diff, y=usd_pledged_real, shape=usd_goal_real)) + geom_point() +
  scale_shape_manual(values=c(1,2)) 


ggplot(data = m, aes(x = date_diff, y = usd_pledged_real))
