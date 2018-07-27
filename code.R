setwd("C:\\R-Programming\\Bike rental")
train<-read.csv("train.csv")

train<-subset(train,count<800)

train$season<-as.factor(train$season)
train$holiday<-as.factor(train$holiday)
train$weather<-as.factor(train$weather)
train$workingday<-as.factor(train$workingday)


train$datetime<-as.character(train$datetime)
model<-lm(count~.-datetime-casual-registered,data=train)

#train$difftemp<-train$atemp-train$temp

month<-c()
date<-c()
hours<-c()

train$datetime<-as.character(train$datetime)
for(i in train$datetime){
  strings<-strsplit(i," ")
  dateString<-strsplit(strings[[1]][1],"-")
  month<-c(month,dateString[[1]][2])
  date<-c(date,dateString[[1]][3])
  timeString<-strsplit(strings[[1]][2],":")
  hours<-c(hours,timeString[[1]][1])
}


train$month<-as.factor((month))
train$date<-as.numeric(date)
train$hours<-as.numeric(hours)

library(lubridate)
dayOfWeek<-wday(train$datetime)
train$dayOfWeek<-as.factor(dayOfWeek)
train$isWeekend<-ifelse(dayOfWeek %in% c(1,7),1,0)
train$isWeekend<-as.factor(train$isWeekend)


train$categoryhours<-cut(train$hours,breaks = c(0,5,11,16,19,24),include.lowest = TRUE)
levels(train$categoryhours)<-c("Night","Morning","Afternoon","Evening","Night")


train$humidityCategory<-cut(train$humidity,breaks = c(0,30,50,80,101),include.lowest = TRUE)
levels(train$humidityCategory)<-c("Low","Mid","High","Very High")

train$windspeedCategory<-cut(train$windspeed,breaks = c(0,7,20,30,60),include.lowest = TRUE)
levels(train$windspeedCategory)<-c("Low","Mid","High","Very High")

train$isTempGood<-as.factor(ifelse((train$temp<35 & train$temp>12),1,0))
train$isaTempGood<-as.factor(ifelse((train$atemp<=35 & train$atemp>=17),1,0))

train$date<-NULL

modelcasual<-lm(casual~.,data=train[,-c(1,11,12)])
train$o1<-predict(modelcasual,train)

modelregistered<-lm(registered~.,data=train[,-c(1,10,12)])
train$o2<-predict(modelregistered,train)

modelcount<-lm(count~.,data=train[,-c(1,10,11)])
train$o3<-predict(modelcount,train)



library(dplyr)
library("xgboost")
library("Ckmeans.1d.dp")

t<- train %>% mutate_if(is.factor,as.numeric)
t$casual<-NULL
t$registered<-NULL
t$datetime<-NULL


#library(glmnet)
#modelcasual = cv.glmnet(as.matrix(t[,-9]),as.matrix(train[,10]),alpha = 0.5,lambda = 10^seq(4,-1,-0.1))
#best_lambda = modelcasual$lambda.min
#en_coeff = predict(modelcasual,s = best_lambda,type = "coefficients")
#t$o4<-predict(modelcasual,as.matrix(t[,-9]),s=best_lambda)

#modelregistered = cv.glmnet(as.matrix(t[,-9]),as.matrix(train[,11]),alpha = 0.5,lambda = 10^seq(4,-1,-0.1))
#best_lambda = modelregistered$lambda.min
##en_coeff = predict(modelcasual,s = best_lambda,type = "coefficients")
#t$o5<-predict(modelregistered,as.matrix(t[,-9]),s=best_lambda)

#modelcount = cv.glmnet(as.matrix(t[,-9]),as.matrix(train[,12]),alpha = 0.5,lambda = 10^seq(4,-1,-0.1))
#best_lambda = modelcount$lambda.min
##en_coeff = predict(modelcasual,s = best_lambda,type = "coefficients")
#t$o6<-predict(modelcount,as.matrix(t[,-9]),s=best_lambda)


##try to normalise the data and then try with XGBoost and Knn algorithm!!!! (might perform better)
#t<-preprocess(t[,-1],c("center","scale"))

data_variables <- as.matrix(t[,-9])
data_label <- t[,"count"]
data_matrix <- xgb.DMatrix(data = data_variables, label = data_label)


xgb_params <- list("objective" = "reg:linear",eta=0.05,gamma=0.9,max_depth=10,eval_metric="rmse")
xgbcv <- xgb.cv( params = xgb_params, data = data_matrix, nrounds = 300, nfold = 10, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 10, maximize = F)

nround    <- xgbcv$best_iteration # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions

bst_model <- xgb.train(params = xgb_params,
                       data = data_matrix,
                       nrounds = nround)


##Visualization
library(ggplot2)
g<-ggplot(train,aes(season))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(holiday))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(workingday))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(weather))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(month))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(hours))+geom_histogram(stat = "count")
print(g)


g<-ggplot(train,aes(dayOfWeek))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(isWeekend))+geom_histogram(stat = "count")
print(g)

g<-ggplot(train,aes(count,fill=humidityCategory))+geom_histogram(stat = "bin",bins=20)
print(g)

g<-ggplot(train,aes(count,fill=windspeedCategory))+geom_histogram(stat = "bin",bins=20)
print(g)

g<-ggplot(train,aes(count,fill=isTempGood))+geom_histogram(stat = "bin",bins=20)
print(g)

g<-ggplot(train,aes(count,fill=isaTempGood))+geom_histogram(stat = "bin",bins=20)
print(g)



## Test data
test<-read.csv("test.csv")
test$season<-as.factor(test$season)
test$holiday<-as.factor(test$holiday)
test$weather<-as.factor(test$weather)
test$workingday<-as.factor(test$workingday)


test$datetime<-as.character(test$datetime)


month<-c()
date<-c()
hours<-c()

test$datetime<-as.character(test$datetime)
for(i in test$datetime){
  strings<-strsplit(i," ")
  dateString<-strsplit(strings[[1]][1],"-")
  month<-c(month,dateString[[1]][2])
  date<-c(date,dateString[[1]][3])
  timeString<-strsplit(strings[[1]][2],":")
  hours<-c(hours,timeString[[1]][1])
}


test$month<-as.factor((month))
test$date<-as.numeric(date)
test$hours<-as.numeric(hours)

library(lubridate)
dayOfWeek<-wday(test$datetime)
test$dayOfWeek<-as.factor(dayOfWeek)
test$isWeekend<-ifelse(dayOfWeek %in% c(1,7),1,0)
test$isWeekend<-as.factor(test$isWeekend)


#test$categoryDate<-cut(test$date,breaks = c(0,10,20,31),include.lowest = TRUE)
#levels(test$categoryDate)<-c("Starting","Mid","End")

test$categoryhours<-cut(test$hours,breaks = c(0,5,11,16,19,24),include.lowest = TRUE)
levels(test$categoryhours)<-c("Night","Morning","Afternoon","Evening","Night")

#test$month<-as.numeric(month)
#test$quater<-cut(test$month,breaks = c(2,4,9,12),include.lowest = TRUE)
#test$quater[is.na(test$quater)]<-as.factor("(9,12]")
#levels(test$quater)<-c("Starting","Mid","End")
#test$month<-as.factor(month)

test$humidityCategory<-cut(test$humidity,breaks = c(0,30,50,80,101),include.lowest = TRUE)
levels(test$humidityCategory)<-c("Low","Mid","High","Very High")

test$windspeedCategory<-cut(test$windspeed,breaks = c(0,7,20,30,50),include.lowest = TRUE)
levels(test$windspeedCategory)<-c("Low","Mid","High","Very High")

test$isTempGood<-as.factor(ifelse((test$temp<35 & test$temp>12),1,0))
test$isaTempGood<-as.factor(ifelse((test$atemp<=35 & test$atemp>=17),1,0))

test$date<-NULL
test$datetime<-NULL

test$o1<-predict(modelcasual,test)

test$o2<-predict(modelregistered,test)

test$o3<-predict(modelcount,test)

t<-test%>%mutate_if(is.factor,as.numeric)

predictions<-predict(bst_model,as.matrix(t))
predictions[predictions<0]<-0

temp<-read.csv("test.csv")

df<-data.frame(datetime=temp$datetime,count=predictions)

write.csv(df,"output.csv",row.names = FALSE)


