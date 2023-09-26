# ****************************** LIBRARIES ***************

library(dplyr)
# Load the required library for logistic regression
#install.packages("glm")

library(caret)
#Load necessary packages
#install.packages("randomForest")
#install.packages("gbm")
library(randomForest)
library(gbm)


# ************************* LOAD DATA ***************************
setwd("C:/Users/Lenovo/Downloads")

train <- read.csv("buad5132-m1-training-data (1).csv")
test <- read.csv("buad5132-m1-test-data (1).csv")



# ********************* 1. Training Dataset **********************

#Cleaning Training Data
train$INDEX <- as.factor(train$INDEX)
train$TARGET_FLAG <- as.factor(train$TARGET_FLAG)
train$SEX <- as.factor(train$SEX)
train$EDUCATION <- as.factor(train$EDUCATION)
train$PARENT1 <- as.factor(train$PARENT1)
train$INCOME <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", train$INCOME)))
train$HOME_VAL <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", train$HOME_VAL)))
train$MSTATUS <- as.factor(train$MSTATUS)
train$REVOKED <- as.factor(train$REVOKED)
train$RED_CAR <- as.factor(ifelse(train$RED_CAR=="yes", 1, 0))
train$URBANICITY <- ifelse(train$URBANICITY == "Highly Urban/ Urban", "Urban", "Rural")
train$URBANICITY <- as.factor(train$URBANICITY)
train$JOB <- as.factor(train$JOB)
train$CAR_USE <- as.factor(train$CAR_USE)
train$CAR_TYPE <- as.factor(train$CAR_TYPE)
train$DO_KIDS_DRIVE <- as.factor(ifelse(train$KIDSDRIV > 0, 1, 0 ))
train$OLDCLAIM <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", train$HOME_VAL)))
train$BLUEBOOK <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", train$BLUEBOOK)))

str(train)


# A. Histogram and Bar Plot of all the variables in train data

# We can create histogram of only numeric data, so we only create a histogram if it's numeric variable. And if its categorical we create Bar Plot
# Loop through all columns in the data frame
for(i in 1:ncol(train)){
  # Check if the column is numeric or categorical
  if(is.numeric(train[,i])){
    # For numeric columns, create a histogram
    par(mfrow=c(1,1),mar=c(4,4,2,1))
    hist(train[,i], main=colnames(train[i]))
  } else {
    # For categorical columns, create a bar plot
    par(mfrow=c(1,1),mar=c(4,4,2,1))
    barplot(table(train[,i]), main=colnames(train[i]))
  }
}


# B. Box Plot of all the variables in the train data to check for outliers

# Loop through all columns in the data frame
for(i in 1:ncol(train)){
  # Check if the column is numeric
  if(is.numeric(train[,i])){
    # For numeric columns, create a box plot
    par(mfrow=c(1,1),mar=c(4,4,2,1))
    boxplot(train[,i], main=colnames(train[i]))
  }
}

# C. Fill Na values


sum(is.na(train))


# This code is imputing missing values on train data. It replaces missing values for the column "AGE" with the mean of the non-missing values in that column. It imputes missing values for the columns "YOJ", "INCOME", "HOME_VAL", and "CAR_AGE" by the mean of their respective non-missing values for each group of "JOB" or "CAR_TYPE". It sets negative values in the "CAR_AGE" column to 0 and creates a new variable "OLDCLAIM" based on the condition of "CAR_AGE" being less than 5 and not missing. The missing values in "OLDCLAIM" are imputed by the mean of its non-missing values. The code then creates a binary variable "HOME_OWNER" based on whether "HOME_VAL" is equal to 0 or not. The code also calculates the square root of "TRAVTIME" and "BLUEBOOK" and creates new variables "SQRT_TRAVTIME" and "SQRT_BLUEBOOK", respectively. 



train$AGE[is.na(train$AGE)] <- mean(train$AGE, na.rm = "TRUE")

# Impute missing values for YOJ by the mean of its non-missing values for each group of JOB
train <- train %>% 
  group_by(JOB) %>% 
  mutate(YOJ = ifelse(is.na(YOJ), mean(YOJ, na.rm = TRUE), YOJ)) %>% 
  ungroup()

# Impute missing values for INCOME by the mean of its non-missing values for each group of JOB
train <- train %>% 
  group_by(JOB) %>% 
  mutate(INCOME = ifelse(is.na(INCOME), mean(INCOME, na.rm = TRUE), INCOME)) %>% 
  ungroup()

# Impute missing values for HOME_VAL by the mean of its non-missing values for each group of JOB
train <- train %>% 
  group_by(JOB) %>% 
  mutate(HOME_VAL = ifelse(is.na(HOME_VAL), mean(HOME_VAL, na.rm = TRUE), HOME_VAL)) %>% 
  ungroup()

# Impute missing values for CAR_AGE by the mean of its non-missing values for each group of CAR_TYPE
train <- train %>% 
  group_by(CAR_TYPE) %>% 
  mutate(CAR_AGE = ifelse(is.na(CAR_AGE), mean(CAR_AGE, na.rm = TRUE), CAR_AGE)) %>% 
  ungroup()

# Impute missing values for OLDCLAIM by the mean of its non-missing values for each group of CAR_AGE
train <- train %>% 
  group_by(CAR_TYPE) %>% 
  mutate(CAR_AGE = ifelse(is.na(CAR_AGE), mean(CAR_AGE, na.rm = TRUE), CAR_AGE)) %>% 
  ungroup() 


train$CAR_AGE[train$CAR_AGE < 0 ] <- 0 
train$OLDCLAIM <- ifelse(train$CAR_AGE < 5 & !is.na(train$CAR_AGE),0,train$OLDCLAIM)
train$HOME_OWNER <- ifelse(train$HOME_VAL == 0, 0, 1)
train$SQRT_TRAVTIME <- sqrt(train$TRAVTIME)
train$SQRT_BLUEBOOK <- sqrt(train$BLUEBOOK)

sum(is.na(train))

# Replace the remaining Na values by 0
train[is.na(train)] <- 0

sum(is.na(train))


# D. Binning Variable Income

# Between 1 to 30000  = Low
# Between 30000 to 80000 = Medium
# Above 80000  = High
train$INCOME_bin[is.na(train$INCOME)] <- "NA"
train$INCOME_bin[train$INCOME == 0] <- "Zero"
train$INCOME_bin[train$INCOME >= 1 & train$INCOME < 30000] <- "Low"
train$INCOME_bin[train$INCOME >= 30000 & train$INCOME < 80000] <- "Medium"
train$INCOME_bin[train$INCOME >= 80000] <- "High"
train$INCOME_bin <- factor(train$INCOME_bin)
train$INCOME_bin <- factor(train$INCOME_bin, levels=c("NA","Zero","Low","Medium","High"))

summary(train)







# *********************** 2. Testing Dataset **********************

# Cleaning on test data
test$INDEX <- as.factor(test$INDEX)
test$TARGET_FLAG <- as.factor(test$TARGET_FLAG)
test$SEX <- as.factor(test$SEX)
test$EDUCATION <- as.factor(test$EDUCATION)
test$PARENT1 <- as.factor(test$PARENT1)
test$INCOME <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$INCOME)))
test$HOME_VAL <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$HOME_VAL)))
test$MSTATUS <- as.factor(test$MSTATUS)
test$REVOKED <- as.factor(test$REVOKED)
test$RED_CAR <- as.factor(ifelse(test$RED_CAR=="yes", 1, 0))
test$URBANICITY <- ifelse(test$URBANICITY == "Highly Urban/ Urban", "Urban", "Rural")
test$URBANICITY <- as.factor(test$URBANICITY)
test$JOB <- as.factor(test$JOB)
test$CAR_USE <- as.factor(test$CAR_USE)
test$CAR_TYPE <- as.factor(test$CAR_TYPE)
test$DO_KIDS_DRIVE <- as.factor(ifelse(test$KIDSDRIV > 0, 1, 0 ))
test$OLDCLAIM <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$HOME_VAL)))
test$BLUEBOOK <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$BLUEBOOK)))

str(test)

# A. Histogram and Bar Plot of all the variables in test data

# We can create histogram of only numeric data, so we only create a histogram if it's numeric variable. And if its categorical we create Bar Plot
# Loop through all columns in the data frame
test <- subset(test, select = -c(TARGET_FLAG))

for(col in 1:ncol(test)){
  temp_col = test[,col][is.finite(test[,col])]
  if(is.numeric(temp_col)){
    hist(temp_col, main = colnames(test)[col])
  } else {
    barplot(table(temp_col), main = colnames(test)[col])
  }
}


# B. Box Plot of all the variables in the test data to check for outliers

# Loop through all columns in the data frame
for(i in 1:ncol(test)){
  # Check if the column is numeric
  if(is.numeric(test[,i])){
    # For numeric columns, create a box plot
    par(mfrow=c(1,1),mar=c(4,4,2,1))
    boxplot(test[,i], main=colnames(test[i]))
  }
}




# C. Fill Na values



sum(is.na(test))



# impute missing values for AGE column with mean
test$AGE[is.na(test$AGE)] <- mean(test$AGE, na.rm = TRUE)

# group by JOB and CAR_TYPE
grouped_test <- test %>% group_by(JOB, CAR_TYPE)

# impute missing values for YOJ, INCOME, HOME_VAL, and CAR_AGE by group mean
test$YOJ[is.na(test$YOJ)] <- mean(grouped_test$YOJ, na.rm = TRUE)
test$INCOME[is.na(test$INCOME)] <- mean(grouped_test$INCOME, na.rm = TRUE)
test$HOME_VAL[is.na(test$HOME_VAL)] <- mean(grouped_test$HOME_VAL, na.rm = TRUE)
test$CAR_AGE[is.na(test$CAR_AGE)] <- mean(grouped_test$CAR_AGE, na.rm = TRUE)

# set negative CAR_AGE values to 0
test$CAR_AGE[test$CAR_AGE < 0] <- 0

# create new variable OLDCLAIM based on CAR_AGE < 5 and not missing
test$OLDCLAIM <- ifelse(test$CAR_AGE < 5 & !is.na(test$CAR_AGE), 1, 0)

# impute missing OLDCLAIM values with mean
test$OLDCLAIM[is.na(test$OLDCLAIM)] <- mean(test$OLDCLAIM, na.rm = TRUE)

# create new binary variable HOME_OWNER based on HOME_VAL
test$HOME_OWNER <- ifelse(test$HOME_VAL > 0, 1, 0)

# calculate square root of TRAVTIME and BLUEBOOK and create new variables
test$SQRT_TRAVTIME <- sqrt(test$TRAVTIME)
test$SQRT_BLUEBOOK <- sqrt(test$BLUEBOOK)

sum(is.na(test))


# D. Binning Variable Income

# Between 1 to 30000  = Low
# Between 30000 to 80000 = Medium
# Above 80000  = High
test$INCOME_bin[is.na(test$INCOME)] <- "NA"
test$INCOME_bin[test$INCOME == 0] <- "Zero"
test$INCOME_bin[test$INCOME >= 1 & test$INCOME < 30000] <- "Low"
test$INCOME_bin[test$INCOME >= 30000 & test$INCOME < 80000] <- "Medium"
test$INCOME_bin[test$INCOME >= 80000] <- "High"
test$INCOME_bin <- factor(test$INCOME_bin)
test$INCOME_bin <- factor(test$INCOME_bin, levels=c("NA","Zero","Low","Medium","High"))




# ****************** 3. Summarize the model after data exploration ***********
summary(train)

summary(test)






# ******************* 4. Split the data *************************

# Split training data into train and validation (8:2) to check for overfitting
# Calculate the number of observations for the validation set
val_obs <- round(0.2 * nrow(train))

# Randomly sample the indices for the validation set
val_indices <- sample(1:nrow(train), val_obs)

# Create the validation data set by subsetting the train data using the sampled indices
validation_data <- train[val_indices, ]

# Remove the validation data from the train data
train_data <- train[-val_indices, ]

# Split train and validation data into dependent and independent variables

colnames(train_data)

X_train <- subset(train_data, select = -c(INDEX, TARGET_FLAG))
y_train_TARGET_FLAG <- subset(train_data, select = TARGET_FLAG)
#y_train_TARGET_AMT <- subset(train_data, select = TARGET_AMT)
colnames(validation_data)

X_valid <- subset(validation_data, select = -c(INDEX, TARGET_FLAG))
y_valid_TARGET_FLAG <- subset(validation_data, select = TARGET_FLAG)
#y_valid_TARGET_AMT <- subset(validation_data, select = TARGET_AMT)

colnames(test)

X_test <- subset(test, select = -c(INDEX))












# ******************* 5. Model creation *************************


# We create both classification and regression models.


# ******************************* CLASSIFICATION ************************



#install.packages("randomForest")
library(randomForest)


# Bagging using random subspace method
bag_model <- randomForest(X_train, y_train_TARGET_FLAG$TARGET_FLAG, ntree = 100, mtry = ncol(X_train) / 2)
bag_pred <- predict(bag_model, X_valid)
bag_acc <- mean(bag_pred == y_valid_TARGET_FLAG$TARGET_FLAG)
cat("Bagging accuracy: ", bag_acc, "\n")



# Random forest
rf_model <- randomForest(X_train, y_train_TARGET_FLAG$TARGET_FLAG, ntree = 100)
rf_pred <- predict(rf_model, X_valid)
rf_acc <- mean(rf_pred == y_valid_TARGET_FLAG$TARGET_FLAG)
cat("Random forest accuracy: ", rf_acc, "\n")


###LOGISTIC REGRESSION MODEL###

Model_logistc<- glm(TARGET_FLAG ~. -INDEX, data = train, family = binomial())
summary(Model_logistc)
Lpred=(predict(Model_logistc,train, type="response"))
Lpred <- ifelse(Lpred > 0.5,1,0)
table(Lpred,train$TARGET_FLAG)
acc_logistic <- mean(Lpred == train$TARGET_FLAG)
acc_logistic
 
#accuracy of 80%
#Accuracy = (917 + 5535) / (917 + 5535 + 1236 + 473) = 0.8007

############SVM MODELS#####

#RBF Kernel##
install.packages("e1071")
library(e1071)

svm_model <- svm(TARGET_FLAG ~ .- INDEX, data = train, kernel = "radial", cost = 1, gamma = 0.1)
# Predict using the SVM model
svm_pred=(predict(svm_model,train, type="response"))
table(svm_pred,train$TARGET_FLAG)
svm_acc <- mean(svm_pred == train$TARGET_FLAG)
svm_acc  
#88%
#Accuracy = (5936 + 1416) / (5936 + 737 + 72 + 1416)


#SVM with LINEAR Kernal

svm_modelinear <- svm(TARGET_FLAG ~.- INDEX,data = train,kernel='linear',scale = FALSE)
svm_predlinear=(predict(svm_modelinear,train, type="response"))
table(svm_predlinear,train$TARGET_FLAG)
svm_acc_linear <- mean(svm_predlinear == train$TARGET_FLAG)
svm_acc_linear 
#ACCURACY 59%

remove.packages("e1071")
install.packages("e1071")
library(e1071)

#SVM WITH POLYNOMIAL##
svm_poly = svm(TARGET_FLAG ~.- INDEX,data = train,kernel='poly', degree=3)
svm_pred_POLY=(predict(svm_poly,train, type="response"))
table(svm_pred_POLY,train$TARGET_FLAG)
svm_acc_poly<- mean(svm_pred_POLY == train$TARGET_FLAG)
svm_acc_poly
#ACCURACY 78%



### Best model is with kernal radial so we will go for that for further resuts

test$'TARGET_FLAG' <- 0
test$'TARGET_FLAG' <- predict(svm_model, newdata = test)



# Output Data File
scores <- test[c("INDEX","TARGET_FLAG")]
write.csv(scores, file = "SVM.csv", row.names = FALSE)

