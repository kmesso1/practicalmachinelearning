# Prediction Assignment Writeup
Kelsey Messonnier  



#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#How the Model was Built
##Reproducibility

The participants were asked to do unilateral bicep burls in five different ways:

* Class A - exactly according to specification
* Class B - throwing the elbows to the front
* Class C - lifting the dumbbell only halfway
* Class D - lowering the dumbbell only halfway
* Class E - throwing the hips to the front. 

The model was built to predict in which of these five ways an exercise was done. 

In order to create reproducible results, a seed of 1000 should be set and all the libraries should be loaded in R.

```r
#Set the seed for reproducibility
set.seed(1000)

#Load libraries
library(ggplot2)
```

```
## Warning: package 'ggplot2' was built under R version 3.3.2
```

```r
library(lattice)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

##Data Preparation 

Several columns needed to be removed from the data before it could be used to model. The ID, username, any timestamps, and any window columns were removed (columns 1 through 7). Then any columns that had more than 75% of data missing were removed. This was done by calculating the rate of missing data for each column.


```r
#Import data file
training_data <- read.csv ("pml-training.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
validation_data <- read.csv ("pml-testing.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

#Remove ID and timestamp columns
training_data <- training_data[,-c(1:7)]
validation_data <- validation_data[,-c(1:7)]

#Remove columns with more than 75% missing data
training_data[training_data == ""] <- NA
NAs1 <- apply(training_data, 2, function(x) sum(is.na(x)))/nrow(training_data)
training_data <- training_data[NAs1<0.75]

#validation_data <- validation_data[,colSums(is.na(validation_data)) == ""]
validation_data[validation_data == ""] <- NA
NAs2 <- apply(validation_data, 2, function(x) sum(is.na(x)))/nrow(validation_data)
validation_data <- validation_data[NAs2<0.75]
```

##Cross Validation

The training data provided was split into two datasets: 75% for training the model and 25% for testing the model. 

```r
#Partition dataset into testing and training for cross validation
inTrain = createDataPartition(training_data$classe, p = 3/4)[[1]]
training = training_data[ inTrain,]
testing = training_data[-inTrain,]
```

Below we look at the dimensions of the new training and testing datasets.

```r
dim(training)
```

```
## [1] 14718    53
```

```r
dim(testing)
```

```
## [1] 4904   53
```

```r
plot(training$classe, main = "Distribution of Target Variable Levels (Classe) in Training Data", xlab = "Classe", ylab = "Frequency")
```

![](Prediction_Assingment_Writeup_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

The graph above shows the frequency of the different categories (A, B, C, D, and E) in the training data. 

#Prediction Model: Random Forest

I decided to use a random forest model because they are useful for categorizing data and correcting overfitting. 

```r
#Create model
rf <- randomForest(classe ~ ., data=training)
rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.43%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4181    3    0    0    1 0.0009557945
## B   10 2833    5    0    0 0.0052668539
## C    0   13 2552    2    0 0.0058433970
## D    0    0   23 2388    1 0.0099502488
## E    0    0    1    4 2701 0.0018477458
```

```r
#Predict on testing dataset
confusionMatrix(testing$classe, predict(rf, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    2  947    0    0    0
##          C    0    3  852    0    0
##          D    0    0    5  798    1
##          E    0    0    0    4  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9947, 0.9981)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9959          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9958   0.9942   0.9950   0.9989
## Specificity            0.9997   0.9995   0.9993   0.9985   0.9990
## Pos Pred Value         0.9993   0.9979   0.9965   0.9925   0.9956
## Neg Pred Value         0.9994   0.9990   0.9988   0.9990   0.9998
## Prevalence             0.2847   0.1939   0.1748   0.1635   0.1831
## Detection Rate         0.2843   0.1931   0.1737   0.1627   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9991   0.9976   0.9967   0.9968   0.9989
```

The random forest model accuracy is 0.996, with 95% Confidence Interval (0.994, 0.998). The expected out of sample effor is 0.45%. The sensitivity is between 0.991 and 0.999 for all classes of exercise A through E. The specificity is between 0.998 and 1.000 for all classes of exercise. 

#Final Validation of Model
##Predict 20 Test Cases

Below shows the results of the random forest model applied to the validation dataset. 


```r
val_prediction <- predict(rf, validation_data)
val_prediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
plot(val_prediction, main = "Distribution of Target Variable Levels (Classe) in Validation Data", xlab = "Classe", ylab = "Frequency")
```

![](Prediction_Assingment_Writeup_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

The graph above shows the frequency distribution for the target variable in the 20 case validation dataset.
