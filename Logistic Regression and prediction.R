install.packages("XLConnect")
install.packages("tidyverse")
install.packages("car")
install.packages("dplyr")
install.packages("plyr")
install.packages("Amelia")
install.packages("caTools")
install.packages("ROCR")
library(ROCR)
library(XLConnect)
library(tidyverse)
library(dplyr)
library(plyr)
library(Amelia)
library(ModelMetrics)

#rain_df$WDF5 >45 & rain_df$WDF5 <=135, 'East', ifelse(

#Read the climate data from the csv

rain_data <- read.csv("G:\\RProject\\SeattleRain Logistic Regression\\RainSeattle2016.csv", header=T,na.strings = c(""))

#Explain the dataset
#Rain is the column which we are trying to predict.
#NAME is the place where the weather data was recorded on respective DATE.
#PRCP and SNOW = Precipitation and Snow in inches
#SNWD = Snow depth
#TAVG = Average Temperature in F
#TMAX, TMIN = max and Min Temperature of the day in F.
#WDF5, WSF5 - Direction of fastest 5-second wind in degree and Fastest 5-second wind speed respectively.
summary(rain_data)

#Check for missing values.
#Amelia library gives us a missmap function that shows the missing details in a visual map.
missmap(rain_data,main="Missing Values in Data Set")

#We have lots of missing values in Fog column. And high winds, sleet, hail, smoke, thunder and heavy fog are unuseable as they have almost all missing values.
#Therefore, we can exclude them from our model.
#Name, Days, DATE, PRCP and SNWD can also be ignored.
#Only include the columns that we would be using for building a model
#In our case, we only use Rain, Season, Ave.wind, PRCP, TAVG, TMAX, TMIN, WDF5 and WSF5
rain_df <- subset(rain_data,select=c(1,3,4,5,7,8,9,10,11,12,13))

#Sneak-peak into the data
head(rain_df)

#Rain     DATE Season Ave.Wind SNOW SNWD TAVG TMAX TMIN WDF5 WSF5
#1    0 1/1/2016 Winter     4.92    0    0   36   46   28   70 13.0
#2    0 1/2/2016 Winter     5.59    0    0   33   42   25  110 29.1
#3    1 1/3/2016 Winter     9.17    0    0   37   40   31  110 36.9
#4    1 1/4/2016 Winter     7.61    0    0   36   38   35   10 17.0
#5    1 1/5/2016 Winter     5.37    0    0   38   46   36   80 18.1
#6    0 1/6/2016 Winter     6.49    0    0   44   53   37  100 16.1

#Before we proceed, let's see if our dataset has any NA values.
#In case missing values are observed, we need to account for them.
#One of the most common ways to exclude the rows containing NA values or fit missing values by inserting the median value of that column.
#One can also plot box-plots to find appropriate value for the missing value. Or use Knn/mice/Amelia imputation.
sapply(rain_df,function(x) sum(is.na(x)))

#We don't have missing values.

#Now that we have the data in place, we need to modify the few columns from continuous type to categorical type.
#In many data analysis settings, it might be useful to break up a continuous variable such as age into a categorical variable. 
#Or, one might want to classify a categorical variable like year into a larger bin, such as 1990-2000

#Here, we categorize two columns in the dataset.
#First, Rain is converted to a factor with 2 levels i.e 0 and 1. 0 indicates no rain, whereas 1 indicates rainfall on that day.

rain_df$Rain <- factor(rain_df$Rain)

#Next, we create an unordered categorical variable for Season.

rain_df$Season <- factor(rain_df$Season,ordered = FALSE)
#Season is now a factorial variable with 4 levels i.e 'Fall', 'Soring', 'Summer', 'Winter'. 
#Ordered = FALSE indicates R that the categories are not ordinal.

#We  need to convert DATE from character type to date type.
#We do that by using the as.Date() function.
rain_df$DATE <- as.Date(rain_df$DATE,'%m/%d/%Y')


#Before we proceed, let's see how R is going to deal with the categorical variables. We can use the contrasts() function for this. 
#This function shows us how the variables have been dummyfied by R and how to interpret them in a model.
#It shows us the reference values for each factorial variable.

contrasts(rain_df$Rain)
contrasts(rain_df$Season)

#In Season category, class 'Fall' is used as reference.

#The Generalized Linear Models in R internally encode categorical variables into n - 1 distinct levels.
#Thus, glm() in R will use one category of Season and use it as reference against remaining three categories in the model.
#Model statistics will indicate the significance of each of those three levels.
#Later, we will check if the categorical variable Season is statistically significant as a whole.


#We split the data into training and test data set. Training set will be used to build the model
#And test data can be used to tes the model fit.
#One can also use n-fold cross validation for better results.

#We will split 80% of data into training and 20% into test set.
set.seed(88)
split <- sample.split(rain_df$Rain,SplitRatio = 0.8)

#Build the training and test chunks.
train_df <- subset(rain_df,split==TRUE)
test_df <- subset(rain_df,split==FALSE)

#Quick Summary of the two data sets.
summary(train_df)
summary(test_df)

#Let's fit the model to our data.
#The code below estimates a logistic regression model using the glm (generalized linear model) function. 
#Be sure to specify the parameter family=binomial in the glm() function.


glm1 <- glm(Rain ~ Season+Ave.Wind+SNOW+TAVG+TMAX+TMIN+WDF5+WSF5,family = binomial(link = "logit"),data = train_df)

#Model statistics
summary(glm1)

#Interpreting the results of our regression model

#The glm function internally encodes categorical variables into n - 1 distinct levels, as mentioned earlier.
#Here, the regression coefficients explain the change in log(odds) of the response variable for one unit change in the predictor variable.
#Std. Error represents the standard error associated with the regression coefficients.
#z value(sometimes called a Wald z-statistic) is analogous to t-statistics in multiple regression output.
#p value has the same interpretation as that in linear regression. With 95% confidence level, a variable having p < 0.05 is considered an important predictor.

#In the above model(glm1), SNOW is least significant. And TMAX is the most significant.
#This means, with all other variables remaining constant, SNOW has the least effect on rain probablity and TMAX has highest effect.
#For every one unit change in TMAX, log odds of rain reduces by 2.902e-01 times.

#Below the table of coefficients are fit indices, including the null and deviance residuals and the AIC. 
#AIC and the residuals are crucial in determining model fit. Model with lower AIC is almost always preferred.

#Next step is to analyze the deviance table.
#We do that by doing a ANOVA Chi-square test to check the overall effect of variables on the dependent variable.

anova(glm1,test='Chisq')

#Null Deviance - Residual Deviance shows our model's performance as compared to a model with no variables(only the intercept) a.k.a the null model.
#The wider the gap, the better is the model.
#In addition, lower null and residual deviance are preferred, whereever applicable.
#Every time a variable is added, deviance drops. We need to identify those variable that do not cause large drop in deviance.
#A large p-value here indicates that the model without the variable explains more or less the same amount of variation.

#It's interesting to notice that Season as a whole is highly significant, but at categorical level, Season(or rather Spring/Summer/Winter) are not significant, not at least in the first model.
#Here, variables WDF5, SNOW and TAVG are not significant. We can remove them sequentially and rebuild glm models and repeat the whole process until all variables are significant.
#AIC for glm1 was 257.56

#After sequentially eliminating SNOW followed by WDF5 and TAVG, we come accross the below model.
glm2 <- glm(Rain ~ Season+Ave.Wind+TMAX+TMIN+WSF5,family = binomial(link = "logit"),data = train_df)
summary(glm2)

#Note that, the Ave.Wind variable is being considered as non-significant. Let's see the ANOVA deviance table.

anova(glm2, test='Chisq')

#ANOVA instead tells us that Ave.Wind is significant. It is important to note how anova performs the test.
#sequentially compares the smaller model with the next more complex model by adding one variable in each step. Each of those comparisons is done via a likelihood ratio test.

#We need to build two models, one with Ave.wind and another without Ave.WInd and then compare the two models and pick the better among them.
#Note the AIC of glm2 = 252.95
#Remove Ave.Wind and build a model
glm3 <- glm(Rain ~ Season+TMAX+TMIN+WSF5,family = binomial(link = "logit"),data = train_df)
summary(glm3)
anova(glm3, test='Chisq')
#Compare the two models in anova

#Our NULL hypothesis is that co-efficient of Ave.Wind is 0. 
#Therefore, if p <0.05, we can reject NULL hypothesis and say that the additional variable must remain in the final model.
anova(glm2,glm3,test='Chisq')

#P = 0.15 and we stick with NULL hypothesis i.e co-efficient of Ave.Wind is 0. 
#Therefore, we pick glm3 to proceed.

#Looking back at coefficients of glm3, we can see Season categorical classes, individually, may or may not be significant.
#In glm(),for each coefficient of every level of the categorical variable, a Wald test is performed to test whether the pairwise difference between the coefficient of the reference class and the other class is different from zero or not. 


#If only one categorical class is insignificant, this does not imply that the whole variable is meaningless and should be removed from the model. 
#Moreover, the categorical class coefficients denote the difference of each class to the reference class. 
#Therefore, they only tell us about the pairwise differences between the levels. To test whether the categorical predictor, as a whole, is significant is equivalent to testing whether there is any heterogeneity in the means of the levels of the predictor.
#Therefore, we need to create two models, one with Season and another without. Then compare the models using ANOVA and pick one from them.

glm4 <- glm(Rain ~ TMAX+TMIN+WSF5, family = binomial(link = "logit"),data = train_df)
summary(glm4)

#Let's compare the above two models(one with Seasons and other without) using Anova.

anova(glm3, glm4, test='Chisq')

#P value is significant and we reject the NULL hypothesis(Season's coefficient is zero).
#Therefore, Season must remain in the model.
#AIC for glm4 is 253.02, whereas AIC for glm3 is 256.4. It is another indication that glm4 is a better model.

#We find the odds ratios and 95% CI
exp(cbind(OR = coef(glm3), confint(glm3)))

#Residual Plot for GLM models are not as useful as the one in LM.
par(mfrow=c(2,2))
plot(glm3)
par(mfrow=c(1,1))

#Now that we have the model, let's go ahead and use this model to predict on the test data set.

#By setting the parameter type='response', R will output probabilities in the form of P(y=1|X)
predicted <- predict(glm3,newdata = test_df,type = "response")


#predicted values are in probablity.
#If probability of Rain > 0.5, we assume that it will rain that day and if probablity < 0.5, we can assume it won't rain that day.
#This threshold can be changed based on subject knowledge.
predicted <- ifelse(predicted > 0.5,1,0)

#Print the predictions
head(data.frame(Date = test_df$DATE,Rain_Actual = test_df$Rain, Predicted_Rain = predicted))

#data.frame(Date = test_df$, Rain_Actual = test_df$Rain, Predicted_Rain = predicted)
#One of the most important performance indicator of a model is the confusion matrix.
#It's a tabular representation of Actual vs Predicted values. This helps us to find the accuracy of the model and avoid overfitting.

table(test_df$Rain, predicted)

#Now, we calculate the accuracy of our prediction.
accuracy <- 1-mean(predicted != test_df$Rain)
accuracy

#Accuracy for our model is 0.8219 or 82.2%. 
#This is fair value for our model. Accuracy can be increased by checking the ROCR curve and making necessary adjustment in the probablity threshold.

#Final step is to plot the ROC(Receiver Operating Characteristic) curve and find out the AUC(area under curve)
#ROC summarizes the model's performance by evaluating the trade offs between true positive rate (TPR or sensitivity) and false positive rate(1- specificity). 
#The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. 
#Higher the area under curve, better the prediction power of the model. Below is a sample ROC curve.
#As a rule of thumb, a model with good predictive ability should have an AUC closer to 1 (=1 ideally) than to 0.5.


ROCRPred <- prediction(predicted,test_df$Rain)
ROCRperf <- performance(ROCRPred, measure='tpr', x.measure='fpr')

#Plot the ROC curve

plot(ROCRperf, colorize=TRUE)

#Calculate the Area under curve(AUC)

auc(test_df$Rain,predicted)

#AUC is 0.8239 which is an indication that our model is a good one.

#Summary
#It's a good practice to start any given classification problem with logistic regression and go from there.
#It sort of lays the benchmark accuracy for other non-linear models that one can try afterwards.


#Logistic Regression is one of the most widely used classification algorithms today.
#It has managed to survive Support Vector Machines, Random Forests and Boosting. 
#It is also a crucial part of modern neural networks.
#As Andrew Ng says, Neural networks are nothing but a stack of multiple logistic regression models.


