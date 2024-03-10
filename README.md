# CSE151A-Group-Project

link to notebook: https://colab.research.google.com/github/BillWang04/CSE151A-Group-Project/blob/main/project.ipynb

### Plan for preprocessing:

We plan to preprocess our data by using some of the packages provided by Sklearn. First, we plan to add features to our data that we expect to be useful for our model's prediction. Some of the features are: `day_of_week`, `month`, and `class`. After that, we going to encode our categorical variable `airline` using *OneHotEncoder*. We also plan to standarize and normalize our **quantitative** data using the ***StandardScaler*** function and the ***MinMaxScaler*** function. We are standardizing and normalizing our data to make our variables look more standard and with respect to their respective mean and variance. This helps us avoid having very large/small weights due to the range of values for the input.


### Model 1: Preprocessing, Training, Evaluation

After training our model, we evaluated its performance using root mean squared error (RMSE) metric. Our model's performance on the training set was extremely weak, as we obtained an RMSE of about 7193. We then evaluated the model's performance on the test data and we obtained a similar score to the training set: an RMSE of 7242. We plotted the fitting graph in our notebook and found that our model falls in the underfitting simple model region in the fitting graph.

The next two models we were thinking about using are ridge regression and random forest regression. We are considering ridge regression because it uses regularization to prevent overfitting and penalizes large coefficients to reduce model complexity and improve generalization. Ridge regression is also very useful when dealing with multicollinearity within the features. The second model we were thinking about using is random forest regression because random forest regression is less prone to overfitting, as each tree in the random forest is trained on a random subset of the training data and a random subset of features. Random forest regression is useful for capturing nonlinear relationships between features and the target variable.

##### Conclusion for Model 1:

After evaluating the model, we found that the model is ineffective and clearly underfitting the data. This indicates that our data isn't linear and we should introduce some non-linearity to our model to increase its performance and minimize the RMSE. We are also planning to explore the data more thoroughly to see if there are noisy features in our design matrix.

#####################################################################################################

### Model 2: Random Forest Regression with hyper parameter tuning and K-fold CV 
For our second model, we used Random Forest Regression. After training our model, we evaluated its performance using the root mean squared error. Our second model has improved, decreasing the testing set’s RMSE to 4,127. The first model’s testing set was 7,242 which makes a difference of 3,116 RMSE between the 2 models. This decrease in error shows that our model has improved. We plotted the fitting graph in our notebook and found that our model continues to fall in the underfitting simple model region in the fitting graph. However, it has moved closer to the ideal range than that of the first model. 
Our group performed hyper parameter tuning and K-fold cross validation. For hyperparameter tuning, we used RandomizedSearchCV and our K-fold cross validation was set to five. 

##### Conclusion for Model 2 :

After evaluating the model, we found that the model is improving but is still ineffective. It’s still underfitting the data. Introducing the non-linearity to our model did help but now we must focus on exploring the data to see if there is still some cleaning to do. Also, this could indicate that we can reexamine our hyper parameter tuning to see if it can be further improved for optimization. 

### Plan for Model 3:

The plan for our third model is to implement ridge regression. This regularization technique can help mitigate our issues with underfitting and as well as any potential overfitting by penalizing large coefficient values. We also plan to change the destination and arrival cities to one hot encoding to better categorize and distinguish the different cities in binary forms. We also plan to change num_stops to binary and add a boolean feature to the weekend category. We also plan to change the month category to boolean. The purpose of these changes is to better categorize our data to produce a more accurate fit to predict our data.

