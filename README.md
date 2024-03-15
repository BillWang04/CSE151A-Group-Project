# CSE151A-Group-Project

Project Notebook: https://colab.research.google.com/github/BillWang04/CSE151A-Group-Project/blob/main/project.ipynb

## Introduction

## Methods

### Data Exploration
![flights_airline](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/bd0a1949-1f78-4316-9ee5-33fd30a409f1)
![price_vs_day](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/41dc56cc-d42d-4da0-827a-1dca93ef9a60)
![price_vs_day_vistara](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/b270a056-e14c-494a-8f57-69f2cf86c008)
![price_by_airline](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/0eef445d-bd35-41e7-8a6a-ee6335d7bfbd)
![departure_vs_price](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/7911f417-67db-452b-a147-bc92cf95e78e)
![flights_by_day](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/9db492e9-e5fc-49a1-8d95-bf76c856f8d1)
![correlation](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/83307072-8442-42d3-8707-77117a3a7e33)
![sampled_pairplot](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/27da58da-3399-4eeb-ad45-ca12b6957181)

### Data Preprocessing

We plan to preprocess our data by using some of the packages provided by Sklearn. First, we plan to add features to our data that we expect to be useful for our model's prediction. Some of the features are: `day_of_week`, `month`, and `class`. After that, we going to encode our categorical variable `airline` using *OneHotEncoder*. We also plan to standarize and normalize our **quantitative** data using the ***StandardScaler*** function and the ***MinMaxScaler*** function. We are standardizing and normalizing our data to make our variables look more standard and with respect to their respective mean and variance. This helps us avoid having very large/small weights due to the range of values for the input.


### Model 1: Linear Regression

After training our model, we evaluated its performance using root mean squared error (RMSE) metric. Our model's performance on the training set was extremely weak, as we obtained an RMSE of about 7193. We then evaluated the model's performance on the test data and we obtained a similar score to the training set: an RMSE of 7242. We plotted the fitting graph in our notebook and found that our model falls in the underfitting simple model region in the fitting graph.

The next two models we were thinking about using are ridge regression and random forest regression. We are considering ridge regression because it uses regularization to prevent overfitting and penalizes large coefficients to reduce model complexity and improve generalization. Ridge regression is also very useful when dealing with multicollinearity within the features. The second model we were thinking about using is random forest regression because random forest regression is less prone to overfitting, as each tree in the random forest is trained on a random subset of the training data and a random subset of features. Random forest regression is useful for capturing nonlinear relationships between features and the target variable.

##### Conclusion for Model 1:

After evaluating the model, we found that the model is ineffective and clearly underfitting the data. This indicates that our data isn't linear and we should introduce some non-linearity to our model to increase its performance and minimize the RMSE. We are also planning to explore the data more thoroughly to see if there are noisy features in our design matrix.

### Model 2: Random Forest Regression with hyper parameter tuning and K-fold CV 
For our second model, we used Random Forest Regression. After training our model, we evaluated its performance using the root mean squared error. Our second model has improved, decreasing the testing set’s RMSE to 4,127. The first model’s testing set was 7,242 which makes a difference of 3,116 RMSE between the 2 models. This decrease in error shows that our model has improved. We plotted the fitting graph in our notebook and found that our model continues to fall in the underfitting simple model region in the fitting graph. However, it has moved closer to the ideal range than that of the first model. 
Our group performed hyper parameter tuning and K-fold cross validation. For hyperparameter tuning, we used RandomizedSearchCV and our K-fold cross validation was set to five. 

##### Conclusion for Model 2:

After evaluating the model, we found that the model is improving but is still ineffective. It’s still underfitting the data. Introducing the non-linearity to our model did help but now we must focus on exploring the data to see if there is still some cleaning to do. Also, this could indicate that we can reexamine our hyper parameter tuning to see if it can be further improved for optimization. 

### Model 3: Extreme Gradient Boosting Regression

## Results
![linreg_testing](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/b0666319-9ef9-45b8-934b-4a8ec8530223)
![linreg_training](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/11f81634-016e-48c9-aece-458f5a501b81)
![decisiontree1](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/1c0c1d39-305e-45b2-b7c6-ffe392f49acc)
![decisiontree2](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/68cf5ba0-8cce-4e00-b15a-def5e1e8a11e)
![xgb_residuals](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/9f6bec51-a0ea-4e56-b300-07a53378e11a)
![xgb_feature_importance](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/c1954d74-f8cc-4440-b14c-9208a5d03754)
![xgb_pred](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/5205d248-9db9-4f29-9970-3a8062df8198)
![model_complexities](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/09693fa3-7a1b-4537-a2d0-3ad8e4057991)

## Discussion

## Conclusion

## Collaboration
- Hillary Chang 
- Kailey Wong
- Alan Espinosa
- Bill Wang
- Nathan Ko
- Philemon Putra
- Royce Huang
- Walter Wong
- Ahmed Mostafa
- Phiroze Duggal

