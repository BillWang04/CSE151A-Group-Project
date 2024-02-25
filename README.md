# CSE151A-Group-Project

link to notebook: https://colab.research.google.com/github/BillWang04/CSE151A-Group-Project/blob/main/project.ipynb

### Plan for preprocessing:

We plan to preprocess our data by using some of the packages provided by Sklearn. First, we are planning to add some features to our data that we expect to be useful for our model's prediction. Some of the features are: `day_of_week`, `month`, and `class`. After that, we going to encode our categroical variable `airline` using the *OneHotEncoder*. We are also planning to standarize and normalize our **quantitative** data using the ***StandardScaler*** function and the ***MinMaxScaler*** function. These decisions are made to make our variables look more standard and with respect to their respective mean and variance. This helps us avoid having very large/small weights due to the range of values for the input.


### Model 1: Preprocessing, Training, Evaluation

After training our model, we evaluated its performance using root mean squared error (RMSE) metric. Our model's performance on the training set was extremely weak where we obtained an RMSE of about 7193. We then evaluated the model's performance on the test data and we obtained a similar socre to the training set which is an RMSE of 7242. We plotted the fitting graph in our notebook and found that our model falls in the underfitting simple model region in the fitting graph.

The next two models we were thinking about using are ridge regression and random forest regression. We are considering ridge regression because it uses regularization to prevent overfitting and penalizes large coefficients to reduce model complexity and improve generalization. Ridge regression is also very useful when dealing with multicollinearity within the features. The second model we were thinking about using is random forest regression because random forest regression is less prone to overfitting, as each tree in the random forest is trained on a random subset of the training data and a random subset of features. Random forest regression is useful for capturing nonlinear relationships between features and the target variable.

##### Conclusion:

After evaluating the model, we find that the model is ineffective and clearly underfitting the data. This indicates that our data isn't linear and we should introduce some non-linearity to our model to increase its performance and minimize the RMSE. We are also planning to explore the data more thoroughly and see if there are noise features in our design matrix.
