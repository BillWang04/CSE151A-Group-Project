# CSE151A-Group-Project

Project Notebook: https://colab.research.google.com/github/BillWang04/CSE151A-Group-Project/blob/main/project.ipynb

## Introduction

Our project aims to predict flight prices using machine learning techniques to address multifaceted features, such as the variation in price amongst different airlines, the impact of last-minute bookings, the correlation between ticket prices and departure/arrival times, and the influence of source and destination cities on ticket pricings. This project offers valuable insights to both passengers and the airline industry, enabling passengers to make informed decisions and potentially saving money when buying plane tickets, as well as assisting airlines in pricing strategies. The development of accurate predictive models for flight prices promotes transparency, competition, and financial stability within air travel. Having a good predictive model for flight prices is important, as it allows consumers when booking flights, while also enabling airlines to optimize their pricing strategies accurately.

## Prior Milestone Submissions
##### Conclusion for Model 1:

After evaluating the model, we found that the model is ineffective and clearly underfitting the data. This indicates that our data isn't linear and we should introduce some non-linearity to our model to increase its performance and minimize the RMSE. We are also planning to explore the data more thoroughly to see if there are noisy features in our design matrix.

### Model 2: Random Forest Regression with hyper parameter tuning and K-fold CV 
For our second model, we used Random Forest Regression. After training our model, we evaluated its performance using the root mean squared error. Our second model has improved, decreasing the testing set’s RMSE to 4,127. The first model’s testing set was 7,242 which makes a difference of 3,116 RMSE between the 2 models. This decrease in error shows that our model has improved. We plotted the fitting graph in our notebook and found that our model continues to fall in the underfitting simple model region in the fitting graph. However, it has moved closer to the ideal range than that of the first model. 
Our group performed hyper parameter tuning and K-fold cross validation. For hyperparameter tuning, we used RandomizedSearchCV and our K-fold cross validation was set to five. 

##### Conclusion for Model 2:

After evaluating the model, we found that the model is improving but is still ineffective. It’s still underfitting the data. Introducing the non-linearity to our model did help but now we must focus on exploring the data to see if there is still some cleaning to do. Also, this could indicate that we can reexamine our hyper parameter tuning to see if it can be further improved for optimization. 

### Model 3: Extreme Gradient Boosting Regression
For our third model, we used Extreme Gradient Boosting Regression. After training our XGBRegressor, we again evaluated its performance using root mean squared error as the metric. The RMSE was around the same as the Random Forest Regressor, with the lowest RMSE achieved being 4192 after some hyperparamter. Hyperparamter tuning was performed using GridSearchCV, with n_estimators, max_depth, and learning_rate being the hyperparamters being tuned with K-fold corss validation of 5 folds. 

##### Conclusion for Model 3:

The lack of improvement in model error shows that we need to possibly choose a completely different kind of model, engineer more meaningful features or train a neural network with varying activation functions to achieve a better predictive tool.


## Methods

### Data Exploration
We explored the data through several different pair plots, a correlation matrix, and plot figures. With these implementations, we were able to differentiate certain impacts the attributes have and identify model coefficients. 
The methods we used to explore data includes: pie charts, box plots, scatterplots, count plots, heat maps, correlation matrix.
Here are some examples of data explored and the methods we used to visualize the data. 

![flights_airline](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/bd0a1949-1f78-4316-9ee5-33fd30a409f1)
Example 1) Airline Vistara has the most flights compared to the other 8 airlines in our dataset. Air India has the second highest and Indigo has the third highest. A pie chart was used to visualize this.
```py
plt.figure(figsize=(10, 15))
airline_counts = cleaned_df['airline'].value_counts()
plt.pie(airline_counts, labels=airline_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Number of Flights by Airline')
plt.show()
```
![price_vs_day](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/41dc56cc-d42d-4da0-827a-1dca93ef9a60)
Example 2) The ticket prices per day were all around the same. Saturday seems to have higher ticket prices. Ticket prices on Tuesday, Friday, and Sunday seem to have the highest outliers. A box plot was used to visualize this.
```py
plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='price', data=cleaned_df)
plt.title('Price vs Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Price')
plt.show()
```
![price_vs_day_vistara](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/b270a056-e14c-494a-8f57-69f2cf86c008)
![price_by_airline](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/0eef445d-bd35-41e7-8a6a-ee6335d7bfbd)
![departure_vs_price](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/7911f417-67db-452b-a147-bc92cf95e78e)
Example 3) For the attribute Time, we used a scatter plot and we see that flights departing between 11pm-5am are less common and cheaper than other departure times. 
```py
plt.figure(figsize=(12, 8))
sns.scatterplot(x='dep_time', y='price', data=cleaned_df, alpha=0.5)
plt.title('Departure Time vs. Price')
plt.xlabel('Departure Time')
plt.ylabel('Price')
plt.show()
```

![flights_by_day](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/9db492e9-e5fc-49a1-8d95-bf76c856f8d1)
Example 4) For the attribute Days,  the count Each day has the same amount of flight data
 ```py
plt.figure(figsize=(10, 6))
sns.countplot(x='day_of_week', data=cleaned_df)
plt.title('Number of Flights by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Flights')
plt.show()
```
![correlation](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/83307072-8442-42d3-8707-77117a3a7e33)
![sampled_pairplot](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/27da58da-3399-4eeb-ad45-ca12b6957181)

### Data Preprocessing

#### Part 1: Cleaning

```py
def clean(df, class_):
    def extract_stops(description):
        stops_match = re.search(r'(\d+)-?stop', description)
        if stops_match:
            return int(stops_match.group(1))
        else:
            return 0
    def duration_to_hours(duration):
        hours, minutes = duration.split('h ')
        hours = float(hours.strip()) if len(hours.strip()) > 1 else 0
        minutes = int(minutes[:-1].strip()) if len(minutes.strip()) > 1 else 0
        total_hours = hours + minutes / 60
        return round(total_hours, 3)
    df_copy = df.copy()
    df_copy["price"] = df_copy["price"].str.replace("," , "").astype(int)
    df_copy["flight_code"] = df_copy["ch_code"].astype(str).str.cat(df_copy["num_code"].astype(str), sep="_")
    df_copy['num_stops'] = df_copy['stop'].apply(extract_stops)
    df_copy["time_taken"] = df_copy["time_taken"].apply(duration_to_hours)
    df_copy['date'] = pd.to_datetime(df_copy['date'], format="%d-%m-%Y")
    df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['arr_time'] = pd.to_datetime(df_copy['arr_time'])
    df_copy['arr_time'] = df_copy['arr_time'].dt.hour + (df_copy['arr_time'].dt.minute >= 30)
    df_copy['dep_time'] = pd.to_datetime(df_copy['dep_time'])
    df_copy['dep_time'] = df_copy['dep_time'].dt.hour + (df_copy['dep_time'].dt.minute >= 30)
    df_copy = df_copy.drop(columns= ["date", "ch_code", "num_code", "stop"])
    df_copy["class"] = class_
    df_copy = df_copy[["airline", "flight_code", "class", "from", "to", "time_taken", "dep_time", "arr_time", "num_stops", "month", "day_of_week", "price"]]
    return df_copy
```

- `price`:  
    - Price is currently in string format, replace the `,` with “” and turn into int  
- `flight_code`:  
    - Grabs `ch_code` and `num_code` and combines the two columns into the flight code  
- `num_stops`:  
    - Searches for a pattern like "X-stop" or "Xstop" in the description string in the ‘stop’ column. Attributes the number of stops for that flight  
- `time_taken`:   
    - Converts `time_taken` column in original df to a float  
- `date`, `day_of_week`, `month`:  
    - Convert `date` to a date_time object  
    - Convert `day_of_week` to a date_time object  
    - Convert `month` to a date_time object  
- `arriva_timel` and `departure_time`:
    - To datetime format and extracts the hour part, considering minutes >= 30 as the next hour.  
- `class`:  
    - Added class `Business` or `Economy` to each flight


```py
cleaned_df = pd.concat([clean(economy, "economy"), clean(business, "business")])
cleaned_df
```
The clean function is called twice, once for the economy class DataFrame and once for the business class DataFrame.
The clean function takes the respective DataFrame (economy or business) and a string representing the class ("economy" or "business").
It performs the necessary data cleaning and preprocessing steps, as defined in the clean function, and returns the cleaned DataFrame.
The pd.concat function concatenates the two cleaned DataFrames along the row axis (vertically).
clean(economy, "economy") returns the cleaned DataFrame for economy class flights.
clean(business, "business") returns the cleaned DataFrame for business class flights.
The pd.concat function combines these two DataFrames into a single DataFrame, cleaned_df.

The resulting cleaned_df contains all the rows from both the economy and business class DataFrames, with the respective "class" column indicating whether the row belongs to the economy or business class.


#### Part 2: Feature Engineering
```py
prepoc = ColumnTransformer([
   ('airline', OneHotEncoder(handle_unknown='ignore'), ['airline', 'class']), #One Hot Encoding airline and class
   ('log', FunctionTransformer(lambda x: np.log(x + 0.001)), ['time_taken']), # Loging Time Taken
   ('test', FunctionTransformer(lambda x: x), ['dep_time', 'arr_time', 'num_stops', 'weekend']), # Keeping all of the values
], remainder='drop')
```

We used a ColumnTransformer in a Pipeline to perform feature engineering on our cleaned data. The first line involves one hot encoding the categorical variables of airline and class. The airline variable is which airline that flight was for and the possible airlines in the dataset were Vistara, Air India, Indigo, GO FIRST, AirAsia, SpiceJet, StarAir, and Trujet. The class variable was simply whether the ticket was for business or economy class. The second line takes the logarithmic transformation of the time taken column. Finally, the last line is just to specify which features to keep as is, as the rest were going to be dropped. We kept departure time, arrival time, number of stops, and whether the flight was on the weekend as is.

### Model 1: Linear Regression

For our first model, we built a Linear regression model and checked its error with Root MSE. 
We plotted a Model Complexity vs. Predictive Error plot to check where our model lies in respect to the ideal range for model complexity and to determine whether the model is underfitting or overfitting. 

```py
pipe = Pipeline([
       ('prepoc', prepoc),
       ('pog', LinearRegression())
   ])


# Fit the pipeline to the training data
pipe.fit(X_train, y_train)


# # Predict on the testing data
predictions = pipe.predict(X_test)
```

### Model 2: Random Forest Regression 
For our second model, we built a Random Forest Regression model and implemented cross validation along with hyper parameter tuning. Also, we continued to use Root MSE. We used RandomizedSearchCV to perform hyperparameter tuning on the model. We plotted a Model Complexity vs. Predictive Error plot to check where our model lies in respect to the ideal range and to compare it with model 1. 
```py
pipe = Pipeline([
       ('prepoc', prepoc),
       ('pog', RandomForestRegressor())
   ])


pipe.fit(X_train, y_train)


predictions = pipe.predict(X_test)


param_dist = {
   'pog__n_estimators': randint(50, 300),
   'pog__max_features': randint(1, 20),
   'pog__min_samples_split': randint(2, 15)
}


random_search = RandomizedSearchCV(
   pipe,
   param_distributions=param_dist,
   n_iter=10,
   cv=5,    
   scoring='neg_mean_squared_error', 
   verbose=1,
   n_jobs=-1,
   random_state=66  
)


random_search.fit(X_train, y_train)
random_search.best_params_
best_params = {
   'n_estimators': 249,
   'max_features': 2,
   'min_samples_split': 14
}


pipe = Pipeline([
       ('prepoc', prepoc),
       ('pog', RandomForestRegressor(**best_params))
   ])


pipe.fit(X_train, y_train)


predictions = pipe.predict(X_test)
```

### Model 3: Extreme Gradient Boosting Regression
For our third model, we built an Extreme Gradient Boosting Regressor model and continued to use Root MSE. We plotted a Model Complexity vs. Predictive Error plot to check where our model lies in respect to the ideal range for model complexity, to compare the model with the first and second model for improvement , and to determine whether the model is underfitting or overfitting. Since this was our final model, we decided to run a GridSearchCV to attempt to get the best hyperparameters.
```py
pipe = Pipeline([
       ('prepoc', prepoc),
       ('pog', XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=20, random_state = 22))
   ])
pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)
param_grid = {'pog__n_estimators': [80, 120, 160, 250],
              'pog__learning_rate': [0.2, 0.1, 0.01, 0.001],
              'pog__max_depth': [5, 7, 10, 15]}

grid_search = GridSearchCV(pipe, param_grid, cv=5)

grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
RMSE_test = np.sqrt(np.sum((y_test - y_pred)**2) / y_test.shape[0])
grid_search.best_params_
```


## Results
### Linear Testing
![linreg_testing](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/b0666319-9ef9-45b8-934b-4a8ec8530223)
![linreg_training](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/11f81634-016e-48c9-aece-458f5a501b81)


For the training set, the RMSE value of 7193.054746715674 indicates that, on average, the model's predictions deviate from the true prices by approximately $7,193. Similarly, for the testing set, the RMSE value of 7242.05710648803 suggests that the model's predictions deviate from the actual prices by around $7,242 on average.





### Decision Tree Testing
![decisiontree1](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/1c0c1d39-305e-45b2-b7c6-ffe392f49acc)
![decisiontree2](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/68cf5ba0-8cce-4e00-b15a-def5e1e8a11e)


For the training set, the RMSE value of 3768.614335146322 indicates that, on average, the model's predictions deviate from the true prices by approximately $3,768. This value is lower than the RMSE for the training set of the linear regression model, suggesting that the Random Forest Regressor is able to better capture the patterns in the training data.

Similarly, for the testing set, the RMSE value of 4126.950091324907 suggests that the model's predictions deviate from the actual prices by around $4,127 on average. While this value is higher than the training set RMSE, it is still lower than the RMSE for the testing set of the linear regression model.


### XG-Boost Tree Testing
![xgb_residuals](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/9f6bec51-a0ea-4e56-b300-07a53378e11a)
![xgb_feature_importance](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/c1954d74-f8cc-4440-b14c-9208a5d03754)
![xgb_pred](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/5205d248-9db9-4f29-9970-3a8062df8198)

The (RMSE) value of the test was 4194.605297728842 for the Random Forest Regressor with XGBoost, it appears that the model's performance is similar to the previous Random Forest Regressor model without XGBoost.

In many cases, the class of travel (business or economy) is one of the primary determinants of flight prices, as business class tickets typically have a significantly higher price point compared to economy class tickets. As shown in the graph, the most important feature is this class which suggests that the model is mainly using the class as a means to predict the price of flights.

### Model Complexities Testing
![model_complexities](https://github.com/BillWang04/CSE151A-Group-Project/assets/61530252/09693fa3-7a1b-4537-a2d0-3ad8e4057991)

## Discussion

### Exploratory Data Visualization

Through a meticulous exploratory data analysis, we gained invaluable insights into the  structure, quality, and  characteristics of our dataset. This comprehensive examination proved instrumental in identifying opportunities for feature engineering, enabling us to craft more robust and informative features for our machine learning models. One notable discovery during this process was the presence of numerous outliers with exorbitant prices, particularly on weekends (Friday, Saturday, and Sunday), despite the overall similarity in pricing patterns. Recognizing the potential significance of this observation, we strategically engineered a dedicated 'weekend' feature to capture and account for this nuanced behavior.

Furthermore, our analysis delved into the distribution of data across different days, as we looked for any discernible patterns or disparities in flight operations. However, our findings revealed no substantial deviations, rendering additional feature engineering efforts in this regard unnecessary. Moreover, our exploration unveiled a notable concentration within the dataset, with the airlines Vistara and Air India dominating the majority of the observations and exhibiting the most extensive price distribution.

### Data Preprocessing Part 1: Data Cleaning

We used the business and economy DataFrames to make them suitable for feature engineering tasks. It performs a series of data cleaning and preprocessing operations on both DataFrames by applying the clean function to each of them individually. The clean function carries out various data transformations, such as removing unnecessary characters, converting data types, extracting relevant information from existing columns, and creating new columns based on the existing data.

After cleaning and preprocessing the business and economy DataFrames separately, the code combines them into a single DataFrame called cleaned_df using the pd.concat function. This merged DataFrame contains all the rows from both the business and economy DataFrames, with an additional column "class" indicating whether a particular row belongs to the business or economy class.

The resulting cleaned_df DataFrame is now structured and formatted in a way that makes it easier to perform feature engineering tasks, which typically involve creating new features or transforming existing features to better represent the underlying data patterns and relationships. The cleaned and merged DataFrame can be used as input for further data analysis, modeling, or machine learning pipelines.

### Data Preprocessing Part 2:

We first decided to one-hot-encode the airline variables as we believed different airlines would have different pricing, as some airlines are on the more budget end while others are more luxury with more amenities. Then, we decided to one-hot-encode class as we were confident that whether a ticket was business or economy would play a significant role in determining the price.

We then decided to perform a logarithmic transformation on the time taken feature (flight duration). We decided to do a log transform as much of the data was pretty right skewed. Since we are only using linear features, this allows the time taken column to be better fit.

Finally, we decided to simply keep the departure time, arrival time, number of stops, and whether the flight was on the weekend as we believed these features are useful and look workable as is. The other features we dropped were mostly beacause we believed they did not play a huge role in determining price, or were collinear with another feature.

### Model 1
The reason we created a linear model was to have baseline as a way to compare our results to the next two models.The linear regression model is struggling to accurately predict flight prices across the entire range of values. The high RMSE values suggest that the model's predictions are deviating significantly from the actual prices.

These high RMSE values could be attributed to the fact that the model is treating the flight prices as two distinct groups or clusters: one around $1,000 for economy flights and another around $5,000 for business flights. Instead of accurately predicting the individual prices, the model seems to be classifying or grouping the prices into these two broad categories.

This behavior might be due to the inherent limitations of linear regression models in capturing complex, non-linear relationships between the features and the target variable. Additionally, the presence of outliers or skewed data distributions could also contribute to the model's inability to make precise predictions across the entire range of flight prices.

### Model 2

The reason we chose a the Random Forest Regressor's is because of its ability to capture non-linear relationships and handle complex interactions between features might be contributing to its improved performance. The lower RMSE values for both the training and testing sets indicate that the Random Forest Regressor is performing better than the linear regression model in predicting flight prices. 


### Model 3

After noticing the dramatic improvement from linear regression to random forest, we decided to implement an XGBoost model to further improve our predictions. Our supposition was that XGBoost gives us more flexibility in hyperparameter tuning than random forest. XGBoost lets us specify the learning rate of our model along with the structure of the tree built for prediction. However, after building the model, we found that the performance of the model is pretty much the same as the random forest. Even after the hyperparameter tuning and using gridsearch to find the best hyperparameter, the model dealt no better results. We anticipate that the problem is from our features and the small size of our dataset as XGBoost stands out with larger datasets.

## Conclusion

In conclusion, our project has provided valuable insights into flight price prediction, emphasizing the importance of data analysis, feature engineering, and model selection in prediction models. There are areas where we could have taken a different approach for better outcomes. For example, our initial data exploration could have been more thorough, identifying noisy features and understanding the relationships between variables more deeply. In the preprocessing stage, while we employed standard techniques, we could have also looked into domain knowledge to enhance model performance. This would include feature engineering more meaningful variables when it comes to differing flights and flight tickets. Additionally, we could have experimented with more algorithms, instead of just linear regression, ridge regression, random forest regression, and extreme gradient boosting regression. We could have implemented a neural network with different activation functions to try to create a better predictive model. The biggest oversight for our modeling process was that the one-hot-encoded feature of class was by far the most significant feature when it came to predicting price. In the future, we would either keep the business and economy class datasets separate, or create a binary classifier to classify the two groups first if the class data is not given. Despite these potential improvements and overlooked aspects, our project still proved insightful and provided a valuable learning experience for all of us.

## Collaboration
- Hillary Chang: Made visualizations, helped with the writeup
- Kailey Wong: Assisted with performing initial data exploration and visualizations, brainstormed and created model and performance visualizations.
- Alan Espinosa: Helped with preprocessing the data, updated the ReadMe file by describing how we built our second model and how well it performed compared to our first model. 
- Bill Wang: Helped implement all three models, feature engineering and synthesized the write-up.
- Nathan Ko: Helped implement all three models, feature engineering and synthesized the write-up.
- Philemon Putra: Helped with preprocessing the data, updated the ReadMe file by describing our ideas and plans for the third model.
- Royce Huang: Assisted with preliminary exploratory data analysis in generating visualizations, and reviewed the initial abstract and research questions
- Walter Wong: Helped implement all three models, feature engineering and synthesized the write-up.
- Ahmed Mostafa: Cleaned the data, assisted in plotting the model performance plot, modified the readme, and helped in implementing the third model.
- Phiroze Duggal: Helped with plotting the model performance plot, and helped in editing the readme
