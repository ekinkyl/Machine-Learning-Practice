import pandas as pd
from sklearn.model_selection import train_test_split #split the dataset into training and testing sets.
from sklearn.preprocessing import StandardScaler #standardize features by removing the mean and scaling to unit variance.
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib #saving the trained model

#load the datates
dataset = pd.read_csv('winequality-red.csv', delimiter=';')

#check for missing values
#print(dataset.isnull().sum)

#process the data
# 1)data cleaning: handling missing values, dealing with outliers, and correcting inconsistencies in the data
# 2)data integration: combining data from multiple sources into one coherent dataset
# 3)data transformation: normalization and standardization of data. Normalization adjusts the numeric values of features to a common scale, often between 0 and 1, making different features comparable. Standardization transforms data to have a mean of 0 and a standard deviation of 1, which can be important for algorithms that assume normally distributed data.
# 4)feature selection/engineering:  creating new features from existing data that might better represent patterns in the data and improve model accuracy.
# 5)data reduction: techniques like Principal Component Analysis (PCA) or feature selection algorithms can reduce the dimensionality of the data while retaining important information.
# 6)data splitting: splitting the dataset into training, validation, and test sets.
# 7)handling categorical data: converting categorical variables into a numerical format that algorithms can process. This might involve techniques like one-hot encoding or label encoding.


#splitting data into features and target variable

x = dataset.drop('quality', axis = 1) #axis=1 specifies that we're dropping a column.
y = dataset['quality'] #this column is our target variable, which we want to predict.

#standardizing the features
scaler = StandardScaler() #standardize the features by removing the mean and scaling to unit variance.
x_scaled = scaler.fit_transform(x) #The result, X_scaled, is the standardized version of x. Standardizing the data can improve the performance of many machine learning algorithms by ensuring that each feature contributes equally.

#splitting the data into training and test sets
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#train the model
#linear regression, decision tree regressor

#initialize the models
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42) #Setting random_state=42 ensures reproducibility of the results.

#train the models
lr_model.fit(x_train,y_train)
dt_model.fit(x_train, y_train)

#evaluate the models: evaluate their performance using the test data. We will use Mean Squared Error (MSE) and R-squared (R²) as the evaluation metrics.

#linear regression

# Evaluate the Linear Regression model
y_pred_lr = lr_model.predict(x_test) #this uses the trained Linear Regression model to make predictions on the test data (X_test).
mse_lr = mean_squared_error(y_test, y_pred_lr) #MSE measures the average squared difference between the actual and predicted values. Lower MSE indicates better performance.
r2_lr = r2_score(y_test, y_pred_lr)

#print("Linear Regression Model Performance:")
#print("Mean Squared Error:", mse_lr)
#print("R-squared:", r2_lr)

# Evaluate the Decision Tree Regressor model
y_pred_dt = dt_model.predict(x_test) 
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

#print("Decision Tree Regressor Model Performance:")
#print("Mean Squared Error:", mse_dt)
#print("R-squared:", r2_dt)

#Hyperparameter Tuning: finding the best set of hyperparameters for a model to improve its performance.

# Define the parameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
#GridSearchCV Object:

#estimator=DecisionTreeRegressor(random_state=42): The model for which we are performing hyperparameter tuning.
#param_grid=param_grid: The parameter grid we defined in the previous step.
#cv=5: Use 5-fold cross-validation. The dataset is split into 5 folds, and the model is trained and evaluated 5 times, each time using a different fold as the test set and the remaining folds as the training set. This helps to ensure the model's performance is robust and not dependent on a particular train-test split.
#scoring='r2': The metric used to evaluate the model's performance. In this case, we are using the R-squared value.
#n_jobs=-1: Use all available CPU cores to perform the search, which speeds up the process.

# Fit the grid search to the data
grid_search.fit(x_train, y_train)
#For each combination, it trains the model using 5-fold cross-validation on the training data (X_train, y_train).
#It evaluates the performance of each combination using the specified scoring metric (R-squared in this case).

# Get the best parameters
best_params = grid_search.best_params_
#print("Best parameters found: ", best_params)

#evaluate the best model
best_dt_model = grid_search.best_estimator_ #retrive the best estimator
y_pred_best_dt = best_dt_model.predict(x_test) #make predictions on the test data
mse_best_dt = mean_squared_error(y_test,y_pred_best_dt)
r2_best_dt = r2_score(y_test, y_pred_best_dt)

#print("Tuned Decision Tree Regressor Model Performance:")
#print("Mean Squared Error:", mse_best_dt)
#print("R-squared:", r2_best_dt)

#Improved Performance:
#Lower MSE: Indicates that the tuned model's predictions are closer to the actual values compared to the untuned model.
#Higher R²: Indicates that the tuned model explains a higher proportion of variance in the target variable. This means the model fits the data better.


#visualization
#The x-coordinate of each point is the actual value (y_test).
#The y-coordinate of each point is the predicted value (y_pred).
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_lr,color='blue',label='Linear Regression')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.savefig('linear_regression_results.png')

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_best_dt, color='green', label='Tuned Decision Tree')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Tuned Decision Tree Regressor: Actual vs Predicted')
plt.legend()
plt.savefig('tuned_decision_tree_results.png')

#Further Hyperparameter Tuning with Randomized Search# Further Hyperparameter Tuning with Randomized Search
param_dist = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'absolute_error']  # Valid criteria for DecisionTreeRegressor
}


random_search = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)
best_random_params = random_search.best_params_ # This retrieves the best combination of parameters found during the random search.
best_random_dt_model = random_search.best_estimator_ # This retrieves the best model with the best parameters.

y_pred_random_best_dt = best_random_dt_model.predict(x_test)
mse_random_best_dt = mean_squared_error(y_test, y_pred_random_best_dt)
r2_random_best_dt = r2_score(y_test, y_pred_random_best_dt)

print("Best parameters from RandomizedSearchCV:", best_random_params)
print("Randomized Search Tuned Decision Tree Regressor Model Performance:")
print("Mean Squared Error:", mse_random_best_dt)
print("R-squared:", r2_random_best_dt)

# Check the prediction values
print("Predicted values:", y_pred_random_best_dt)
print("Actual values:", y_test.values)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_random_best_dt, color='green', label='Further Tuned Decision Tree')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Further Tuned Decision Tree Regressor: Actual vs Predicted')
plt.legend()
plt.savefig('further_tuned_decision_tree_results.png')


print("Linear Regression Model Performance:")
print("Mean Squared Error:", mse_lr)
print("R-squared:", r2_lr)

print("Original Decision Tree Regressor Model Performance:")
print("Mean Squared Error:", mse_dt)
print("R-squared:", r2_dt)

print("Tuned Decision Tree Regressor Model Performance:")
print("Mean Squared Error:", mse_best_dt)
print("R-squared:", r2_best_dt)

print("Further Tuned Decision Tree Regressor Model Performance:")
print("Mean Squared Error:", mse_random_best_dt)
print("R-squared:", r2_random_best_dt)

#Feature importance: helps us understand which features (or columns) in our dataset have the most influence on the predictions made by the model. 

importances = best_random_dt_model.feature_importances_
features = dataset.drop('quality', axis=1).columns

#Create a DataFrame for Feature Importances:
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})

#Sort the DataFrame by Importance:
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importances_df)


# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.savefig('feature_importances.png')  # Save the plot as a PNG file


# Save the scaler and the best model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_random_dt_model, 'best_random_dt_model.pkl') 

# Load the trained model and the scaler
model = joblib.load('best_random_dt_model.pkl')
scaler = joblib.load('scaler.pkl')




# Function to make predictions
def predict_wine_quality(input_data):
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    input_scaled = scaler.transform(input_data)
    predictions = model.predict(input_scaled)
    
    return predictions

# Example input as a dictionary
input_features = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11,
    'total sulfur dioxide': 34,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

predicted_quality = predict_wine_quality(input_features)
print(f"Predicted wine quality: {predicted_quality[0]}")