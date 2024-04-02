#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import plotly.express as px
from ydata_profiling import ProfileReport

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Importing the Boston House Price Dataset

# In[2]:


house_price_dataframe = pd.read_csv("housing.csv")


# In[3]:


house_price_dataframe.head()


# In[4]:


# dataframede satır ve Sütun sayısını kontrol etme
house_price_dataframe.shape


# In[5]:


house_price_dataframe['ocean_proximity'].replace('<1H OCEAN', 'oneH_OCEAN', inplace=True)


# In[6]:


# house_price_dataframe.info()


# cleans the dataset by removing any row that contains missing values in any of its columns,

# In[7]:


house_price_dataframe.info()


# In[8]:


house_price_dataframe.dropna(inplace=True)


# In[9]:


print(house_price_dataframe.isnull().sum())


# In[10]:


house_price_dataframe.info()


# In[11]:


house_price_dataframe['total_rooms'].describe()


# In[12]:


# house_price_dataframe.ocean_proximity.value_counts()


# In[13]:


import pandas as pd

# Assuming df is your DataFrame
# columns_to_exclude = ['longitude', 'latitude', 'ocean_proximity']

columns_to_exclude = ['longitude', 'latitude', 'ocean_proximity']
columns_to_check = house_price_dataframe.columns.difference(columns_to_exclude)

for column in columns_to_check:
    Q1 = house_price_dataframe[column].quantile(0.25)
    Q3 = house_price_dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Update the DataFrame to exclude outliers
    house_price_dataframe = house_price_dataframe[(house_price_dataframe[column] >= lower_bound) & (house_price_dataframe[column] <= upper_bound) | house_price_dataframe[column].isnull()]

# df now has outliers removed from all columns except 'longitude' and 'latitude'


# In[14]:


# house_price_dataframe


# In[15]:


house_price_dataframe.reset_index(inplace=True)
house_price_dataframe.drop(['index'],inplace=True,axis=1)


# In[16]:


house_price_dataframe


# In[17]:


house_price_dataframe = house_price_dataframe.join(pd.get_dummies(house_price_dataframe.ocean_proximity)).drop(['ocean_proximity'], axis=1)


# In[18]:


sns.boxplot(x= house_price_dataframe['total_rooms'])


# In[19]:


house_price_dataframe.columns


# ask muhamede 

# In[20]:


house_price_dataframe.hist(bins=50, figsize=(20, 15))
plt.show()


# In[21]:


# house_price_dataframe['total_rooms'] = np.log(house_price_dataframe['total_rooms'] + 1)
# house_price_dataframe['total_bedrooms'] = np.log(house_price_dataframe['total_bedrooms'] + 1 )
# house_price_dataframe['population'] = np.log(house_price_dataframe['population'] + 1)
# house_price_dataframe['households'] = np.log(house_price_dataframe['households'] + 1)
 


# In[22]:


# # veri kümesinin istatistiksel ölçümleri
# house_price_dataframe.describe().round(2)


# In[23]:


house_price_dataframe.sample(2)


# In[24]:


house_price_dataframe['median_income'].value_counts()


# In[25]:


house_price_dataframe['total_rooms'].describe()


# In[26]:


house_price_dataframe['longitude'].value_counts()


# In[27]:


# house_price_dataframe = house_price_dataframe.rename(columns={"people_per_household": "Avg_people_per_household"})


# In[28]:


# total_rooms_sum = house_price_dataframe['total_rooms'].sum()


# In[29]:


# total_rooms_sum


# In[30]:


# Count the number of houses in each category of ocean proximity
# ocean_proximity_counts =house_price_dataframe['ocean_proximity'].value_counts()

# print(ocean_proximity_counts)



# how is each feature relates to house prices and  show correlation coefficients between median_house_value and other features

# In[31]:


# ordinal_mapping = {'ISLAND': 0, 'oneH_OCEAN': 1, 'INLAND': 2,'NEAR OCEAN': 3,'NEAR BAY': 4}
# house_price_dataframe['ocean_proximity'] = house_price_dataframe['ocean_proximity'].map(ordinal_mapping)


# In[32]:


house_price_dataframe.head()


# In[33]:


house_price_dataframe['bedroom_ratio'] = house_price_dataframe['total_bedrooms'] / house_price_dataframe['total_rooms']
house_price_dataframe['Avg_rooms_per_house'] = house_price_dataframe['total_rooms'] / house_price_dataframe['households']
house_price_dataframe['Avg_people_per_household'] = house_price_dataframe['population'] / house_price_dataframe['households']


# In[34]:


house_price_dataframe.sample(2)


# In[35]:


# house_price_dataframe['Avg_rooms_per_house'].value_counts()


# In[36]:


house_price_dataframe['median_house_value'].max()


# In[37]:


# check for missing values
# house_price_dataframe.isnull().sum()


# How many homes have the lowest or highest value?

# In[38]:


house_price_dataframe['median_house_value'].max()


# In[39]:


value = house_price_dataframe['median_house_value'].max()

(house_price_dataframe['median_house_value'] == value).sum()


# In[40]:


# Adjust the last bin edge to slightly exceed the maximum value or remove the manual max value
max_value = house_price_dataframe['median_house_value'].max() + 1  # ensure inclusion of the max value
bins = [14000, 100000, 250000, 350000, 480000, max_value]
labels = ['14000-100000', '100000-250000', '250000-350000', '350000-480000', '480000+']

# Create a new column 'price_category' to categorize 'median_house_value'
house_price_dataframe['median_house_value_category'] = pd.cut(house_price_dataframe['median_house_value'], bins=bins, labels=labels, include_lowest=True)


# In[41]:


bins = [0, 10, 20, 30, 40, 50, float('inf')]
labels = ['Under 10', '10-19', '20-29', '30-39', '40-49', '50+']
house_price_dataframe['age_category'] = pd.cut(house_price_dataframe['housing_median_age'], bins=bins, labels=labels)


# In[42]:


# house_price_dataframe[['median_house_value', 'median_house_value_category']].sample(2)


# In[43]:


house_price_dataframe.sample(2)


# In[44]:


# house_price_dataframe['median_house_value'].describe()


# In[45]:


# house_price_dataframe['median_house_value'].value_counts()


# In[46]:


# ProfileReport(house_price_dataframe)


# In[47]:


# plt.figure(figsize=(15, 8))
# sns.scatterplot(x="latitude", y="longitude", data=house_price_dataframe, hue="median_house_value", palette="coolwarm")


# In[48]:


# pip install folium


# In[49]:


# import folium
# from folium.plugins import HeatMap

# # Create a base map
# map = folium.Map(location=[36.7783, -119.4179], zoom_start=6)  # Coordinates for California

# # Prepare data for the HeatMap
# # Scale the median_house_value to bring down the values to a narrower range for better visualization
# house_price_dataframe['scaled_price'] = np.log(house_price_dataframe['median_house_value'])

# heat_data = [[row['latitude'], row['longitude'], row['scaled_price']] for index, row in house_price_dataframe.iterrows()]

# # Add heat map
# HeatMap(heat_data, radius=8, gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(map)

# # Display the map
# map

# # Or save to an HTML file


# ####Median House Value

# In[50]:


# house_price_dataframe['population'].value_counts().plot(kind ='bar')


# ######  what is the value of the house in the house that have 5 people 

# In[51]:


# # # house_price_dataframe.groupby('ocean_proximity').sum()['median_house_value']
# house_price_dataframe.groupby('population').mean(numeric_only=True)['median_house_value']


# ######  how many houses in age 1 mean the house it was built before one year

# In[52]:


# house_price_dataframe.groupby('housing_median_age').count()['median_house_value']


# In[53]:


#  house_price_dataframe.groupby('housing_median_age').median()['median_house_value']


# In[ ]:





# In[54]:


# fig = px.bar(
#     data_frame=house_price_dataframe,
#     x = house_price_dataframe.groupby('ocean_proximity').sum(numeric_only=True)['median_house_value'].index,
#     y = house_price_dataframe.groupby('ocean_proximity').sum(numeric_only=True)['median_house_value'].values
# )
# fig.update_xaxes(title='Subject')
# fig.update_yaxes(title='# median_house_value')
# fig.show()


# In[ ]:





# In[55]:


## For 'median_house_value'
# sns.histplot(house_price_dataframe['median_house_value'], kde=True)
# plt.title('Distribution of Median House Value')
# plt.show()


# In[56]:


# sns.boxplot(x=house_price_dataframe['median_house_value'])
# # plt.title('Box Plot of Median House Value')
# # plt.show()


# In[57]:


house_price_dataframe['housing_median_age'].mean()


# ##### we saw when the house is expensiv the ege is small 

# In[58]:


# plt.figure(figsize=(15, 8))
# plt.hexbin(house_price_dataframe['housing_median_age'], house_price_dataframe['median_house_value'], gridsize=50, cmap='plasma', mincnt=2)
# plt.colorbar(label='')
# plt.title('Hexbin plot of Housing Median Age and Median House Value')
# plt.xlabel('Housing Median Age')
# plt.ylabel('Median House Value')
# plt.show()


# ##### 3] Multi-variate Analysis

# In[59]:


# import plotly.express as px

# # Assuming 'NEAR BAY' is a binary variable (e.g., 0 for not near the bay, 1 for near the bay)
# # If 'NEAR BAY' is not already a binary or categorical variable, you may need to adjust the data accordingly.
# fig = px.box(data_frame=house_price_dataframe, 
#              y='median_house_value',
#              color='NEAR BAY',  # This assumes 'NEAR BAY' is a categorical column indicating near or not near the bay.
#              labels={'NEAR BAY': 'Near Bay', 'median_house_value': 'Median House Value'})
# fig.update_layout(title='Distribution of Median House Value by Proximity to Bay',
#                   xaxis_title='Near Bay',
#                   yaxis_title='Median House Value')
# # fig.show()


# #### how many houses near of ocen regarding median_house_value_category

# From the figure, it seems that the majority of the houses in the dataset are not near the ocean (0) and fall within the 14000-100000 value range. The least number of houses are in the highest value category (480000+).
# 
# There is a significantly higher count of houses not near the ocean (0) across all house value categories compared to those near the ocean (1).
# The most common value range for houses not near the ocean is the second lowest value category (100000-250000).
# For houses near the ocean, the most common value range is also the second lowest value category (100000-250000), but the count is much less compared to houses not near the ocean.
# 
# we can see thers is now Cheap house in first categore in 1 and we have litle bit black cuz we just have one house 
# 

# In[60]:


#  import plotly.express as px
# fig = plt.figure(figsize=(15,8))

# palette = {
#     "14000-100000": "yellow",
#     "100000-250000": "brown",  
#     "250000-350000": "blue", 
#     "350000-480000": "red",
#     "480000+":"black"
# }

# sns.countplot(
#     data=house_price_dataframe,
#     x='oneH_OCEAN',
#     hue='median_house_value_category',
#     palette=palette
# )


# In[61]:


house_price_dataframe.head()


# In[62]:


# how much the income for all households > its mean that every two familes how much did they made 
# # i understand that if the number of the households alot dous not mean the incom will be good


# In[63]:


house_price_dataframe.groupby(['households','age_category'],as_index=False)['median_income'].sum()


# ##### Regarding duration median_house_value_category, What is the household in age category?
# 
# 
# Question: How does the number of households within different house value categories
#     vary across housing age categories, and do older 
#     or newer homes tend to be associated with more households?
# 
#     How does the number of households vary across different 
#     housing age categories and median house value ranges?
# 
# 

# In[64]:


# plt.figure(figsize=(15,8))

# palette = {
#     "14000-100000": "yellow",
#     "100000-250000": "brown",  
#     "250000-350000": "blue", 
#     "350000-480000": "red",
#     "480000+":"black"
# }

# sns.barplot(
#     data=house_price_dataframe,
#     x='age_category', 
#     y='households',
#     hue='median_house_value_category',
#      palette=palette
# )



# What is the relationship between median income and housing age categories within different ranges of median house value?
# 

# In[65]:


# plt.figure(figsize=(15,8))

# palette = {
#     "14000-100000": "yellow",
#     "100000-250000": "brown",  
#     "250000-350000": "blue", 
#     "350000-480000": "red",
#     "480000+":"black"
# }

# sns.barplot(
#     data=house_price_dataframe,
#     x='age_category', 
#     y='median_income',
#     hue='median_house_value_category',
#      palette=palette
# )


# In[66]:


# px.pie(data_frame=house_price_dataframe, names='age_category', values='median_income', color='median_house_value_category')


# the age of houses to their median value can reveal if older or newer houses tend to be more valuable in the dataset.

# hat is the distribution of median house values across different housing age categories?
# 
# Even here I understand that, even if the edge of the house more than 50 the price of the house doesn't go down 

# In[67]:


# import plotly.express as px

# fig = px.pie(data_frame=house_price_dataframe, 
#              names='total_rooms', 
#              values='median_house_value', 
#              title='Distribution of Median House Value by Total Rooms')
# fig.show()


# In[68]:


# import matplotlib.pyplot as plt
# import numpy as np

# # Sample data: Median House Value by Total Rooms
# # Note: This is arbitrary data for demonstration purposes.
# total_rooms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# median_house_value = np.array([200000, 180000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000])

# # Creating the figure and the axes
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plotting the data
# ax.scatter(total_rooms, median_house_value, color='blue', alpha=0.75)

# # Adding titles and labels
# ax.set_title('Distribution of Median House Value by Total Rooms', fontsize=14)
# ax.set_xlabel('Total Rooms', fontsize=12)
# ax.set_ylabel('Median House Value ($)', fontsize=12)

# # Adding a grid
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Show plot
# plt.tight_layout()
# plt.show()


# In[69]:


# plt.figure(figsize=(15, 8))
# sns.boxplot(data=house_price_dataframe, x='age_category', y='median_house_value')
# plt.title('Median House Value by Housing Age Category')
# plt.xlabel('Housing Age Category')
# plt.ylabel('Median House Value')
# plt.show()


# what is relationship between the age of the housing and the median income of the residents

# In[70]:


# plt.figure(figsize=(15, 8))
# sns.barplot(data=house_price_dataframe, x='age_category', y='median_income')
# plt.title('Median Income by Housing Age Category')
# plt.xlabel('Housing Age Category')
# plt.ylabel('Median Income')
# plt.show()


# The results could reveal if older homes are in more established, densely populated neighborhoods or if newer homes are built in areas of recent expansion.
# 

# In[71]:


# plt.figure(figsize=(15, 8))
# sns.scatterplot(data=house_price_dataframe, x='age_category', y='Avg_people_per_household')
# plt.title('Population Density by Housing Age Category')
# plt.xlabel('Housing Age Category')
# plt.ylabel('Population Density')
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ###### models 
# 

# In[72]:


import math

# Assuming 'house_price_dataframe' is already defined and preprocessed
# Define X (features) and y (target variable)
X = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1)
y = house_price_dataframe['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_reg_model = LinearRegression()

# Fit the model on the training data
linear_reg_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = linear_reg_model.predict(X_test)

# Calculate the model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

math.sqrt(mse)

math.sqrt(mse), r2


# In[ ]:


house_price_dataframe.head()


# In[92]:


from sklearn.tree import DecisionTreeRegressor

# Splitting the dataset into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor(max_depth=10)

# Fitting the model to the training data
decision_tree_regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred_dt = decision_tree_regressor.predict(X_test)

# Calculating the Mean Squared Error and R-squared value
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

math.sqrt(mse_dt)

math.sqrt(mse_dt), r2_dt
# ......

###### # feature_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
#                  'total_bedrooms', 'population', 'households', 'median_income',
#                  'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN',
#                  'oneH_OCEAN', 'bedroom_ratio', 'Avg_rooms_per_house',
#                  'Avg_people_per_household']

# # Creating the DataFrame with the correct structure
# arr = pd.DataFrame([[-122.24, 37.85, 52.0, 1467.0, 190.0, 496.0, 177.0, 7.2574,
#                     0, 0, 1, 0, 0, 0.129516, 8.288136, 2.802260]], columns=feature_names)

# # Predicting with the Decision Tree Regressor


# decision_tree_regressor.predict(arr)
##########


# In[91]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Create and evaluate a Random Forest Classifier using the same features
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_rf.fit(X_train, y_train)
# y_pred_rf = clf_rf.predict(X_test)

# print("\nResults for Random Forest Classifier:")
# print("Accuracy:", accuracy_score(y_test, y_pred_dt))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
# print("Classification Report:\n", classification_report(y_test, y_pred_dt))


# In[78]:


# Visualization 3: Feature Importance in Random Forest
feature_importance = clf_rf.feature_importances_
plt.figure(figsize=(10, 6))

sns.barplot(x=feature_importance, y=list(X.columns), palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()


# In[80]:


from sklearn.tree import DecisionTreeClassifier

# Visualize the Bias-Variance Tradeoff (example with Decision Tree)
depths = np.arange(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))
    
    # Visualization 4: Bias-Variance Tradeoff
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Training Accuracy', marker='o')

plt.plot(depths, test_scores, label='Testing Accuracy', marker='o')

plt.title('Bias-Variance Tradeoff for Decision Tree')
plt.xlabel('Max Depth of Tree')
plt.ylabel('الت')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Visualize the Bias-Variance Tradeoff (example with Decision Tree)
depths = np.arange(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))
    
    # Visualization 4: Bias-Variance Tradeoff
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Training Accuracy', marker='o')

plt.plot(depths, test_scores, label='Testing Accuracy', marker='o')

plt.title('Bias-Variance Tradeoff for Decision Tree')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

features = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1)  # This removes the target column from the features
target = house_price_dataframe['median_house_value']  # This is the target column


# Prepare the data for Random Forest
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_regressor.predict(X_test)

# Calculate MSE and R^2 for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

math.sqrt(mse_rf)

math.sqrt(mse_rf), r2_rf


# 

# In[ ]:


# Visualization 3: Feature Importance in Random Forest
feature_importance = clf_rf.feature_importances_
plt.figure(figsize=(10, 6))

sns.barplot(x=feature_importance, y=list(features.columns), palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()


# In[ ]:


from sklearn.svm import SVR

X = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1)  # This removes the target column from the features
y = house_price_dataframe['median_house_value']  # This is the target column


# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
svr = SVR(kernel='linear')

# Fit the model
svr.fit(X_train, y_train)

# Predict
y_pred_svr = svr.predict(X_test)

# Evaluate the model
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

math.sqrt(mse_svr)
math.sqrt(mse_svr), r2_svr


# In[ ]:


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import fetch_california_housing

# # Create the Gradient Boosting Regressor model
# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# # Train the model
# gbr.fit(X_train, y_train)

# # Make predictions using the testing set
# y_pred = gbr.predict(X_test)

# # The mean squared error
# print("Mean squared error: %.2f" % math.sqrt(mean_squared_error(y_test, y_pred)))

# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))



# In[ ]:


# math.sqrt( mean_squared_error(y_test, y_pred))


# In[ ]:


# from sklearn.model_selection import GridSearchCV

# features = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1)  # This removes the target column from the features
# target = house_price_dataframe['median_house_value']  # This is the target column

# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 150],  # Number of trees in the forest
#     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
# }

# # Initialize the GridSearchCV object
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            scoring='neg_mean_squared_error',  # Scoring metric
#                            cv=5,  # Number of folds in cross-validation
#                            verbose=2,  # Controls the verbosity: the higher, the more messages
#                            n_jobs=-1)  # Number of jobs to run in parallel

# # Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)

# # Print the best parameters
# print("Best parameters:", grid_search.best_params_)

# # Get the best estimator
# best_model = grid_search.best_estimator_

# # Use the best model for predictions
# predictions = best_model.predict(X_test)

# # Evaluate the best model
# mse = mean_squared_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# # math.sqrt(mse_svr)
# # math.sqrt(mse_svr),r2


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming 'house_price_dataframe' contains your dataset

# Define features and target variable
p = house_price_dataframe.drop(columns=['median_house_value',"median_house_value_category","age_category"])
y = house_price_dataframe['median_house_value_category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(p, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier (Gaussian Naive Bayes for continuous features)
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_g = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_g)
print("Accuracy:", accuracy)

# Generate a classification report
print("Classification Report:\n", classification_report(y_test, y_pred_g))

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_g)
print("Confusion Matrix:\n", conf_matrix)

# mse_svr = mean_squared_error(y_test, y_pred_g)
# r2_svr = r2_score(y_test, y_pred_g)

# math.sqrt(mse_svr)
# math.sqrt(mse_svr), r2_svr


# ######Support Vector Regression

# In[ ]:


# # we use svr not svc becuz we are tried to prodect a continuse valuse the price the price cant be yes or no or 1000 or 2000 
# # Percentage of error in forecasting house prices
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
import math 
from sklearn.metrics import mean_squared_error


# Assuming you have your dataset loaded into DataFrames named train_house_price_dataframe and test_house_price_dataframe
# Extracting features (X) and target variable (y)
x_train, y_train = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1), house_price_dataframe['median_house_value']
x_test, y_test = house_price_dataframe.drop(['median_house_value',"median_house_value_category","age_category"], axis=1), house_price_dataframe['median_house_value']


# In[ ]:


# Feature scaling - important for SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit the scalers on the training data
X_train_scaled = scaler_X.fit_transform(x_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # Reshape y_train to 2D array

# Initializing and training the SVR model
svr_model = SVR(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.
svr_model.fit(X_train_scaled, y_train_scaled.ravel())

# Transform the test data using the fitted scalers
X_test_scaled = scaler_X.transform(x_test)

# Ensure that y_test is a DataFrame or a Series before using reshape
y_test = pd.Series(y_test)

# Reshape y_test to 2D array
y_test_reshaped = y_test.values.reshape(-1, 1)

# Testing the model on the test set
y_pred_scaled = svr_model.predict(X_test_scaled)

# Reshape y_pred_scaled to 2D array
y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)

# Inverse transform to get the predictions in the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled_reshaped)

# Evaluating the model performance

# Evaluating the model performance using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_reshaped, y_pred)
print(f'Mean Absolute Error: {mae}')



# In[ ]:





# In[ ]:




