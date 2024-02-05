#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


# In[163]:


pip install mlxtend 


# In[2]:


df = pd.read_csv("California Housing.csv")


# In[4]:


df.shape


# In[3]:


df.isnull().sum()


# In[7]:


sb.boxplot(x = df.total_bedrooms, data = df)


# In[8]:


df.total_bedrooms.plot.box()


# In[3]:


df.total_bedrooms.mode()


# In[4]:


df.describe()


# In[4]:


df.total_bedrooms.plot.density()
plt.axvline(df.total_bedrooms.mean(), color = 'black')


# Since the Data for total bedrooms is righty skewed most of the data is situated around the mode of the feature. Thus, we will use the mode to impute the missing values in the total_bedroom feature.

# In[5]:


df.total_bedrooms = df.total_bedrooms.fillna(df.total_bedrooms.mode()[0])

# we write [0] after mode because the mode function returns in series datatype.


# In[4]:


df.isnull().sum()


# In[48]:


df.describe()


# In[7]:


cols = ['housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
cols


# In[9]:


for i in df.select_dtypes(include = 'number'):
    sb.boxplot(x = df[i])
    plt.show()


# From the above box plots we can see that none of the features have outliers on the lower end. House Median Age has no outliers.

# In[21]:


c = 0
for j in df.select_dtypes(include = 'number'):
    u = (df[j] > (df[j].quantile(0.75) + 1.5 * (df[j].quantile(0.75) - df[j].quantile(0.25)))).sum()
    l = (df[j] < (df[j].quantile(0.25) - 1.5 * (df[j].quantile(0.75) - df[j].quantile(0.25)))).sum()
    print(j,u+l)
    c = 0


# In[25]:


# density plots to check for normal 
for i in cols:
    sb.kdeplot(x = df[i])
    plt.show()


# features - total_rooms, total_bedrooms, population, households, median income are rightly skewed. 
# The target value median_house_value has a bimodel distribution

# In[8]:


for i in cols:
    sb.histplot(x = df[i])
    plt.show()


# In[6]:


df1 = pd.DataFrame([df.housing_median_age, df.total_rooms,df.total_bedrooms, df.population, df.households, df.median_income, df.median_house_value]).T
df1


# In[51]:


# using pairplots to check correlation  between the features
sb.pairplot(df1)


# In[9]:


pd.DataFrame([df.housing_median_age,df.total_rooms]).T


# We can see that there is a positive correlation between, total rooms and total bedrooms, total rooms and population, 
# total rooms and household, total bedroom and population, total bedroom and household, population and household.
# 
# The intution behind these relationship is very logical.As the number of bedrooms increase in a district the number of rooms increase. As the households increase requirement of bedrooms increase and thus total bedrooms increase. 
# 

# In[7]:


plt.figure(figsize = (20,10))
sb.scatterplot(x = df.latitude, y = df.longitude, hue = df.median_house_value)


# Comparing the above graph to the map of California we can see the houses situated on the sea side near ocean have a higher value

# Feautre Engineering

# In[60]:


df.ocean_proximity.value_counts()


# In[8]:


df.drop(df[df.ocean_proximity == "ISLAND"].index, inplace = True)
df.shape


# In[9]:


df.ocean_proximity.value_counts()


# In[78]:


df['ocean_proximity'].value_counts().plot.bar(color = 'darksalmon', width = 0.5)
plt.ylabel("Counts")


# In[10]:


dummies_df = pd.get_dummies(df.ocean_proximity, drop_first = True , dtype = int)
df2 = pd.concat([df, dummies_df], axis = 1)
df2


# In[12]:


num_df = df2.select_dtypes(include = "number")
num_df.drop(["longitude", "latitude"], axis = 1)
num_df


# In[13]:


plt.figure(figsize = (14,14))
sb.heatmap(num_df.corr(), annot = True)


# # FEATURE ENGINEERING

# In[14]:


# adding Bedroom Ratio in the table
df2.bedroom_ratio = df2.total_bedrooms/df2.total_rooms


# In[129]:


df2


# In[15]:


df2['bedroom_ratio'] = (df2['total_bedrooms']/df2['total_rooms'])
df2['rooms_per_household'] = (df2['total_rooms']/df2['households'])
df2


# In[16]:


num_df = df2.select_dtypes(include = "number")
num_df = num_df.drop(['longitude', 'latitude'], axis =1)
num_df


# In[151]:


plt.figure(figsize = (14,14))
sb.heatmap(num_df.corr(), annot = True)


# In[20]:


df_y = num_df['median_house_value']
df_x = num_df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
       'households', 'median_income', 'INLAND',
       'NEAR BAY', 'NEAR OCEAN', 'bedroom_ratio', 'rooms_per_household']]


# In[21]:


df_x
df_y


# In[17]:


#feature selection using exhaustive feature selection

from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector

# Define the estimator (model)
model = LinearRegression()

# Define the feature selector
efs = ExhaustiveFeatureSelector(model, min_features=4, max_features=6, scoring='r2')

# Perform feature selection
efs.fit(df_x, df_y)

# Get the selected feature subsets
best_features = efs.best_feature_names_

# Print the results
print('Best features:', best_features)
print('Best score:', efs.best_score_)


# # Splitting the data into train and test 

# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(df_x,df_y, random_state = 0, train_size = 0.75)


# In[18]:


x_train.shape, x_test.shape

#fitting the model
# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model = LinearRegression().fit(x_train,y_train)
r_sq = model.score(x_train, y_train)
r_sq


# In[24]:


y_pred = model.predict(x_train)


# In[25]:


error = y_train-y_pred
error


# In[22]:


error.mean()
# the mean of error terms i.e.E(ui) = 0 is a major assumption of OLS estimators and this is followed.


# In[210]:


mse = mean_squared_error(y_train, y_pred)
mse


# In[195]:


sm.qqplot(error,line='45',fit=True,dist=stats.norm)


# In[221]:


sb.kdeplot(error)
plt.axvline(error.mean(), color = 'black')
# the error terms almost follow a normal distribution which can be seen from the density plots as well as the qqplots ploted abovr


# In[23]:


r_sq = model.score(x_test, y_test)
r_sq


# In[24]:


y_test_pred = model.predict(x_test)
error_test = y_test - y_test_pred


# In[25]:


mse_test = sum(error_test*error_test)/len(error_test)
mse_test


# In[214]:


c = model.coef_
c


# In[26]:


#using train data on OLS from scippy stats
x_train1 = sm.add_constant(x_train) # adding the coefficient term
model1 = sm.OLS(y_train, x_train1).fit()

#view model summary
(model1.summary())


# In[28]:


residuals = model1.resid
residuals


# # interpret standard error of model in this project, interpret durbin watson and jarque-bera  

# Now when p-value is less than 0.05 we assume them to be significant (because it helps us reject the null hypothesis that b=0)
# In this model the significant variables are housing_median_age, population, households, median_income, Inland, Near ocean, bedroom_ratio, rooms_per_household.
# One of the dummy variables NEAR BAY is insignificant, but we cannot leave it out because as a rule of thumb we cannot leave out any dummy variables even if it is insignificant
# 

# In[235]:


# Creating new model with significant variables 
x_train2 = x_train[['housing_median_age', 'population', 'households', 'median_income', 'INLAND', 'NEAR OCEAN',"NEAR BAY", 'bedroom_ratio', 'rooms_per_household']]
x_test2 = x_test[['housing_median_age', 'population', 'households', 'median_income', 'INLAND', 'NEAR OCEAN',"NEAR BAY", 'bedroom_ratio', 'rooms_per_household']]


# In[234]:


x_train2 = sm.add_constant(x_train2) # adding the coefficient term
model2 = sm.OLS(y_train, x_train2).fit()

#view model summary
(model2.summary())

#There is no significant change in R2 by removing insignificant variables, But the F-stat for overall significance of model increases.
#This shows that the features total_rooms and total_bedrooms had no significant impact on the predicted median_house_value
#In place of these variables there were new engineeed features bedroom_ratio and rooms_per_household that have a significant impact on Y


# In[241]:


#Checking for Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
coll = x_train2.columns
len(coll)


# In[250]:


vif = pd.DataFrame()
vif['features'] = coll
vif['VIF'] = [VIF(x_train2,i) for i in (range(len(coll)))]
vif 
#population and households has a high VIF, which exceeds 10. 


# In[269]:


x_train3 = x_train2.drop('households', axis = 1)
x_train3
x_test3 = x_test2.drop('households', axis = 1)


# In[257]:


# 3rd Model after removing collinear variable household
model3 = LinearRegression().fit(x_train3,y_train)
model3.score(x_train3, y_train)


# In[281]:


# Checking the model without multicollinearity on Test Set
model3.score(x_test3, y_test)


# In[265]:


colll = x_train3.columns
len(colll)

# VIF score of an independent variable represents how well the variable is explained by other independent variables.
VIF determines the strength of the correlation between the independent variables. It is predicted by taking a variable and regressing it against every other variable.
R^2 value is determined to find out how well an independent variable is described by the other independent variables. A high value of R^2 means that the variable is highly correlated with the other variables. 
# In[266]:


vif2 = pd.DataFrame()
vif2['features'] = colll
vif2['VIF'] = [VIF(x_train3,i) for i in (range(len(colll)))]
vif2 

#After removing household which had a higher VIF of two(population, household), the VIF of all factors are under 10.


# In[283]:


# fitting a Ridge Regression Model
from sklearn.linear_model import Ridge
ridge = Ridge()
model4 = ridge.fit(x_train3,y_train)
model4.score(x_train3,y_train)


# In[282]:


# Checking ridge model for train dataset
model4.score(x_test3,y_test)


# In[ ]:


# testing for heteroscedasticity
# Decision Tree Model
# Random Forest


# In[31]:


from statsmodels.stats.diagnostic import het_white
white_test_results = het_white(residuals, x_train1)
print("White's Test Results:")
print(white_test_results)


# In[28]:


plt.acorr(error)


# In[ ]:




