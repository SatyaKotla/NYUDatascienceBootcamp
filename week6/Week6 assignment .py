#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)


# In[2]:


data = pd.read_csv("train.csv")


# In[3]:


# Dimensions of training data
data.shape


# In[4]:


# Print first few rows of data
data.head()


# In[5]:


# drop id, timestamp and country columns
data = data.drop(columns=['id', 'timestamp','country'])


# In[6]:


# Explore columns
data.columns


# In[7]:


data.info()


# In[8]:


# replace NANs in hours_per_week and telecommute_days_per_week with median value of the columns respectively  
data.loc[data['hours_per_week'].isna(), 'hours_per_week'] = data['hours_per_week'].median()
data.loc[data['telecommute_days_per_week'].isna(), 'telecommute_days_per_week'] = data['telecommute_days_per_week'].median()


# In[9]:


#Handling null values in categorical columns
data = data.dropna()


# In[10]:


data.info()


# In[11]:


# joint plots for numeric variables

cols = ["job_years", "hours_per_week"]
for c in cols:
    sns.jointplot(x=c, y="salary", data=data, kind = 'reg', height = 5)
plt.show()


# In[12]:


cols = ["job_years", "hours_per_week"]
for c in cols:
    sns.distplot(data[c])
    plt.grid()
    plt.show()


# In[13]:


# distribution of target variable
sns.distplot(data['salary'])
plt.grid()
plt.title('Distribution of Target Variable in Data')
plt.show()
print('max:', np.max(data['salary']))
print('min:', np.min(data['salary']))


# In[14]:


# create another copy of dataset and append encoded features to it
data_train = data.copy()
data_train.head()


# In[15]:


# select categorical features
cat_cols = [c for c in data_train.columns if data_train[c].dtype == 'object' 
            and c not in ['is_manager', 'certifications']]
cat_data = data_train[cat_cols]
cat_cols


# In[16]:


#Encoding binary variables
binary_cols = ['is_manager', 'certifications']
for c in binary_cols:
    data_train[c] = data_train[c].replace(to_replace=['Yes'], value=1)
    data_train[c] = data_train[c].replace(to_replace=['No'], value=0)


# In[17]:


final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True)
final_data.shape


# In[18]:


final_data.columns


# In[19]:


final_data


# In[20]:


y = final_data['salary']
X = final_data.drop(columns=['salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("Training Set Dimensions:", X_train.shape)
print("Validation Set Dimensions:", X_test.shape)


# **Normalizing data**

# In[21]:


# select numerical features
num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
num_cols


# In[22]:


# Apply standard scaling on numeric data 
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])


# In[23]:


X_train


# In[24]:


reg=LinearRegression()
reg.fit(X_train, y_train)


# In[25]:


reg.coef_


# In[26]:


reg.intercept_


# In[27]:


mean_absolute_error(y_train,reg.predict(X_train))


# In[28]:


mean_squared_error(y_train,reg.predict(X_train))**0.5


# ## Practice
# 1. Preprocess Test data and get predictions
# 2. Compute Mean Abolute Error, Mean Square error for test data
# 3. Implement [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge) and [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) Regression and then compute the following metrics on test data

# In[29]:


X_test[num_cols] = scaler.transform(X_test[num_cols])
y_pred = reg.predict(X_test)


# In[30]:


#mean absolute error and mean square error
print(mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred)**0.5)


# In[31]:


X_test.describe()


# In[32]:


#Ridge regression
ridge = Ridge(alpha=1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test), label='true')
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred), label = 'pred')
plt.legend()


# In[33]:


ridge.coef_


# In[34]:


#lasso regression
lasso = Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test))
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred))


# In[35]:


lasso.coef_


# # Decision Trees and Random Forests

# In[36]:


# train Decision Tree regression model
decisiontree = DecisionTreeRegressor(max_depth = 10, min_samples_split = 5)
decisiontree.fit(X_train, y_train)

#evaluating train error
mean_absolute_error(y_train,decisiontree.predict(X_train))


# In[37]:


max_depth_list = [2,3,4,5,6,7,8,9,10,11,12,20]
train_error = []
test_error =[]

for md in max_depth_list:

    decisiontree = DecisionTreeRegressor(max_depth = md, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(max_depth_list,train_error,label = 'train error')
plt.plot(max_depth_list,test_error,label = 'test error')
plt.legend()


# In[38]:


# Fitting a Random Forest Regressor
randomf = RandomForestRegressor()
randomf.fit(X_train, y_train)
mean_absolute_error(y_train,randomf.predict(X_train))


# In[39]:


max_depth_list = [10,11,12,13,14,15,16,17,18,19,20]
train_error = []
test_error =[]
N_estimator=[20,30,40,50,60b,70,80,90,100]
for n in N_estimator:

    decisiontree = RandomForestRegressor(n_estimators=n, max_depth = 12, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(N_estimator,train_error,marker='o',label = 'train error')
plt.plot(N_estimator,test_error,marker='o',label = 'test error')
plt.legend()


# In[40]:


pd.DataFrame({'feature':X_train.columns, "importance":randomf.feature_importances_*100}).sort_values(by='importance', ascending=False)


# Practice Questions:
# 
# 1.Compute errors on test sets
# 
# 2.Play with different parameter of decision trees and random forests and see the impact on train and test error
# 
# 3.[OPTIONAL] implement cross validation and get best hyperparameters

# In[41]:


print('the train_error is :',train_error)
print('the test_error is :',test_error)


# In[47]:


#random forest regressor - change in N_estimator
max_depth_list = [10,11,12,13,14,15,16,17,18,19,20]
train_error1 = []
test_error1 =[]

N_estimator=[15,25,45,55,61,76,88,94,100]
for n in N_estimator:

    decisiontree = RandomForestRegressor(n_estimators=n, max_depth = 12, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error1.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error1.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(N_estimator,train_error,marker='o',label = 'train error1')
plt.plot(N_estimator,test_error,marker='o',label = 'test error1')
plt.legend()

max_depth_list = [10,11,12,13,14,15,16,17,18,19,20]
train_error = []
test_error =[]
N_estimator=[20,30,40,50,60,70,80,90,100]
for n in N_estimator:

    decisiontree = RandomForestRegressor(n_estimators=n, max_depth = 12, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(N_estimator,train_error,marker='o',label = 'train error')
plt.plot(N_estimator,test_error,marker='o',label = 'test error')
plt.legend()


# In[48]:


#decision tree regressor - change in N-estimator
max_depth_list = [2,3,4,5,6,7,8,9,10,11,12,20]
train_error = []
test_error =[]

for md in max_depth_list:

    decisiontree = DecisionTreeRegressor(max_depth = md, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(max_depth_list,train_error,label = 'train error')
plt.plot(max_depth_list,test_error,label = 'test error')
plt.legend()

max_depth_list = [2,3,5,7,9,11,12,14,17,18,19,20]
train_error1 = []
test_error1 =[]

for md in max_depth_list:

    decisiontree = DecisionTreeRegressor(max_depth = md, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error1.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error1.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(max_depth_list,train_error,label = 'train error1')
plt.plot(max_depth_list,test_error,label = 'test error1')
plt.legend()

