#!/usr/bin/env python
# coding: utf-8

# **NUMPY**
# 
# **1.Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.**

# In[1]:


import numpy as np

#creating two custom numpy arrays A and B
A = np.arange(5)
B = np.arange(6,11)
print(A)
print(B)


# In[2]:


#creating a new numpy array by stacking A and B vertically
C = np.vstack([A,B])
C


# In[3]:


#creating a new numpy array by stacking A and B horizontally
D = np.hstack([A,B])
D


# **2. Find common elements between A and B. [Hint : Intersection of two sets]**

# In[4]:


# finding common elements using intersect1d
np.intersect1d(A,B)


# **3. Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]**

# In[16]:


#extracting all numbers from A between 1 and 4 

#using np.where to find list of indices where A>1
idx1 = np.where(A >1)
print(idx1)

#using np.where to find list of indices where A<4
idx2 = np.where(A<4)
print(idx2)

#using np.intersect1d to find list of common indices i.e where A is between 1 and 4
idx_common = np.intersect1d(idx1,idx2)

print("the numbers from A which are between 1 and 4 are :",A[idx_common])


# **4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
#     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#     iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])**

# In[6]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url,delimiter =',',dtype = 'float',usecols = [0,1,2,3])


# In[20]:


print(iris_2d.shape)
#m is the number of rows in iris_2d
#n is the number of columns in iris_2d

m = iris_2d.shape[0]
n = iris_2d.shape[1]


# In[17]:


#the first five elements in the dataset to know the element structure
iris_2d[:5,:]


# In[39]:


#finding the row indices where petallength (3rd column) > 1.5
idx1 = np.zeros([m,1])
idx1 = np.where(iris_2d[:,3] >1.5)

#finding the row indices where sepallength (1st column) < 5.0
idx2 = np.zeros([m,1])
idx2 = np.where(iris_2d[:,0] <5.0)

#finding the common indices
idx_common = np.intersect1d(idx1,idx2)
print("the row index/indices where that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0 is/are:",idx_common)

print("the row/s where that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0 is/are:",iris_2d[idx_common])


# **PANDAS**
# 
# **1. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).**
# 
# ```
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# ```

# In[161]:


import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')


# In[162]:


df.head(10)


# In[163]:


df.shape


# In[164]:


df.info


# In[179]:


#creating new dataframe with columns Manufacurer to Type from original dataframe
cars_new = df.loc[:,"Manufacturer":"Type"]

#fitering the rows with 20 step size
cars_new[0:93:20]


# **2. Replace missing values in Min.Price and Max.Price columns with their respective mean.**
# 
# ```
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# ```

# In[180]:


#creating a new data frame cars from data frame
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
cars = df.loc[:,:]
column1 = df.columns[3]
column2 = df.columns[5]
cars2 = cars.rename(columns = {column1:'Min_price',column2:'Max_price'})


# In[181]:


#finding the mean for min_price and max_price
cars2.head()
#print("the mean for Max_price is ",df.Max_price.mean())


# In[182]:


m1= cars2.Min_price.mean()
print(m1)


# In[183]:


m2 = cars2.Max_price.mean()
print(m2)


# In[184]:


cars2['Min_price'] = cars2['Min_price'].replace(np.nan,m1)
cars2['Max_price'] = cars2['Max_price'].replace(np.nan,m2)
#cars2['Max_price'].replace('Nan',m2,inplace = True)

cars2.head(10)


# **3. How to get the rows of a dataframe with row sum > 100?**
# 
# ```
# df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
# ```

# In[145]:


#creating a dataframe df
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))


# In[146]:


#knowing the shape of the dataframe
df.shape


# In[147]:


#printing the first 5 rows in the dataframe
df.head()


# In[153]:


#lets create a new dataframe with sum of columns of original dataframe
df_new = df.agg("sum",axis ="columns")

df_new


# In[154]:


#finding the rows with sum>100

df_new[(df_new[:])>100]


# In[155]:


#finding the rows with sum>100 in the original dataframe
df[df.agg("sum",axis ="columns")>100]

