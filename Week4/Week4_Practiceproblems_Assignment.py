#!/usr/bin/env python
# coding: utf-8

# 1. Create a line plot of ZN and INDUS in the housing data.
# 
# 	a. For ZN, use a solid green line. For INDUS, use a blue dashed line.
# 	b. Change the figure size to a width of 12 and height of 8.
# 	c. Change the style sheet to something you find https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html.
# 
# 2. Create a bar chart using col1 and col2 of dummy data.
# 
# 	a. Give the plot a large title of your choosing.
# 	b. Move the legend to the lower-left corner.
# 	c. Do the same thing but with horizontal bars.
# 	d. Move the legend to the upper-right corner.
# 
# 3. Create a histogram with pandas for using MEDV in the housing data.
# 
# 	a. Set the bins to 20
# 
# 4. Create a scatter plot of two heatmap entries that appear to have a very positive correlation.
# 
# 5. Now, create a scatter plot of two heatmap entries that appear to have negative correlation.

# In[1]:


#setting up the packages

from IPython.display import HTML

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#changing the style to Classic 
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')

# Increase default figure and font sizes for easier viewing.
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# In[2]:


#Creating the dummy data from lecture
df = pd.DataFrame(np.random.randn(10, 4), 
                  columns=['col1', 'col2', 'col3', 'col4'],
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])


# In[3]:


# Read in the Boston housing data.
housing_csv = 'data/boston_housing_data.csv'
housing = pd.read_csv(housing_csv)

# Read in the drinks data.
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'data/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)

# Read in the ufo data.
ufo = pd.read_csv('data/ufo.csv')
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo['Year'] = ufo.Time.dt.year


# 1. Create a line plot of ZN and INDUS in the housing data.
# 
# 	a. For ZN, use a solid green line. For INDUS, use a blue dashed line.
# 	b. Change the figure size to a width of 12 and height of 8.
# 	c. Change the style sheet to something you find https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html.

# In[4]:


housing.head()


# In[5]:


#plotting the columns ZN and INDUS as a line plot the style used is classic
housing[["ZN","INDUS"]].plot(figsize = (12,8),style = {"ZN":"g","INDUS":":b"})


# 2. Create a bar chart using col1 and col2 of dummy data.
# 
# 	a. Give the plot a large title of your choosing.
# 	b. Move the legend to the lower-left corner.
# 	c. Do the same thing but with horizontal bars.
# 	d. Move the legend to the upper-right corner.

# In[6]:


#plotting a bar chart using col1 and col2 of dummy data
ax = df[["col1","col2"]].plot(kind = "bar")

#providing title to the plot
ax.set_title("Col1 & Col2 bar chart",fontsize = 30,y=1)

#moving the legend to the lower left corner
ax.legend(loc=3)


# In[7]:


#creating a bar chart between Col1 and Col2 in using barh
ax = df[["col1","col2"]].plot(kind = "barh")

#providing the title
ax.set_title("Col1 & Col2 horizontal bar chart",fontsize = 30,y = 1)

#setting the legend in the upper right corner
ax.legend(loc=1)


# 3. Create a histogram with pandas for using MEDV in the housing data.
# 
# 	a. Set the bins to 20

# In[8]:


#histogram with 20 bins column = MEDV from housing data

housing.MEDV.plot(kind = "hist",bins = 20)


# 4. Create a scatter plot of two heatmap entries that appear to have a very positive correlation.

# In[9]:


#plotting the heatmap using seaborn library
housing_correlations = housing.corr();
sns.heatmap(housing_correlations);


# *From the above result we can observe that columns CRIM and TAX have positive correlation*

# In[24]:


#plotting a scatter plot between CRIM and TAX
housing.plot(x='TAX', y='CRIM', kind='scatter', 
           color='crimson', figsize=(15,7), s=250,alpha = 0.3);


# 5. Now, create a scatter plot of two heatmap entries that appear to have negative correlation.

# *From the heat map of housing data correlation we can observe that the columns "DIS" and "NOX" have negative correlation*

# In[13]:


#scatter plot between "DIS" and "NOX"
housing.plot(x='DIS', y='NOX', kind='scatter', 
           color='violet', figsize=(15,7), s=250,alpha = 0.6);

