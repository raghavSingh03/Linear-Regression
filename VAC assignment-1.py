#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with python
# 
#                                     -Raghvendra pratap singh
#                                     Computer science business system
#                                     3rd year
#                                     RA1911042020041

# ### Import the libraries

# In[1]:


import warnings
warnings.simplefilter('ignore')


# ### Import numpy and pandas

# In[2]:


import numpy as np
import pandas as pd


# ### Import data visualization libraries

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import the dataset

# In[4]:


DS = pd.read_csv(r"C:\data sets\house_price.csv", encoding="ANSI")
DS


# In[5]:


DS.head()


# In[6]:


DS.info()


# In[7]:


DS.columns


# In[8]:


DS= DS.drop(['date','street','statezip','country'],axis=1)
DS


# In[9]:


plt.rcParams['figure.figsize']=12,10
plt.show


# ### 4) Handling the Categorical data

# In[10]:


cat_cols=['city']
from sklearn.preprocessing import LabelEncoder
for each_col in cat_cols:
    Le=LabelEncoder()
    DS[each_col]=Le.fit_transform(DS[each_col])


# In[11]:


DS


# ### Spliting the dataset into training and testing

# In[12]:


DS.columns


# In[13]:


x=DS[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated', 'city']].values
y=DS[['price']].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size =0.8)


# In[14]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# ### Import and perform Linear Regression

# In[15]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[16]:


lr.fit(x_train,y_train)


# ### Predicting  the values

# In[17]:


y_pred= lr.predict(x_test)
y_pred.round(2)


# In[18]:


check = pd.DataFrame(x_test,columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated', 'city'])


# In[19]:


check['price_Actual']=y_test


# In[20]:


check['price_Predicted']=y_pred.round(2)
check


# In[21]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(y_test,y_pred)


# In[22]:


mean_absolute_error(y_test,y_pred)


# In[ ]:




