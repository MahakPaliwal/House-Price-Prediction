#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[80]:


df=pd.read_csv("C:/Users/91720/Downloads/archive (1)/MagicBricks.csv")
df


# In[81]:


df.corr()


# In[82]:


df.shape


# In[83]:


df.isnull().sum()


# In[84]:


df.drop(['Furnishing','Locality','Status','Transaction','Type'],axis=1,inplace=True)
df.shape


# In[85]:


df['Price'].describe()


# In[86]:


df.dropna(how='any',inplace=True)


# In[87]:


df.isnull().sum()


# In[88]:


df.info()


# In[89]:


X=df[['Area','BHK','Bathroom','Parking','Price','Per_Sqft']]
y=df['Price']


# In[90]:


y


# In[91]:


X


# In[92]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.35, random_state=40)


# In[93]:


print("X_train.shape :",X_train.shape)
print("X_test.shape :",X_test.shape)
print("y_train.shape :",y_train.shape)
print("y_test.shape :",y_test.shape)


# In[94]:


lr=LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)


# In[95]:


y_predict=lr.predict(X_test)
y_predict.shape


# In[96]:


g=plt.plot((y_test-y_predict),marker='o',linestyle='')


# In[97]:


lr=LinearRegression(fit_intercept=False)
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
y_predict.shape


# In[98]:


g=plt.plot((y_test-y_predict),marker='o',linestyle='')


# In[99]:


plt.scatter(y_test,y_predict)


# In[100]:


y_predict[0]


# In[101]:


y_test


# In[102]:


y_predict[1]


# In[103]:


score=metrics.r2_score(y_test,y_predict)
print(score)


# In[ ]:





# In[ ]:




