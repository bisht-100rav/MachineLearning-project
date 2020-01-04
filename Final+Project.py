#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('E:/Python/training_internshala/Final Project/data sets/train.csv')


# In[4]:


train.head(5)


# In[7]:


train.tail(5)


# In[8]:


train.corr()


# In[ ]:





# In[14]:


train = pd.get_dummies(train)


# In[15]:


train.isnull().sum()


# In[16]:


X = train.drop(['subscribed_yes','subscribed_no'],axis = 1)


# In[17]:


y = train['subscribed_yes']


# In[18]:


from sklearn.cross_validation import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

LOGISTIC REGRESSION
# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


logreg = LogisticRegression()


# In[22]:


logreg.fit(X_train,y_train)


# In[23]:


predictions = logreg.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


classification_report(y_test,predictions)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test,predictions)


# Linear Regression

# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


linreg = LinearRegression(n_jobs = 1)


# In[30]:


linreg.fit(X_train,y_train)


# In[31]:


predictions_linear = linreg.predict(X_test)


# In[32]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[33]:


linreg.score(X_train,y_train)


# Decision Tree

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


clf = DecisionTreeClassifier()


# In[36]:


clf.fit(X_train,y_train)


# In[37]:


clf.score(X_train,y_train)


# In[38]:


clf.score(X_test,y_test)


# In[39]:


predictions_clf = clf.predict(X_test)


# In[40]:


predictions_clf


# In[41]:


from sklearn.metrics import accuracy_score


# In[42]:


accuracy_score(y_test,predictions_clf)


# Predicting from the test data sets

# In[43]:


test = pd.read_csv('E:/Python/training_internshala/Final Project/data sets/test.csv')


# In[44]:


test.columns


# In[45]:


test_dummy = pd.get_dummies(test)


# In[46]:


test_dummy.shape


# In[52]:


predicitons_clf = clf.predict(test_dummy)


# In[53]:


test['subscribed_clf'] = predicitons_clf


# In[54]:


predictions_log = logreg.predict(test_dummy)


# In[55]:


test['subscribed_log'] = predictions_log


# In[ ]:





# In[56]:


del(test['poutcome'])


# In[57]:


test.columns


# In[58]:


#test.to_csv('E:/Python/training_internshala/Final Project/data sets/final_solution.csv')


# In[59]:


print("decision tree", predicitons_clf.sum())
print("logistic ", predictions_log.sum())


# In[60]:


test['subscribed_clf'] = test['subscribed_clf'].map({0 : "no", 1: "yes"})


# In[61]:


test['subscribed_log'] = test['subscribed_log'].map({0 : "no", 1: "yes"})


# In[62]:


test.to_csv('E:/Python/training_internshala/Final Project/data sets/solution_final.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




