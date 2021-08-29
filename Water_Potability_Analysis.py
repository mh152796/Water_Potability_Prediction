#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('D:/water_potability.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.nunique()


# In[9]:


df.isnull().sum()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.dtypes


# In[12]:


df.ph = df.ph.fillna(df.ph.mean())
df.Sulfate = df.Sulfate.fillna(df.Sulfate.mean())
df.Trihalomethanes = df.Trihalomethanes.fillna(df.Trihalomethanes.mean())


# In[13]:


df.isnull().sum()


# In[14]:


sns.heatmap(df.isnull())


# In[15]:


value = df.	Potability.value_counts()


# In[16]:


plt.bar(["Not Potable (0)","Potable (1)"],value)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[18]:


x = df.drop(['Potability'], axis = 'columns')


# In[19]:


x.head()


# In[20]:


y = df.Potability


# In[21]:


y.head()


# In[22]:


x.shape, y.shape


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)


# In[24]:


x_train


# In[25]:


y_train


# # Using Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[27]:


model = LogisticRegression()


# In[28]:


model.fit(x_train, y_train)


# In[29]:


LgPred = model.predict(x_test)


# In[30]:


lc_matrix = confusion_matrix(y_test, LgPred)
axes = sns.heatmap(lc_matrix, square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)


# In[ ]:





# In[31]:


Lgaccuracy = accuracy_score(y_test, LgPred)
Lgaccuracy


# In[32]:


print(classification_report(y_test,LgPred))


# In[33]:


model.predict_proba(x_test)


# # Using Decision Tree Classifier

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


model = DecisionTreeClassifier()


# In[36]:


model.fit(x_train, y_train)


# In[37]:


dPred = model.predict(x_test)


# In[38]:


dtc_matrix = confusion_matrix(y_test, dPred)
axes = sns.heatmap(dtc_matrix, square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)


# In[39]:


dAccuracy = accuracy_score(y_test, dPred)
dAccuracy


# In[40]:


print(classification_report(y_test,dPred))


# # Using Support Vector Machine

# In[41]:


from sklearn.svm import SVC


# In[42]:


model_svm = SVC()


# In[43]:


model_svm.fit(x_train, y_train)


# In[44]:


pred_svm = model_svm.predict(x_test)


# In[45]:


svm_matrix = confusion_matrix(y_test, pred_svm)
axes = sns.heatmap(svm_matrix , square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)


# In[46]:


svmAaccuracy = accuracy_score(y_test, pred_svm)
svmAaccuracy


# In[47]:


print(classification_report(y_test,pred_svm))


# # Using Random Forest Classifier

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


rNd_model = RandomForestClassifier(n_estimators = 450)


# In[50]:


rNd_model.fit(x_train, y_train)


# In[51]:


rndPred = rNd_model.predict(x_test)


# In[52]:


rfc_matrix = confusion_matrix(y_test, rndPred)
axes = sns.heatmap(rfc_matrix , square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)


# In[53]:


rndAaccuracy = accuracy_score(y_test, rndPred)
rndAaccuracy


# In[54]:


print(classification_report(y_test,pred_svm))


# In[55]:


models = pd.DataFrame({
    'Model':['Logistic Regression', 'Decision Tree','Support Vector Machine','Random Forest Classifier'],
    'Accuracy_score' :[Lgaccuracy, dAccuracy, svmAaccuracy, rndAaccuracy]
})
sns.barplot(x='Accuracy_score', y='Model', data=models)

models.sort_values(by='Accuracy_score', ascending=False)


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




