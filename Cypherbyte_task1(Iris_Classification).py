#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


df = pd.read_csv("C:/Users/asus/Downloads/Telegram Desktop/Iris Flower - Iris.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


sns.pairplot(df.drop(["Id"],axis=1),hue='Species')


# # Slitting Data

# In[8]:


#split dataset in features and target variable
x=df.drop('Species',axis=1) 
y=df['Species']
x.head()


# In[9]:


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


# In[10]:


# Standardize features (optional but recommended)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Model developement and Prediction

# In[11]:


# instantiate the model (using the default parameters)
model = LogisticRegression(random_state=16)

# fit the model with data
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)


# # Model evolution using confusion matrix

# In[12]:


# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# # Classification report

# In[13]:


target_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[14]:


sns.heatmap(cnf_matrix,annot=True,fmt='d',cmap='Reds', xticklabels=model.classes_, yticklabels=model.classes_ )
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




