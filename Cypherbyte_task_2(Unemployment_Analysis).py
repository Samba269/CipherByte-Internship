#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install plotly


# In[3]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as ps
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("C:/Users/asus/Downloads/Unemployment in India - Unemployment in India.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.describe()


# In[8]:


df.info()


# # Data Processing

# In[9]:


# Data Cleaning 
df.isnull().sum() #Checking for NULL values


# In[10]:


df.isna().sum()  # returns the count of NA values in each column .NA values can include both null values and other types of missing values


# In[11]:


df = df.dropna() # Drop rows with missing values


# In[12]:


df.isna().sum()


# # Data Reshaping

# In[13]:


df.tail(2)


# In[14]:


df.rename(columns={'Region': 'State'}, inplace=True) # Renaming the 'Region' Column to 'State' Column
df.tail(2)


# In[15]:


# Calculate average unemployment rate by state
average_unemployment_rate = df.groupby('State')['Estimated Unemployment Rate (%)'].mean()


# # State with highest unemployment rate

# In[16]:


# Find the state with the highest unemployment rate
state_with_highest_unemployment = average_unemployment_rate.idxmax()
highest_unemployment_rate = average_unemployment_rate.max()


# In[17]:


print("State with the highest unemployment rate:", state_with_highest_unemployment)
print("Highest unemployment rate:", highest_unemployment_rate)


# # State with lowest unemployment rate

# In[18]:


# Find the state with the lowest unemployment rate
state_with_lowest_unemployment = average_unemployment_rate.idxmin()
lowest_unemployment_rate = average_unemployment_rate.min()


# In[19]:


print("State with the lowest unemployment rate:", state_with_lowest_unemployment)
print("Lowest unemployment rate:", lowest_unemployment_rate)


# # Data Visualization

# In[21]:


# Line plot showing unemployment rate over time:
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)
plt.xticks(rotation=45)
plt.title('Unemployment Rate Over Time')
plt.show()


# In[22]:


# Bar plot displaying average unemployment rate by state:
state_avg_unemployment = df.groupby('State')['Estimated Unemployment Rate (%)'].mean().reset_index()
sns.barplot(x='State', y='Estimated Unemployment Rate (%)', data=state_avg_unemployment)
plt.xticks(rotation=90)
plt.title('Average Unemployment Rate by State')
plt.show()


# In[23]:


# Histogram of Unemployment Rate Distribution:
sns.histplot(df['Estimated Unemployment Rate (%)'], bins=20, kde=True)
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.title('Unemployment Rate Distribution')
plt.show()


# In[24]:


# Heatmap of Correlation Matrix:
correlation_matrix = df[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()     


# In[25]:


# Boxplot of Unemployment Rate by State:
sns.boxplot(x='State', y='Estimated Unemployment Rate (%)', data=df)
plt.xticks(rotation=90)
plt.title('Unemployment Rate by State')
plt.show()


# In[ ]:




