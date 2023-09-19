#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Wine Quality Prediction


# In[1]:


#import the package

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


# In[3]:


#reading the data
dataset = pd.read_csv('winequality-red.csv')
dataset .head()


# In[4]:


#shape of datasets
print("Shape of our datasets of Red-Wine:{s}".format(s = dataset.shape))
print("Column headers/names: {s}".format(s = list(dataset)))


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


dataset[ 'quality'].unique()


# In[8]:


dataset.quality.value_counts().sort_index()


# In[9]:


dataset['alcohol'].describe()


# In[10]:


dataset['sulphates'].describe()


# In[11]:


dataset['citric acid'].describe()


# In[12]:


dataset['fixed acidity'].describe()


# In[13]:


dataset['residual sugar'].describe()


# In[16]:


Ql = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Ql

print(IQR)


# In[17]:


#The data point where we have False that means these values are valid whereas True presence of an outlier.
print(dataset < (Ql - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))


# In[18]:


dataset_out = dataset[~((dataset < (Ql - 1.5 * IQR))|(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
dataset_out.shape


# In[19]:


dataset_out


# In[20]:


correlations = dataset_out.corr()['quality'].drop('quality')
print(correlations)


# In[21]:


sns.heatmap(dataset.corr())
plt.show()


# In[22]:


#impact of various factor on quality
correlations. sort_values(ascending=False)


# In[25]:


def get_features(correlation_threshold) :
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations


# In[27]:


# taking features with correlation more than @.@5 as input x and quality as target variable y
features = get_features(0.05)

print (features)

x = dataset_out[features]

y = dataset_out['quality']


# In[28]:


#to finding the no of outiers we have in our dataset with proprties
bx = sns.boxplot(x='quality', y='alcohol', data = dataset)
bx.set(xlabel='Quality ', ylabel='Alcohol ', title='Alcohol % in different samples')


# In[31]:


bx = sns.boxplot(x='quality', y='citric acid', data = dataset)
bx.set(xlabel='Quality ', ylabel='Citric Acid ', title='Citric Acid % in different samples')


# In[32]:


bx = sns.boxplot(x='quality', y='fixed acidity', data = dataset)
bx.set(xlabel='Quality ', ylabel='Fixed Acidity', title='Fixed Acidity % in different samples')


# In[33]:


x


# In[34]:


y


# In[35]:


x_train,x_test,y_train,y_test=train_test=train_test_split(x,y, test_size=0.30,random_state=3)


# In[36]:


y_test.shape


# In[37]:


# fitting Linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[38]:


#To retrieve the intercept
regressor.intercept_


# In[39]:


# this gives the coefficients of the 1@ features selected above.
regressor.coef_


# In[40]:


train_pred = regressor.predict(x_train)
train_pred


# In[41]:


test_pred = regressor.predict(x_test)
test_pred


# In[43]:


train_rmse=metrics.mean_squared_error(train_pred, y_train) ** 0.5
train_rmse


# In[45]:


test_rmse = metrics.mean_squared_error(test_pred, y_test) ** 0.5
test_rmse


# In[46]:


# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
predicted_data


# In[48]:


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, test_pred))
rmse = np.sqrt(metrics.mean_squared_error(y_test, test_pred))

print("Root Mean Squared Error: ",rmse)


# In[49]:


from sklearn.metrics import r2_score
r2_score(y_test, test_pred)


# In[50]:


coeffecients = pd.DataFrame(regressor.coef_, features)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[53]:


ax=plt.axes()
color1= 'green'
color2= 'blue'
ax.arrow(0,0,1,0.56, head_width=0.00,head_length=0, fc=color2,ec=color2)
ax.arrow(0,0,2,0.63,head_width=0.00,head_length=0.05,fc=color1,ec=color1, linestyle='--')
ax.set_ylim([0,0.8])
ax.set_xlim([0,4])
plt.grid()
plt.title('RMSE_Score')
plt.show()


# In[54]:


ax=plt.axes()
colori= 'green'
color2= 'blue'
ax.arrow(0,0,2,0.40,head_width=0.00,head_length=0, fc=color2,ec=color2)
ax.arrow(0,0,1,0.34,head_width=0.00,head_length=0.05,fc=color1,ec=color1, linestyle='--')
ax.set_ylim([0,0.6])
ax.set_xlim([0,3])
plt.grid()
plt.title('R2_Score')
plt.show()


# In[55]:


ax=plt.axes()
color1= 'green'
color2= 'blue'
ax.arrow(0,0,1,0.45, head_width=0.00,head_length=0, fc=color2,ec=color2)
ax.arrow(0,0,2,0.49,head_width=0.00,head_length=0.05,fc=color1,ec=color1, linestyle='--')
ax.set_ylim([0,0.6])
ax.set_xlim([0,3])
plt.grid()
plt.title('MAE')
plt.show()


# In[ ]:




