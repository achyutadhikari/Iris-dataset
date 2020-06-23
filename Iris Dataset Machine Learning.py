#!/usr/bin/env python
# coding: utf-8

# ## Iris Dataset Machine learning Project

# In[1]:


import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data
# 
# Import the iris.csv data 

# In[2]:


iris_data = pd.read_csv('iris.csv')


# In[3]:


iris_data.head(10)


# ## Discovering the data 

# In[4]:


iris_data.shape


# In[5]:


iris_data.species.value_counts()


# In[6]:


iris_data.species.unique()


# In[7]:


iris_data.groupby('species').size()


# In[8]:


iris_data


# ## Cleaning the data 

# In[9]:


#Strip ' cm' and ' mm' from each data point, and convert them to floats
#dataset 1
iris_data['sepal_length'] = list(map(lambda x: x[:-2], iris_data['sepal_length'].values))
iris_data['sepal_width'] = list(map(lambda x: x[:-2], iris_data['sepal_width'].values))
iris_data['petal_length'] = list(map(lambda x: x[:-2], iris_data['petal_length'].values))
iris_data['petal_width'] = list(map(lambda x: x[:-2], iris_data['petal_width'].values))


# In[10]:


iris_data


# In[11]:


iris_data.dtypes


# In[12]:


# convert them to floats
iris_data.loc[:,'sepal_length'] = pd.to_numeric(iris_data.loc[:,'sepal_length'],errors='coerce')
iris_data.loc[:,'sepal_width'] = pd.to_numeric(iris_data.loc[:,'sepal_width'],errors='coerce')
iris_data.loc[:,'petal_length'] = pd.to_numeric(iris_data.loc[:,'petal_length'],errors='coerce')
iris_data.loc[:,'petal_width'] = pd.to_numeric(iris_data.loc[:,'petal_width'],errors='coerce')


# The dataset sepal width is mistakenly 10 times larger than normal which is impossible so we reduced it by 10 times. 

# In[13]:


iris_data['sepal_width'] = iris_data['sepal_width'].div(10)


# In[14]:


iris_data.dtypes


# ## Statistical properties of data 

# In[15]:


iris_data.min()


# In[16]:


iris_data.max()


# In[17]:


iris_data.mean()


# In[18]:


iris_data.median()


# In[19]:


iris_data.std()


# ## Summary statistics
# It summarize the central tendency, dispersion and dataset distribution 

# In[20]:


summary = iris_data.describe().T


# In[21]:


summary.head()


# From the summary, we can analyse that there is huge range in the size of Sepal Length and Petal Length also verified by their higher standard deviation among other. Further, we will use exploratory analysis if size is related to the iris species. 

# ## Exploratory Data Analysis (EDA)

# ## Boxplot 
# 
# It visually compares distributions of sepal length, sepal width, petal length, petal width based on numerical data through their quartiles.

# In[22]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='species', y = 'sepal_length', palette="husl", data=iris_data)
plt.subplot(2,2,2)
sns.boxplot(x='species', y = 'sepal_width', palette="husl", data=iris_data)

plt.subplot(2,2,3)
sns.boxplot(x='species', y = 'petal_length',palette="husl",  data=iris_data)
plt.subplot(2,2,4)
sns.boxplot(x='species', y = 'petal_width', palette="husl", data=iris_data)


# ## Pairplot
# 
# Relationships between variables across multiple dimensions

# In[23]:


sns.pairplot(iris_data, hue="species",palette="husl", diag_kind="kde", markers=["o", "s", "D"])
plt.show()


# In[24]:


sns.pairplot(iris_data,hue="species", kind='reg',  palette="husl")
plt.show()


# In[25]:


sns.set(style="whitegrid", palette="husl", rc={'figure.figsize':(11.7,8.27)})

# "Melt" the dataset
iris2 = pd.melt(iris_data, "species", var_name="measurement")

# Draw a categorical scatterplot
sns.swarmplot(x="measurement", y="value", hue="species",palette="husl", data=iris2)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[26]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species', y = 'sepal_length', palette="husl",data=iris_data)
plt.subplot(2,2,2)
sns.violinplot(x='species', y = 'sepal_width', palette="husl", data=iris_data)

plt.subplot(2,2,3)
sns.violinplot(x='species', y = 'petal_length', palette="husl", data=iris_data)
plt.subplot(2,2,4)
sns.violinplot(x='species', y = 'petal_width',palette="husl", data=iris_data)


# ## Classification problem 
# 
# We use classification algorithm to build a model. 
# 
# Attributes: These are property or also known as features are used to determine its classification. Here, attributes are sepal length, sepal width, petal length and petal width. 
# 
# Targets or Outputs are 3 flower species

# In[28]:


# importing alll the necessary packages to use the various classification algorithms
import sklearn
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm
from sklearn.model_selection import train_test_split # to split the dataset for training and testing 
from sklearn.neighbors import KNeighborsClassifier # KNN classifier
from sklearn import svm # for suport vector machine algorithm
from sklearn import metrics # for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier # for using DTA


# In[29]:


iris_data.shape


# While training any algorithm, the number of features and their correlation plays an important role. If many of the features are highly correlated, then training an algorithm with all these features will reduce an accuracy. Feature selection should be done carefully. Existing dataset has less features although there is some correlation. 

# In[30]:


plt.figure(figsize=(8,4))
sns.heatmap(iris_data.corr(), annot=True) # draws heatmap with input as correlation matrix calculated by iris.corr() 
plt.show()


# From heatmap we can observe that sepal length and width are not correlated but petla length and width are coorrelated. We will use all the features for training the algorithm and check the accuracy. 
# 
# We will use petal and sepal feature to check the accuracy in the algorithm .
# 
# So, we will split dataset into training and testing dataset. Testing dataset is generally smaller than training since more training datasets the model will be better. 

# ## Splitting the data into training and testing dataset

# In[31]:


train, test = train_test_split(iris_data, test_size = 0.3) # dataset is split into 70% training and 30% testing
print(train.shape)
print(test.shape)


# In[32]:


train


# In[33]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_y = test.species


# In[34]:


train_X.head()


# In[35]:


test_X.head()


# In[36]:


train_y.head()


# In[37]:


test_y.head()


# ## Support Vector Machine SVM

# In[38]:


model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_y))


# ## Logistic Regression

# In[39]:


model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of Logistic Regression is: ', metrics.accuracy_score(prediction, test_y))


# ## Decision Tree

# In[40]:


model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of Decision Tree is: ', metrics.accuracy_score(prediction, test_y))


# ## K-Nearest Neighbors

# In[41]:


model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the data into class
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of KNN is: ', metrics.accuracy_score(prediction, test_y))


# ## Let's check the accuracy for various values of n for K-Nearest nerighbours

# In[42]:


a_index = list(range(1,11))
a = pd.Series()
for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, test_y)))
plt.plot(a_index, a)
x = [1,2,3,4,5,6,7,8,9,10]
plt.xticks(x)


# Above graph shows accuracy of KNN models using different values of n

# ## All the features of iris models are used, lets use petal and sepal 
# 
# ## training and testing data for petals and sepals

# In[43]:


petal = iris_data[['petal_length','petal_width','species']]
sepal = iris_data[['sepal_length','sepal_width','species']]


# ## Iris Petal

# In[44]:


train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) 

train_x_p = train_p[['petal_length','petal_width']]
train_y_p = train_p.species

test_x_p = test_p[['petal_length','petal_width']]
test_y_p = test_p.species


# ## Iris Sepal

# In[45]:


train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals
train_x_s = train_s[['sepal_length','sepal_width']]
train_y_s = train_s.species

test_x_s = test_s[['sepal_length','sepal_width']]
test_y_s = test_s.species


# ## SVM algorithm

# In[46]:


model=svm.SVC()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model=svm.SVC()
model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the SVM using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# ## Logistic Regression

# In[47]:


model = LogisticRegression()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# ## Decision Tree

# In[48]:


model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# ## K-Nearest Neighbors

# In[49]:


model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))


# ## Analysis
# From the mathematical models we used we can confirm that using petal features gives more accuracy.
# Further it was validated by the heatmap high correlation between petal length and width than that of sepal length and width. 
