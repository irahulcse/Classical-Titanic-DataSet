#%%[markdown][markdown]
# # Titanic(Classical Problem) For Machine Learning Engineers
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')


df_train=pd.read_csv('/home/rahul/Desktop/Link to rahul_environment/Kernels/Titanic_Disaster/train.csv')
df_train.head()
test=pd.read_csv('/home/rahul/Desktop/Link to rahul_environment/Kernels/Titanic_Disaster/test.csv')
test.head()
type(df_train)
type(test)

#%%[markdown]
# # Scatter plot for the given dataset in the age and fare or you can say that
g=sns.FacetGrid(df_train,hue="Survived",col="Pclass",margin_titles=True,palette={1:"seagreen",0:"gray"})
g=g.map(plt.scatter,"Fare","Age",edgecolor="w").add_legend();

# # Box plot for graphically depicting groups of numerical data through their quartiles.
#%%[markdown]
ax=sns.boxplot(x="Pclass",y="Age",data=df_train)
ax=sns.stripplot(x="Pclass",y="Age",data=df_train,jitter=True,edgecolor="grey")
plt.show()
#%%[markdown]
# # Plotting all the subplots in the diagram
f,ax=plt.subplots(1,2,figsize=(20,10))
df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title("Survived==1")
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


#Now plotting the above predictions on the pie chart according to people who had survived or not in the situations
#%%[markdown]
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


#%%[markdown]
#  # finding the stats according to the sex of the person'
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived or Dead')
plt.show()


#making the multivariable plots
# we will be going to look at the scattterplots of all pairs of the attributes.
#%%[markdown]
# # Plotting the scattter plots 
pd.plotting.scatter_matrix(df_train,figsize=(10,10))
plt.figure()

#%%[markdown]
# # Creating the violin plots of the given graph which can be used for creating of the plots
f,sns.violinplot(data=df_train,x="Sex",y="Age")

#%%[markdown]
# # Making the violin plot  with Pclass and Age vs Survived
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age",hue="Survived",data=df_train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age",hue="Survived",data=df_train,split=True,ax=ax[1])
ax[1].set_title("Sex and Age vs Survived")
ax[1].set_yticks(range(0,110,10))
plt.show()

#%%[markdown]
# # Using the seaborn pairplot 
sns.pairplot(df_train,hue='Sex')
#%%[markdown]
#using the kde plot for creating the histogram in the diagonal in the pairplot by kde
sns.FacetGrid(df_train,hue="Survived",size=5).map(sns.kdeplot,"Fare").add_legend()
plt.show()
#%%[markdown]
# # Creating a joint plot between the fare and age column of the df_train which can help the find of the
sns.jointplot(x='Fare',y='Age',data=df_train)

sns.jointplot(x='Fare',y='Age',data=df_train,kind='reg')

#%%[markdown]
# # Creating a swarm plot for the age and pclass 
sns.swarmplot(x='Pclass',y='Age',data=df_train)

#creating a heatmap for the plot so that we can find out to use it
plt.figure(figsize=(7,4))
sns.heatmap(df_train.corr(),annot=True,cmap='cubehelix_r')
plt.show()


#%%[markdown]
# # Creating a bar plot for the given point in the form of the graph
df_train['Pclass'].value_counts().plot(kind="bar")

#creating a factor plot for the for the plcass which includes the survived and sex of the persons
#%%[markdown]
# # Going to create the factor plot
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)
plt.show()

#%%[markdown]
# # Going to create the subplots which can be used for training the datasets.
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass2')
sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass3')
plt.show()



## Now we will be going to proceed towards the data preprocessing 
#Some of the steps going to be done during the data processing are as follows:
# removing Target column (id)
# Sampling (without replacement)
    # Dealing with Imbalanced Data
    # Introducing missing values and treating them (replacing by average values)
    # Noise filtering
    # Data discretization
    # Normalization and standardization
    # PCA analysis
    # Feature selection (filter, embedded, wrapper)

#explorer dataset
#%%[markdown]
print(df_train.shape)
print(df_train.size)

#how many na elements in every column is equal to
#%%[markdown]
df_train.isnull().sum()

print(df_train.shape)
#%%[markdown]
df_train.info()
#%%[markdown]
df_train['Age'].unique()
#%%[markdown]
df_train['Pclass'].value_counts()
#%%[markdown]
#for finding the first 5 data from the datasets
df_train.head()
#%%[markdown]
#for finding the last 5 datas from the dateset
df_train.tail()
#%%[markdown]
# # For popping the 5 elements randomly from the datasets
df_train.sample()
#%%[markdown]
# # Very important
# # For getting the statistical summary of the datasets
df_train.describe()
#%%[markdown]
# # To check out the no of null values in the datasets you cann find out using isnull().sum()
df_train.isnull().sum()
#%%[markdown]
df_train.groupby('Pclass').count()

# Getting the columns of the dataset for the index columns
#%%[markdown]
df_train.columns

#getting the columns through the where clause
#%%[markdown]
df_train.where(df_train['Age']==30)
#%%[markdown]
# # Performing some of the query on the dataset
df_train[df_train['Age']<7.2].head(2)
#%%[markdown]
# # Separating the data into dependent and independent variables
X=df_train.iloc[:,:-1].values
y=df_train.iloc[:,-1].values
#Now we will be going to do the Data Cleaning
#%%[markdown]
# # Data Cleaning

#     When dealing with real-world data, dirty data is the norm rather than the exception.
#     We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records.
#     We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.
#     The primary goal of data cleaning is to detect and remove errors and anomalies to increase the value of data in analytics and decision making.[8]

# 6-4-1 Transforming Features

# Data transformation is the process of converting data from one format or structure into another format or structure[wiki]

#     Age
#     Cabin
#     Fare
#     Name
#%%[markdown]
# # Data Transformation 
def simplify_age(df):
    df.Age=df.Age.fillna(-0.5)
    bins=(-1,0,5,12,18,25,35,60,120)
    group_names=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
    categories=pd.cut(df.Age,bins,labels=group_names)
    df.Age=categories
    return df

def simplify_cabins(df):
    df.Cabin=df.Cabin.fillna('N')
    df.Cabin=df.Cabin.apply(lambda x:x[0])
    return df

def simplify_fares(df):
    df.Fare=df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names=['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories=pd.cut(df.Fare,bins,labels=group_names)
    df.Fare=categories
    return df
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df=simplify_age(df)
    df=simplify_cabins(df)
    df=simplify_fares(df)
    df=format_name(df)
    df=drop_features(df)
    return df

df_train=transform_features(df_train)
test=transform_features(test)
df_train.head()
#%%[markdown]
# # Feature Encoding
#      In machine learning projects, one important part is feature engineering. It is very common to see categorical features in a dataset. However, our machine learning algorithm can only read numerical values. It is essential to encoding categorical features into numerical values[28]

#     Encode labels with value between 0 and n_classes-1
#     LabelEncoder can be used to normalize labels.
#     It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.

def encode_features(df_train,test):
    features=['Fare','Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined=pd.concat()