# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 07:10:42 2017

@author: David
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

test='file:///C:/Users/David/Downloads/test(1).csv'
train='file:///C:/Users/David/Downloads/train(1).csv'
test=pd.read_csv(test)
train=pd.read_csv(train)
sns.set_style('whitegrid')

print(train.describe())
print(test.describe())
print(train.head())
print(test.head())


f,ax = plt.subplots(3,4,figsize=(20,18))
sns.countplot('Pclass',data=train,ax=ax[0,0])
ax[0,0].set_title('Total passengers in class1,class2,class3')
sns.countplot('Sex',data=train,ax=ax[0,1])
ax[0,1].set_title('Gender size')
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[0,2])
ax[0,2].set_title('Sibbling,Spouse that Survived')#from this we see this category of people died'

sns.countplot('Parch',hue='Survived',data=train,ax=ax[0,3])
ax[0,3].set_title('Parent and Children that Survived')

sns.distplot(train[train['Survived']==0]['Age'].dropna(),color='r',kde=False,ax=ax[1,0],bins=5)#We can see that Ages between
sns.distplot(train[train['Survived']==1]['Age'].dropna(),color='b',kde=False,ax=ax[1,0],bins=5)
ax[1,0].set_title('Age group that Survived')

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,1])#here we see class 1 had more survival rates
ax[1,1].set_title('Passenger Class that survived')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,2])
ax[1,2].set_title('Survival rates based on Gender')

sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,3])
ax[1,3].set_title('Parents that came with children that survived')

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2,0])
ax[2,0].set_title('People from which cities had highest survival rate')

co=sns.FacetGrid(col='Embarked',data=train)
co.map(sns.pointplot,'Pclass','Survived','Sex',palette='viridis')
co.add_legend()

#finding missing data
f,ax = plt.subplots(1,2,figsize=(15,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull() ,yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])

#cleaning Age that are missing
x=train['Age'].isnull()
ds=train['Name'][x]
print(ds)
def clean(cols):
    Age=cols[0]
    PClass=cols[1]
    
    if pd.isnull(Age):
        if PClass==1:
            return 34
        elif PClass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age']=train[['Age','Pclass']].apply(clean, axis=1)
test['Age'] = test[['Age','Pclass']].apply(clean,axis=1)
f,ax = plt.subplots(1,2,figsize=(15,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
train['Age'].to_csv('ad.csv')
#Fillling Missing Values
train['Cabin'].fillna('No Cabin',inplace=True)

#cleaning title
#combine dataset 1st for easier Feature Engineering
train['IsTrain'] = 1
test['IsTrain'] = 0
df = pd.concat([train,test])

df['Title']=df['Name'].str.split(', ').str[1].str.split('.').str[0]
print(df['Title'].value_counts())
df['Title'].replace('Mme','Mrs',inplace=True)
df['Title'].replace(['Ms','Mlle'],'Mrs',inplace=True)
df['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)    
print(df['Title'].value_counts())
df.drop('Name',inplace=True,axis=1)
df.head()
#df.to_csv('new.csv')

f,ax = plt.subplots(1,2,figsize=(15,8))

#child = 0, Young Adult = 1, Adult = 2, Old = 3, Veteran = 4
df['AgeGroup'] = df['Age']
df.loc[df['AgeGroup']<=19, 'AgeGroup'] = 0
df.loc[(df['AgeGroup']>19) & (df['AgeGroup']<=30), 'AgeGroup'] = 1
df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=45), 'AgeGroup'] = 2
df.loc[(df['AgeGroup']>45) & (df['AgeGroup']<=63), 'AgeGroup'] = 3
df.loc[df['AgeGroup']>63, 'AgeGroup'] = 4
sns.countplot(x='AgeGroup',hue='Survived',data=df[df['IsTrain']==1],palette='husl',ax=ax[0])
df.drop('Age',axis=1,inplace=True)


df['FareGroup'] = df['Fare']
df.loc[(df['FareGroup']<=50),'FareGroup'] = 0
df.loc[(df['FareGroup']>50) & (df['FareGroup']<=100),'FareGroup'] = 1
df.loc[(df['FareGroup']>100) & (df['FareGroup']<=200),'FareGroup'] = 2
df.loc[(df['FareGroup']>200) & (df['FareGroup']<=300),'FareGroup'] = 3
df.loc[df['FareGroup']>300,'FareGroup'] = 4
df['FareGroup'].fillna(0,inplace=True)
sns.countplot(x='FareGroup',hue='Survived',data=df[df['IsTrain']==1],palette='husl',ax=ax[1])
df.drop('Fare',inplace=True,axis=1)

#No of people in each deck(A-G & T) that Survived
f,ax = plt.subplots(1,2,figsize=(15,8))
df['deck']=df['Cabin']
df.loc[(df['deck']=='No Cabin'),'deck']='N/A'
df['deck']=df['deck'].str.split(' ').str[0]
df['deck']=df['deck'].str.rstrip('0123456789')
sns.countplot('deck',data=df[df['IsTrain']==1],hue='Survived',ax=ax[0])
ax[0].set_title('No of people in each deck(A-G & T) that Survived')
df.drop('Cabin',inplace=True,axis=1)
df.drop('Ticket',inplace=True,axis=1)

#0=not alone, 1= is alone
df['is_alone']=0
df.loc[(df['Parch']==0) & (df['SibSp']==0),'is_alone'] =1
sns.countplot('is_alone',data=df[df['IsTrain']==1],hue='Survived',ax=ax[1])
ax[1].set_title('0=Not Alone, 1=Alone')
df.drop('Parch',axis=1,inplace=True)
df.drop('SibSp',axis=1,inplace=True)

#Converting Categorical data into dummy values
df=pd.get_dummies(df)

#Splitting the data for X,Y
dataset=df[df['IsTrain']==1]
test_data=df[df['IsTrain']==0]
test_id = test_data['PassengerId']
dataset.drop(['IsTrain','PassengerId'],axis=1,inplace=True)
test_data.drop(['IsTrain','PassengerId','Survived'],axis=1,inplace=True)
X = dataset.drop(['Survived'],axis=1)
Y = dataset['Survived'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=44)
#Prediciting using logistic regression
kfold = KFold(n_splits=4, random_state=15)
rfc = RandomForestClassifier()
lgr = LogisticRegression()
print('Logistic Regression: ',cross_val_score(lgr, X_train, y_train, cv=kfold).mean()*100)
lgr.fit(X,Y)
predict=lgr.predict(test_data)
submission=pd.DataFrame({'PassengerId':test_id,'Survived':predict})
submission.to_csv('Submission22.csv')
