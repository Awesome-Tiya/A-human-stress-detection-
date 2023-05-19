# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:39:11 2023

@author: TIYASHI DAS
"""

import numpy as np
from datetime import datetime
import joypy
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\Tiya\Datasets\Stress-Lysis.csv')
e_df=df.copy(deep = True) #pd.read_csv('D:\Tiya\Datasets\Stress-Lysis.csv')
df.head()
df.columns
df.info()
df.describe()
print(e_df.isnull().sum())
df['label']=df['Stress Level'].map({0:'No Stress',1:"Mild Stress",2:"Stress"})
df['label_value']=df['label'].map({'No Stress':0,"Mild Stress":1,"Stress":2})
plt.show()
plt.title("Temperature Vs label")
df.groupby('Temperature')['Stress Level'].mean().plot(kind="line")
plt.show()
p = sns.heatmap(e_df.corr(), annot=True,cmap ='RdYlGn')
plt.show()
plt.title("Humidity Vs label")
df.groupby('Humidity')['Stress Level'].mean().plot(kind="line")

df_r = df[['Humidity', 'label']].groupby('label').apply(lambda x: x.mean())
df_r.sort_values('Humidity', inplace=True)
df_r.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=df_r.index, ymin=0, ymax=df_r.Humidity, color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=df_r.index, y=df_r.Humidity, s=75, color='firebrick', alpha=0.7)

ax.set_title('Lollipop Chart for Humidity and Stress Level', fontdict={'size':22})
ax.set_ylabel('Humidity')
ax.set_xticks(df_r.index)
ax.set_xticklabels(df_r.label.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(0, 30)

for row in df_r.itertuples():
    ax.text(row.Index, row.Humidity+.5, s=round(row.Humidity, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

plt.show()

plt.show()
plt.title("Step count Vs label")
df.groupby('Step count')['Stress Level'].mean().plot(kind="line")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

'''X=df['Step count'].values
y=e_df['Stress Level'].values
X=X.reshape(-1, 1)'''
y = df.label_value
y
X = e_df.drop('Stress Level', axis=1)
y = df['label_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=43)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X,y))
print("\n")
clf.predict(X_test)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X,y))
print("\n")
print(clf.feature_importances_)

from sklearn.neighbors import KNeighborsClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
print("\n")
print("Knn accuracy: ")
print(knn.score(X,y))
print("\n")

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X_train, y_train)
print("\n")
print("Naive bayes accuracy: ")
print(clf.score(X,y))
print("\n")

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X1=df['Step count'].values
y=e_df['Stress Level'].values
X1=X1.reshape(-1, 1)
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=43)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X1_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X1,y))
print("\n")
clf.predict(X1_test)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X1_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X1,y))
print("\n")
print(clf.feature_importances_)

from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier()
knn.fit(X1_train, y_train)
print("\n")
print("Knn accuracy: ")
print(knn.score(X1,y))
print("\n")

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X1_train, y_train)
print("\n")
print("Naive bayes accuracy: ")
print(clf.score(X1,y))
print("\n")

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X1_train, y_train)
svc_predict = svc_model.predict(X1_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X1,y)) 

X2=df['Humidity'].values
y=e_df['Stress Level'].values
X2=X2.reshape(-1, 1)
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.9, random_state=43)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X2_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X2,y))
print("\n")
clf.predict(X2_test)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X2_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X2,y))
print("\n")
print(clf.feature_importances_)

from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier()
knn.fit(X2_train, y_train)
print("\n")
print("Knn accuracy: ")
print(knn.score(X2,y))
print("\n")

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X2_train, y_train)
print("\n")
print("Naive bayes accuracy: ")
print(clf.score(X2,y))
print("\n")

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X2_train, y_train)
svc_predict = svc_model.predict(X2_test)
print("\n")
print("support vector accuracy: ")
print(svc_model.score(X2,y)) 

X3=df['Temperature'].values
y=e_df['Stress Level'].values
X3=X3.reshape(-1, 1)
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.9, random_state=43)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X3_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X3,y))
print("\n")
clf.predict(X3_test)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X3_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X3,y))
print("\n")
print(clf.feature_importances_)

from sklearn.neighbors import KNeighborsClassifier 
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size = 0.9, random_state=42)
knn=KNeighborsClassifier()
knn.fit(X3_train, y_train)
print("\n")
print("Knn accuracy: ")
print(knn.score(X3,y))
print("\n")

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X3_train, y_train)
print("\n")
print("Naive bayes accuracy: ")
print(clf.score(X3,y))
print("\n")

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X3_train, y_train)
svc_predict = svc_model.predict(X3_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X3,y)) 