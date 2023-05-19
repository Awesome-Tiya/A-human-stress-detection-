# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:09:47 2023

@author: TIYASHI DAS
"""

import numpy as np
from datetime import datetime
import squarify
import joypy
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\Tiya\Datasets\SaYoPillow.csv')
e_df=df.copy(deep = True) #pd.read_csv('D:\Tiya\Datasets\Stress-Lysis.csv')
df.head()
df.columns
df.info()
df.describe()
print(e_df.isnull().sum())
df['label']=df['sl'].map({0:'No Stress',1:"Mild Stress",2:"Stress",3:"More Stress",4:"Severe Stress"})
plt.title("Respiration rate Vs Stress Level")
df.groupby('rr')['sl'].mean().plot(kind="line")
plt.show()
plt.title("Snoring rate Vs Stress Level")
df.groupby('sr')['sl'].mean().plot(kind="line")
plt.show()
plt.title("heart rate Vs Stress Level")
df.groupby('hr')['sl'].mean().plot(kind="line")
plt.show()
plt.title("limb Movement rate Vs Stress Level")
df.groupby('lm')['sl'].mean().plot(kind="line")
plt.show()
plt.figure(figsize=(13,10), dpi= 80)
sns.violinplot(x='sl', y='bo', data=df, scale='width', inner='quartile')
plt.title("blood oxygen level Vs Stress Level")
plt.show()
sns.catplot(x = "sl", y = "rem", data = df)
plt.title("Rapid Eye Movement Vs Stress Level")
plt.show()
sns.violinplot(x='sl', y='t', data=df, scale='width', inner='quartile')
plt.title("body temperature Vs Stress Level")
plt.show()
sns.catplot(x = "sl", y = "sr.1", data = df)
plt.title("numbers of hours of sleep Vs Stress Level")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
y = df.sl
y
X = e_df.drop('sl', axis=1)
y = df['sl']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.43, random_state=43)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X,y))
print("\n")
a=clf.predict(X_train)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X,y))
print("\n")
print(clf.feature_importances_)

'''from sklearn.neighbors import KNeighborsClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=43)
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
print("\n")'''

from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X1=df['rr'].values
y=e_df['sl'].values
X1=X1.reshape(-1, 1)
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=43)
 


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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X2=df['sr'].values
y=e_df['sl'].values
X2=X2.reshape(-1, 1)
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.25, random_state=43)


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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X3=df['lm'].values
y=e_df['sl'].values
X3=X3.reshape(-1, 1)
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.25, random_state=42)


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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X4=df['rem'].values
y=e_df['sl'].values
X4=X4.reshape(-1, 1)
X4_train, X4_test, y_train, y_test = train_test_split(X4, y, test_size=0.25, random_state=42)                                                     
 

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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X5=df['bo'].values
y=e_df['sl'].values
X5=X5.reshape(-1, 1)
X5_train, X5_test, y_train, y_test = train_test_split(X5, y, test_size=0.25, random_state=42) 


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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X6=df['sr.1'].values
y=e_df['sl'].values
X6=X6.reshape(-1, 1)
X6_train, X6_test, y_train, y_test = train_test_split(X6, y, test_size=0.25, random_state=42)   

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

'''from sklearn.neighbors import KNeighborsClassifier 
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X7=df['t'].values
y=e_df['sl'].values
X7=X1.reshape(-1, 1)
X7_train, X7_test, y_train, y_test = train_test_split(X7, y, test_size=0.25, random_state=42)  

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

'''from sklearn.neighbors import KNeighborsClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state=42)
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
print("\n")'''

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print("\n")
print("Support vector accuracy: ")
print(svc_model.score(X,y)) 

X8=df['hr'].values
y=e_df['sl'].values
X8=X8.reshape(-1, 1)
X8_train, X8_test, y_train, y_test = train_test_split(X8, y, test_size=0.25, random_state=42)                                              
plt.figure()
