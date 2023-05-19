# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:00:04 2023

@author: TIYASHI DAS
"""
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\Tiya\Datasets\Stress.csv')
e_df=pd.read_csv('D:\Tiya\Datasets\Stress.csv')
df.head()
df.columns
df.info()
df.describe()
print(df['subreddit'].value_counts())
df['len']=[len(text) for text in df['text']]
df['label_value']=df['label'].map({0:'No Stress',1:"Stress"})
df['date']=[datetime.fromtimestamp(value) for value in df['social_timestamp']]
e_df['len']=[len(text) for text in df['text']]
e_df['label_value']=df['label'].map({0:'No Stress',1:"Stress"})
e_df['date']=[datetime.fromtimestamp(value) for value in df['social_timestamp']]
re_cols=['post_id','sentence_range','confidence','social_timestamp','date']
e_df.drop(re_cols,axis=1,inplace=True)
df.hist(column='len',by='label_value',bins=61)
plt.show()
plt.title("confidence Vs stress")
df.groupby('confidence')['label'].mean().plot(kind="line")
sns.catplot(data=df,x='label_value',y='len',col='subreddit',col_wrap=3,hue='label_value',sharex=False)
plt.show()
plt.title("subreddit Vs confidence")
e_df.groupby('subreddit')['label'].mean().plot(kind="bar")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

X=e_df['len'].values
y=e_df['label_value'].values
'''X1=np.array(df['subreddit'])

cv = CountVectorizer()
X = cv.fit_transform(X1)

y=np.array(df['label_value'])'''
X=X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X_train, y_train)
print("\n")
print("Random forest accuracy: ")
print(clf.score(X,y))
print("\n")
clf_test=clf.predict(X_test)
clf_train = clf.predict(X_train)
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_train)))
''''print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_test)))'''
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("\n")
print("Decision tree accuracy: ")
print(clf.score(X,y))
print("\n")
print(clf.feature_importances_)
clf_train = clf.predict(X_train)
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_train)))
'''print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_test)))'''
from sklearn.neighbors import KNeighborsClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
print("\n")
print("Knn accuracy: ")
print(knn.score(X,y))
print("\n")
knn_test = knn.predict(X_train)
knn_train = knn.predict(X_train)
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, knn_train)))
'''print("Accuracy_Score =", format(metrics.accuracy_score(y_train, knn_test)))'''

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(X_train, y_train)
print("\n")
print("Naive bayes accuracy: ")
print(clf.score(X,y))
print("\n")
clf_train = clf.predict(X_train)
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_train)))
'''print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_test)))'''
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predict = svc_model.predict(X_test)
print(svc_model.score(X,y))
clf_train = clf.predict(X_train)
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_train)))
'''print("Accuracy_Score =", format(metrics.accuracy_score(y_train, clf_test)))'''
