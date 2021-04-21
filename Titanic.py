# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

BASE_DIR = os.path.dirname('/kaggle/input/titanic/train.csv')

df_train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
df_train.head()
df_train.shape
df_train.tail()
#counting null values
len(df_train) - df_train.count()

df_test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
df_test.head()
df_test.shape
#counting null values
len(df_test) - df_test.count()

df_combined = df_train.append(df_test)
df_combined.head()
len(df_combined) - df_combined.count()

df_combined.drop(['Cabin'], axis=1, inplace=True)
df_combined.head()

df_combined[df_combined['Fare'].isnull()]
# Missing Fare is from Pclass '3', male and Embarked from 'S' (id = 1044)

df_combined.groupby(['Pclass','Sex', 'Embarked'])['Fare'].agg(['mean'])
# Pclass '3', male and Embarked from 'S' - Mean Fare = 13

df_combined.loc[df_combined['Fare'].isnull(), 'Fare'] = 13
df_combined[df_combined['PassengerId'] == 1044]

df_combined[df_combined['Embarked'].isnull()]
# Pclass '1', female, Fare = 80.0 (same ticket) (id = 62 & 830)

df_combined.groupby(['Pclass', 'Sex', 'Embarked'])['Fare'].agg(['mean'])
# Pclass '1', female, Fare close to 80.0, embarked from 'Q'

df_combined.groupby(['Pclass', 'Embarked'])['Fare'].agg(['mean'])
# Excluding sex, the same result - Pclass '1', Fare close to 80.0, embarked from 'Q'

df_combined.loc[df_combined['Embarked'].isnull(), 'Embarked'] = 'Q'
df_combined[df_combined['PassengerId'] == 62]
df_combined[df_combined['PassengerId'] == 830]

# Analysing survivor's ages
survivors = df_train.loc[df_train['Survived'] == 1]
survivors.shape

import matplotlib.pyplot as plt
intervals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(survivors['Age'], bins=intervals)

# Creating passengers title
df_combined['Title'] = df_combined['Name'].str.extract('([a-zA-Z ]+)\.', expand=False).str.strip()
df_combined.head()

df_combined[df_combined['Title'].isnull()]
# All records have titles - OK

df_combined[df_combined['Age'].isnull()]

# Analysing groups by Title, Pclass and Age
ages_analysis = df_combined[df_combined['Age'].notnull()]
ages_analysis.groupby(['Title', 'Pclass', 'Sex'])['Age'].agg(['mean'])

# Identifyng null ages by groups
null_ages = df_combined[df_combined['Age'].isnull()]
null_ages.groupby(['Title', 'Pclass', 'Sex'])['PassengerId'].agg(['count'])
# Mr, Pclass '3' - count = 136 records


# For all null age records
# Identifyng null ages by groups

null_ages = df_combined[df_combined['Age'].isnull()]
null_ages.groupby(['Title', 'Pclass', 'SibSp', 'Parch'])['PassengerId'].agg(['count'])

not_null_ages = df_combined[df_combined['Age'].notnull()]
group_all_ages = not_null_ages.groupby(['Title', 'Pclass', 'SibSp','Parch'])['Age'].agg(['mean'])
group_all_ages

all_ages_dict = group_all_ages.to_dict('dict')
all_ages_dict

df_combined['Mean_Age_Index'] = list(zip(df_combined['Title'], df_combined['Pclass'], df_combined['SibSp'], df_combined['Parch']))
df_combined.head()

null_ages.shape
# 263 null ages

no_age = (df_combined['Age'].isnull())
df_combined[no_age].shape

count = 0
for index, row in df_combined[no_age].iterrows():    
    try:      
        new_age = round(all_ages_dict['mean'][(row['Mean_Age_Index'])], 2)
        print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', new_age)             
        df_combined.loc[index, 'Age'] = new_age             
        count += 1
        
    except KeyError:
        if (row['Mean_Age_Index'][1] == 3):
            if (row['Mean_Age_Index'][0] == 'Mr') & (row['Mean_Age_Index'][2] > 5):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '14')            
                df_combined.loc[index, 'Age'] = 14

            elif (row['Mean_Age_Index'][0] == 'Mr') & (row['Mean_Age_Index'][3] > 6):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '40')            
                df_combined.loc[index, 'Age'] = 40

            elif (row['Mean_Age_Index'][0] == 'Master') & (row['Mean_Age_Index'][2] == 0) & (row['Mean_Age_Index'][3] == 0):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '14.5')            
                df_combined.loc[index, 'Age'] = 14.5
                
            elif (row['Mean_Age_Index'][0] == 'Miss') & (row['Mean_Age_Index'][2] > 5):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '23.65')           
                df_combined.loc[index, 'Age'] = 23.65
                
            elif (row['Mean_Age_Index'][0] == 'Mrs') & (row['Mean_Age_Index'][3] > 6):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '43')           
                df_combined.loc[index, 'Age'] = 43
                
            elif (row['Mean_Age_Index'][0] == 'Ms'):
                print('PassengerId: ', row['PassengerId'], '   Previous Age: ', row['Age'], '   Updated Age: ', '28')           
                df_combined.loc[index, 'Age'] = 28
        
        else:
            pass
            
        count += 1
            
print(count)

# Checking result
no_age = (df_combined['Age'].isnull())
df_combined[no_age].shape
# All records with defined age - OK

df_combined['Sex'].replace(['female'], 0, inplace = True)
df_combined['Sex'].replace(['male'],1, inplace = True)
df_combined.head()

################################################
# Machine Learning Models:

df_combined.shape

model_data = df_combined[df_combined['Survived'].notnull()]
model_data.head()
model_data.shape

model_data.corr()
# Strongest correlations with survival: Sex(-54,34%) and Pclass (-33,85%)

model_data.head()

Feature = model_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
Feature.head()

# Creating X
X = Feature
X[0:5]

# Creating y
y = model_data['Survived'].values
y[0:5]

# Create train and test dataframes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Calculate accuracy for different Ks
Ks = 30

mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
print("The best KNN accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# Found best k=7
k = 7
knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#Predicting
KNN_predictor = knn.predict(X_test)
KNN_predictor [0:5]

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
dec_tree

# Fit data
dec_tree.fit(X_train, y_train)

# Predictions
tree_predictor = dec_tree.predict(X_test)

print (tree_predictor [0:5])
print (y_test [0:5])

# Support Vector Machine - SVM - (linear, poly, rbf, sigmoid)
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

SVM_predictor = clf.predict(X_test)
SVM_predictor [0:5]

# SVM Evaluation
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

print("F1 Score: ", f1_score(y_test, SVM_predictor, average='weighted'))
print("Jaccard: ", jaccard_score(y_test, SVM_predictor))
# Best is linear

# Logistic Regression - (try all the solvers: newton-cg, lbfgs, liblinear, sag, saga)
from sklearn.linear_model import LogisticRegression


LR = LogisticRegression(C=0.01, solver='newton-cg')
LR.fit(X_train,y_train)

LR_predictor = LR.predict(X_test)
LR_predictor_prob = LR.predict_proba(X_test)

# Evaluation
from sklearn.metrics import classification_report

print (classification_report(y_test, LR_predictor))
# Best is newton-cg

# Final Evaluation
from sklearn.metrics import log_loss

# KNN
KNN_f1_score = f1_score(y_test, KNN_predictor, average='weighted')
KNN_jaccard = jaccard_score(y_test, KNN_predictor)

# Decision tree
tree_f1_score = f1_score(y_test, tree_predictor, average='weighted')
tree_jaccard = jaccard_score(y_test, tree_predictor)

# SVM
SVM_f1_score = f1_score(y_test, SVM_predictor, average='weighted')
SVM_jaccard = jaccard_score(y_test, SVM_predictor)

# Logistic Regression
LR_f1_score = f1_score(y_test, LR_predictor, average='weighted')
LR_jaccard = jaccard_score(y_test, LR_predictor)
LR_log_loss = log_loss(y_test, LR_predictor_prob)

# Report
report_values = {'Algorithm': ['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'],
                 'Jaccard': [KNN_jaccard, tree_jaccard, SVM_jaccard, LR_jaccard],
                 'F1-score': [KNN_f1_score, tree_f1_score, SVM_f1_score, LR_f1_score],
                 'LogLoss': ['NA', 'NA', 'NA', LR_log_loss]                 
                }

df_report = pd.DataFrame(report_values)

df_report
# Best is Decision Tree

# Train data
train_data = df_combined[df_combined['Survived'].notnull()]
X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
y_train = train_data['Survived'].values
print ('Train set:', X_train.shape,  y_train.shape)

# Test data
test_data = df_combined[df_combined['Survived'].isnull()]
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
print ('Test set:', X_test.shape)

# Decision Tree
dec_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
dec_tree

# Fit data
dec_tree.fit(X_train, y_train)

# Predictions
y_hat = dec_tree.predict(X_test)
y_hat

print ('Result:', test_data.shape,  y_hat.shape)

test_data.head()

df_y_hat = pd.DataFrame({'Survived':y_hat})
df_y_hat['Survived'] = df_y_hat['Survived'].astype(int)
df_y_hat.head()

df_id = test_data['PassengerId']
df_id

df_result = pd.concat([df_id, df_y_hat], axis=1)
df_result.reset_index(drop=True, inplace=True)
df_result

df_result.to_csv("titanic_result.csv", index = False)

os.remove("titanic_result.csv")



################################################
# Tests run after mass average age calculation #
################################################

# Checking result
no_age = (df_combined['Age'].isnull())
df_combined[no_age].shape
# 7 records remaining with no age

null_ages = df_combined[df_combined['Age'].isnull()]
null_ages.groupby(['Title', 'Pclass', 'SibSp', 'Parch'])['PassengerId'].agg(['count'])
# Master, Miss, Mrs, Ms, all Pclass = 3 

not_null_ages = df_combined[df_combined['Age'].notnull()]
group_all_ages = not_null_ages.groupby(['Title', 'Pclass', 'SibSp','Parch'])['Age'].agg(['mean'])
group_all_ages

#Master
Master_3_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Master') & (df_combined['Pclass'] == 3)
df_combined[Master_3_age].shape

group_Master_3_ages = df_combined[Master_3_age].groupby(['SibSp', 'Parch'])['Age'].agg(['mean'])
group_Master_3_ages

# Miss
Miss_3_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Miss') & (df_combined['Pclass'] == 3)
df_combined[Miss_3_age].shape

group_Miss_3_ages = df_combined[Miss_3_age].groupby(['SibSp', 'Parch'])['Age'].agg(['mean'])
group_Miss_3_ages

# Mrs
Mrs_3_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Mrs') & (df_combined['Pclass'] == 3)
df_combined[Mrs_3_age].shape

group_Mrs_3_ages = df_combined[Mrs_3_age].groupby(['SibSp', 'Parch'])['Age'].agg(['mean'])
group_Mrs_3_ages

# Ms
Ms_3_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Ms') & (df_combined['Pclass'] == 3)
df_combined[Ms_3_age].shape
# Only 1 Ms in Pclass 3

# Calculating for Pclass = 2
Ms_2_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Ms') & (df_combined['Pclass'] == 2)
df_combined[Ms_2_age].shape

group_Ms_2_ages = df_combined[Ms_2_age].groupby(['SibSp', 'Parch'])['Age'].agg(['mean'])
group_Ms_2_ages

##################################################


#################################################
# Tests previously run to calculate average age #
#################################################

Mr_3_age = (df_combined['Age'].notnull()) & (df_combined['Title'] == 'Mr') & (df_combined['Pclass'] == 3)
df_combined[Mr_3_age].shape
# 312 Mr Pclass '3' with notnull ages

# Grouping by Sib and Parch
group_Mr_3_ages = df_combined[Mr_3_age].groupby(['SibSp', 'Parch'])['Age'].agg(['mean'])
group_Mr_3_ages

Mr_3_no_age = (df_combined['Age'].isnull()) & (df_combined['Title'] == 'Mr') & (df_combined['Pclass'] == 3)
df_combined[Mr_3_no_age].shape
df_combined[Mr_3_no_age].head()

group_Mr_3_no_ages = df_combined[Mr_3_no_age].groupby(['SibSp', 'Parch'])['PassengerId'].agg(['count'])
group_Mr_3_no_ages

df_Mr_3_no_age = df_combined[Mr_3_no_age]
df_Mr_3_no_age.shape
# 136 records selected - OK

Mr_3_dict = group_Mr_3_ages.to_dict('dict')
Mr_3_dict

df_combined['SibPar'] = list(zip(df_combined['SibSp'], df_combined['Parch']))

count = 0
for index, row in df_combined[Mr_3_no_age].iterrows():    
    try:      
        new_age = round(Mr_3_dict['mean'][(row['SibPar'])], 2)
        print('PassengerId: ', row['PassengerId'],\
              '   Previous Age: ', row['Age'],\
              '   Updated Age: ', new_age)      
             
        df_combined.loc[index, 'Age'] = new_age             
        count += 1
        
    except KeyError:
        if row['SibSp'] > 5:
            print('PassengerId: ', row['PassengerId'],\
                  '   Previous Age: ', row['Age'],\
                  '   Updated Age: ', '14')
            
            df_combined.loc[index, 'Age'] = 14
            
        elif row['Parch'] > 6:
            print('PassengerId: ', row['PassengerId'],\
                  '   Previous Age: ', row['Age'],\
                  '   Updated Age: ', '40')
            
            df_combined.loc[index, 'Age'] = 40
            
        count += 1
            
print(count)

#Checking if all records were updated
Mr_3_no_age = (df_combined['Age'].isnull()) & (df_combined['Title'] == 'Mr') & (df_combined['Pclass'] == 3)
df_combined[Mr_3_no_age].shape
# OK

####################################################

TitleDict = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \
             "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \
             "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \
             "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}