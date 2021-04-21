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

df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
df.head()
df.shape
df.tail()
#counting null values
len(df) - df.count()

df_tst = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
df_tst.head()
#counting null values
len(df_tst) - df_tst.count()


df.groupby(['Pclass', 'Sex', 'Survived'])['Age'].agg(['mean', 'count'])

df_clean = df.drop(['Name','Ticket','Fare'], axis=1)
df_clean



df_clean.drop(['Cabin'], axis=1, inplace=True)
df_clean.head()

#counting null values
len(df_clean) - df_clean.count()

females = df_clean.loc[df_clean['Sex'] == 'female']
females.shape
# 314 females

males = df_clean.loc[df_clean['Sex'] == 'male']
males.shape
# 577 males

avg_age = df_clean['Age'].astype('float').mean(axis=0)
avg_age
# Average age = 29.7
df_clean['Age'].value_counts()
df_clean['Age'].value_counts().idxmax()

avg_fem_age = females['Age'].astype('float').mean(axis=0)
avg_fem_age
# Average female age = 27.9

avg_male_age = males['Age'].astype('float').mean(axis=0)
avg_male_age
# Average male age = 30.7




survivors = df_clean.loc[df_clean['Survived'] == 1]
survivors.head()
survivors.shape
# 342 survivors
len(survivors.loc[survivors['Sex'] == 'female'])
# 233 female survivors
len(survivors.loc[survivors['Age'] < 18])
len(survivors.loc[survivors['Age'] < 10])

avg_survivors_age = survivors['Age'].astype('float').mean(axis=0)
avg_survivors_age
# 28.3 Avg survivors age

dead = df_clean.loc[df_clean['Survived'] == 0]
dead.head()
dead.shape

children = df_clean.loc[df_clean['Age'] < 18]
children.head()
children.shape

from matplotlib import pyplot as plt
plt.hist(survivors['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Survivors by Age')



