
import numpy as np # linear algebra
import pandas as pd # data processing

#To read the train dataset
train = pd.read_csv("train.csv", sep = ';')

#To read the test dataset
test = pd.read_csv("val.csv", sep = ';')

all = pd.concat([train, test], sort = False) #Combining both the dataset, Hence it helps in data cleanzing and data wrangling on both the datasets


#Computing Median to all the NULL values in the respective columns
all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())

all['Embarked'] = all['Embarked'].fillna('S')

#Combining 2 column into 1 to ease the process
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1

#As the values need to be categorical and "Sex" column is an important categorical variable, it should be mapped to numerical values (i.e from strings to numerical)
all["Sex"] = all["Sex"].replace("male", 0)
all["Sex"] = all["Sex"].replace("female", 1)


#Deleting or dropping out unwanted anamolies
all_1 = all.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket','Cabin'], axis = 1)

all_dummies = pd.get_dummies(all_1)

all_train = all_dummies[all_dummies['Survived'].notna()]

#Feature Importance
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
X=all_train.drop(["Survived"],axis=1)
y=all_train['Survived']
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #using inbuilt class feature_importances of tree based classifiers
#plotting graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

all_train.to_csv('modeltrain.csv') #Saving the cleanzed, filtered dataset for further usage

