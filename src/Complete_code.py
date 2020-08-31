
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

#all_train.to_csv('modeltrain.csv') #Saving the cleanzed, filtered dataset


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib


#Reading the input file
#all_train = pd.read_csv('modeltrain.csv')

#Using zscore to eliminate the outliers
from scipy import stats
z = np.abs(stats.zscore(all_train))
all_train = all_train[(z < 3).all(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['Survived'],axis=1), 
                                                    all_train['Survived'], test_size=0.30, 
                                                    random_state=101)

model_dict = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=9),
    'LogisticRegression': LogisticRegression(max_iter=100000, random_state=0),
    'LinearSVC': LinearSVC(max_iter=100000, random_state=0),
    'SVC': SVC(gamma=0.1, C=0.3),
    'GaussianNB': GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=4, random_state=0),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50, max_depth=1, random_state=0),
    'MLPClassifier': MLPClassifier(random_state=0, solver='lbfgs', hidden_layer_sizes=[10, 10], alpha=0.0100),
    'XGBClassifier': XGBClassifier(random_state=0, learning_rate=0.01, n_estimators=50, subsample=0.8, max_depth=4)
}

data_list = list()
for name, model in model_dict.items():
    data_dict = dict()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    data_dict['model'] = name
    data_dict['train_score'] = train_score
    data_dict['test_score'] = test_score
    data_list.append(data_dict)
score_df = pd.DataFrame(data_list)
score_df

# Save best grid search pipeline to file
dump_file = 'data_model.pkl'
joblib.dump(model, dump_file, compress=1)
print('Model Dumped')

model = joblib.load('data_model.pkl')

model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

