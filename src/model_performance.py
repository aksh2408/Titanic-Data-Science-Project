

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
import numpy as np

#Reading the input file
all_train = pd.read_csv('modeltrain.csv')

#Using zscore to eliminate the outliers
from scipy import stats
z = np.abs(stats.zscore(all_train))
all_train = all_train[(z < 3).all(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['Survived'],axis=1), 
                                                    all_train['Survived'], test_size=0.30, 
                                                    random_state=101)
#Implementing different types of models
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

#Predicting the accuracy rate
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
