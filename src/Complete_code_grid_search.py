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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm
import numpy as np
import pandas as pd


#Reading the input file
#all_train = pd.read_csv('modeltrain.csv')

#Using zscore to eliminate the outliers
from scipy import stats
z = np.abs(stats.zscore(all_train))
all_train = all_train[(z < 3).all(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['Survived'],axis=1), 
                                                    all_train['Survived'], test_size=0.30, 
                                                    random_state=101)
# Constructing some pipelines
pipe_lr = Pipeline([('scl', StandardScaler()),
			('clf', LogisticRegression(random_state=42))])


pipe_rf = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestClassifier(random_state=42))])


pipe_svm = Pipeline([('scl', StandardScaler()),
			('clf', svm.SVC(random_state=42))])


pipe_xgb = Pipeline([('scl', StandardScaler()), 
            ('clf', xgb.XGBClassifier(random_state = 42))])
			
# Set grid search params for each model

param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1, 0.5]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
		'clf__C': param_range_fl,
		'clf__solver': ['liblinear']}] 

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
		'clf__min_samples_leaf': param_range,
		'clf__max_depth': param_range,
		'clf__min_samples_split': param_range[1:]}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
		'clf__C': param_range}]
            
grid_params_xgb = [{
    'clf__max_depth': param_range,
    'clf__n_estimators': [10, 100, 500],
}]


# Construct grid searches for each model
jobs = -1

gs_lr = GridSearchCV(estimator=pipe_lr,
			param_grid=grid_params_lr,
			scoring='accuracy',
			cv=10) 
			

gs_rf = GridSearchCV(estimator=pipe_rf,
			param_grid=grid_params_rf,
			scoring='accuracy',
			cv=10, 
			n_jobs=jobs)



gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=10,
			n_jobs=jobs)

                         
gs_xgb = GridSearchCV(estimator = pipe_xgb, param_grid = grid_params_xgb, cv=10, n_jobs=jobs, scoring='accuracy')


# List of pipelines for ease of iteration
grids = [gs_lr, gs_rf, gs_svm, gs_xgb]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Logistic Regression', 1: 'Random Forest', 2: 'Support Vector Machine', 3: 'XGBoost'}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
	print('\nEstimator: %s' % grid_dict[idx])	
	# Fit grid search	
	gs.fit(X_train, y_train)
	# Best params
	print('Best params: %s' % gs.best_params_)
	# Best training data accuracy
	print('Best training accuracy: %.3f' % gs.best_score_)
	# Predict on test data with best params
	y_pred = gs.predict(X_test)
	# Test data accuracy of model with best params
	print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
	# Track best (highest test accuracy) model
	if accuracy_score(y_test, y_pred) > best_acc:
		best_acc = accuracy_score(y_test, y_pred)
		best_gs = gs
		best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

# Save best grid search pipeline to file
dump_file = 'best_gs_pipeline.pkl'
joblib.dump(best_gs, dump_file, compress=1)
print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))

