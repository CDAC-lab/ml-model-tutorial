import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

#Load the CSV file containing the dataset.
hr_dataset = pd.read_csv("hr.csv")

# Prepend column name prior to encoding
hr_dataset['salary'] = 'salary_' + hr_dataset['salary'].astype(str)
# one hot encoding
one_hot_salary = pd.get_dummies(hr_dataset['salary'])
#append as a new column
hr_dataset = hr_dataset.join(one_hot_salary)
# Prepend column name prior to encoding
hr_dataset['department'] = 'dept_' + hr_dataset['department'].astype(str)
# one hot encoding
one_hot_department = pd.get_dummies(hr_dataset['department'])
#append as a new column
hr_dataset = hr_dataset.join(one_hot_department)
#To avoid multicollinearity, we must drop one of the new columns created during one hot encoding
hr_dataset = hr_dataset.drop(columns=['salary', 'department', 'salary_low', 'dept_IT'])

train, test, validate = np.split(hr_dataset.sample(frac=1), [int(.6*len(hr_dataset)), int(.8*len(hr_dataset))])
X_train = train.drop(columns=['Resigned'])
print(X_train.columns)

y_train = train[['Resigned']]
X_test = test.drop(columns=['Resigned'])
y_test = test[['Resigned']]
X_validate = validate.drop(columns=['Resigned'])
y_validate = validate[['Resigned']]

# XGBoost
hrXGB = xgb.XGBClassifier()
hrXGB.fit(X_train,y_train)

#Saving the machine learning model to a file
joblib.dump(hrXGB, "model/rf_model.pkl")