from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    model = joblib.load('model/rf_model.pkl')
    df_request = pd.DataFrame(json, index=[0])
    # Prepend column name prior to encoding
    df_request['salary'] = 'salary_' + df_request['salary'].astype(str)

    # function for adding non-existing dummy columns
    def add_missing_dummy_columns(df, columns):
        missing_cols = set(columns) - set(df.columns)
        for col_name in missing_cols:
            df[col_name] = 0

    # one hot encoding
    salary_categories = ['salary_high', 'salary_low', 'salary_medium']
    one_hot_salary =  pd.get_dummies(df_request['salary'])
    add_missing_dummy_columns(one_hot_salary, salary_categories)

    # append as a new column
    hr_df = df_request.join(one_hot_salary)

    # Prepend column name prior to encoding
    hr_df['department'] = 'dept_' + hr_df['department'].astype(str)

    # one hot encoding
    departments = ['dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr',
       'dept_management', 'dept_marketing', 'dept_product_mng', 'dept_sales',
       'dept_support', 'dept_technical']

    one_hot_department =  pd.get_dummies(hr_df['department'])
    add_missing_dummy_columns(one_hot_department, departments)

    # append as a new column
    hr_df = hr_df.join(one_hot_department)

    hr_df = hr_df.drop(columns=['salary', 'department', 'salary_low', 'dept_IT'])

    # Re-order the model features required for the classifier
    feature_order = model.get_booster().feature_names
    df_predict = hr_df[feature_order]
    y_predict = model.predict(df_predict)

    if y_predict[0] == 1:
        result = {"Predicted Churn Status": "Yes"}
    else:
        result = {"Predicted Churn Status": "No"}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)