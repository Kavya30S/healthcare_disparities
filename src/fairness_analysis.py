import pandas as pd
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate
import joblib

data = pd.read_csv('../data/preprocessed_data.csv')
model = joblib.load('../models/model.pkl')

X = data[['GENDER', 'RACE', 'ETHNICITY']]
y = data['TREATMENT']

sensitive_features_list = ['GENDER', 'RACE', 'ETHNICITY']
results = []

for sf in sensitive_features_list:
    sensitive_features = data[sf]
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test)
    mf = MetricFrame(metrics={'selection_rate': selection_rate},
                     y_true=y_test,
                     y_pred=y_pred,
                     sensitive_features=sensitive_test)
    mf_df = mf.by_group.reset_index()
    mf_df['sensitive_feature'] = sf
    results.append(mf_df)

final_results = pd.concat(results, ignore_index=True)
final_results.to_csv('../results/fairness_metrics.csv', index=False)
print("Fairness metrics saved to ../results/fairness_metrics.csv")