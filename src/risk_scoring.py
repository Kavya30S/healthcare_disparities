import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('../data/preprocessed_data.csv')

X = data[['GENDER', 'RACE', 'ETHNICITY']]
y = data['TREATMENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train)

joblib.dump(risk_model, '../models/risk_model.pkl')
print("Risk model saved to ../models/risk_model.pkl")