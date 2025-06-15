import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

data = pd.read_csv('../data/preprocessed_data.csv')
print(f"Loaded {len(data)} records")

X = data[['GENDER', 'RACE', 'ETHNICITY']]
y = data['TREATMENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = LogisticRegression(random_state=42)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

joblib.dump(model, '../models/model.pkl')
print("Model saved to ../models/model.pkl")