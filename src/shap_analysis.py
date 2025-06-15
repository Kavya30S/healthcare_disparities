import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from shap import links  # Import for logit link function

# Load data and model
data = pd.read_csv('../data/preprocessed_data.csv')
model = joblib.load('../models/model.pkl')

# Prepare data
X = data[['GENDER', 'RACE', 'ETHNICITY']]
y = data['TREATMENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Summarize background data for efficiency
background = shap.sample(X_train, 100).values

# Create explainer with logit link
explainer = shap.LinearExplainer(model, background, link=shap.links.logit)

# Compute SHAP values
shap_values = explainer.shap_values(X_test.values)

# Generate and save summary plot
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
plt.savefig('../results/shap_plot.png', bbox_inches='tight')
plt.close()
print("SHAP plot saved to ../results/shap_plot.png")