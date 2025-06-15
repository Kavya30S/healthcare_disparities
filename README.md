AI for Detecting and Mitigating Healthcare Disparities
Problem
Healthcare disparities in urban India lead to unequal treatment and outcomes for marginalized groups, increasing mortality and costs.
Solution
This AI-driven platform detects disparities in treatment across demographics using supervised learning and fairness metrics. It visualizes results on a Streamlit dashboard, suggests interventions via generative AI, and integrates CDC WONDER API data for context.
Setup

Clone the repository: git clone https://github.com/Kavya30S/healthcare-disparities.git
Activate Conda environment: conda activate healthcare_disparities
Install dependencies: conda install pandas scikit-learn tensorflow streamlit shap requests matplotlib plotly; conda install -c conda-forge fairlearn; pip install transformers
Download Synthea dataset from Synthea Downloads and place in data/synthea/.
Run scripts in order: preprocess_data.py, bias_detection.py, fairness_analysis.py, interventions.py, shap_analysis.py, audit_trail.py, cdc_api.py, dashboard.py.

Features

Bias detection using Random Forest.
Fairness metrics with Fairlearn.
Generative AI (GPT-2) for intervention suggestions.
SHAP for explainable AI.
Streamlit dashboard with CDC WONDER API integration.
Simulated audit trail for accountability.

Deployment
Hosted on Streamlit Sharing.
Challenges

Limited storage (80GB) and no GPU required lightweight libraries and CPU-based models.
Simplified federated learning and blockchain for hackathon feasibility.

License
MIT License
