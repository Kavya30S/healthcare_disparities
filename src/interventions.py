import pandas as pd
from transformers import pipeline

fairness_data = pd.read_csv('../results/fairness_metrics.csv')
gender_data = fairness_data[fairness_data['sensitive_feature'] == 'GENDER']
if not gender_data.empty:
    min_sr = gender_data['selection_rate'].min()
    max_sr = gender_data['selection_rate'].max()
    min_group = gender_data.loc[gender_data['selection_rate'].idxmin(), 'GENDER']
    max_group = gender_data.loc[gender_data['selection_rate'].idxmax(), 'GENDER']
    gender_labels = {0: 'Male', 1: 'Female', 2: 'Unknown'}
    min_group_label = gender_labels.get(min_group, str(min_group))
    max_group_label = gender_labels.get(max_group, str(max_group))
    disparity_description = f"Gender group {min_group_label} has a selection rate of {min_sr:.2f}, while gender group {max_group_label} has {max_sr:.2f}, a disparity of {max_sr - min_sr:.2f}."
else:
    disparity_description = "No gender disparity data available."

kb = pd.read_csv('../data/interventions_kb.csv')
disparity_type = "gender_disparity"
relevant_kb = kb[kb['disparity_type'] == disparity_type]['suggested_intervention'].values

generator = pipeline('text-generation', model='gpt2')
if len(relevant_kb) > 0:
    prompt = f"Based on the disparity: {disparity_description}, and considering these interventions: {', '.join(relevant_kb)}, suggest a tailored intervention."
else:
    prompt = f"Based on the disparity: {disparity_description}, suggest an intervention."

suggestion = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
with open('../results/interventions.txt', 'w') as f:
    f.write(suggestion)
print("Intervention suggestion saved to ../results/interventions.txt")