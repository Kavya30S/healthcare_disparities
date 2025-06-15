import pandas as pd

try:
    patients = pd.read_csv('../data/synthea/patients.csv')
    encounters = pd.read_csv('../data/synthea/encounters.csv')
    print(f"Loaded patients.csv with {len(patients)} records")
    print(f"Patients columns: {list(patients.columns)}")
    print(f"Loaded encounters.csv with {len(encounters)} records")
    print(f"Encounters columns: {list(encounters.columns)}")

    data = pd.merge(patients, encounters, left_on='Id', right_on='PATIENT', how='inner')
    print(f"Merged data has {len(data)} records")
    print(f"Merged columns: {list(data.columns)}")

    if 'Id_x' in data.columns:
        data = data.rename(columns={'Id_x': 'Id'})
    if 'Id_y' in data.columns:
        data = data.drop(columns=['Id_y'])
    if 'PATIENT' in data.columns:
        data = data.drop(columns=['PATIENT'])
    if 'Id' not in data.columns:
        raise ValueError("Id column missing after merge")

    columns = ['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY', 'ENCOUNTERCLASS']
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    data = data[columns]
    print(f"After selecting columns, data has {len(data)} records")

    print(f"Missing values per column:\n{data.isnull().sum()}")

    data['GENDER'] = data['GENDER'].fillna('Unknown')
    data['RACE'] = data['RACE'].fillna('Unknown')
    data['ETHNICITY'] = data['ETHNICITY'].fillna('Unknown')
    data = data.dropna(subset=['ENCOUNTERCLASS'])
    print(f"After handling missing values, data has {len(data)} records")

    data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1, 'Unknown': 2})
    race_mapping = {'white': 0, 'Unknown': -1}
    ethnicity_mapping = {'non-hispanic': 0, 'hispanic': 1, 'Unknown': -1}
    data['RACE'] = data['RACE'].map(race_mapping).fillna(-1)
    data['ETHNICITY'] = data['ETHNICITY'].map(ethnicity_mapping).fillna(-1)
    data['TREATMENT'] = data['ENCOUNTERCLASS'].apply(lambda x: 1 if x in ['inpatient', 'emergency'] else 0)

    data.to_csv('../data/preprocessed_data.csv', index=False)
    print("Data preprocessing complete. Saved to ../data/preprocessed_data.csv")
    print(f"Final dataset has {len(data)} records")
    print(f"TREATMENT distribution:\n{data['TREATMENT'].value_counts()}")

except Exception as e:
    print(f"Error in preprocessing: {str(e)}")