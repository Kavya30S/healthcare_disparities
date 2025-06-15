import pandas as pd
import datetime
import os

intervention = {'timestamp': datetime.datetime.now(), 'action': 'Staff training recommended'}
audit_log = pd.DataFrame([intervention])

file_path = '../results/audit_log.csv'

try:
    df = pd.read_csv(file_path)
    header = False
except (FileNotFoundError, pd.errors.EmptyDataError):
    header = True

audit_log.to_csv(file_path, mode='a', header=header, index=False)
print("Intervention logged to ../results/audit_log.csv")