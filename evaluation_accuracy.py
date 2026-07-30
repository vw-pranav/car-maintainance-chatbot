import pandas as pd
from pathlib import Path

path = Path('data/evaluation.xlsx')
if not path.exists():
    raise FileNotFoundError(f'{path} not found')

df = pd.read_excel(path)
print('columns:', list(df.columns))

expected_col = None
actual_col = None
for c in df.columns:
    low = c.lower()
    if 'expected' in low and expected_col is None:
        expected_col = c
    if 'actual' in low and actual_col is None:
        actual_col = c

if expected_col is None or actual_col is None:
    raise ValueError('Could not find expected/actual column names')

print('expected_col:', expected_col)
print('actual_col:', actual_col)

exp = df[expected_col].astype(str).str.strip().str.lower()
act = df[actual_col].astype(str).str.strip().str.lower()
exact = exp == act
count = exact.sum()
accuracy = count / len(df) if len(df) else 0
print('rows:', len(df))
print('exact matches:', count)
print('accuracy:', accuracy)
print('\nSample rows:')
print(df[[expected_col, actual_col]].head(20).to_string(index=False))
