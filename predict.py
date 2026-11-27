import argparse
import joblib
import pandas as pd
import sklearn
from src.reference import predict_reference

p = argparse.ArgumentParser(description='KAIROS single Random Forest research inference. Load trusted artifacts only.')
p.add_argument('--bundle', default='kairos_datasets/KAIROS_REFERENCE.joblib')
p.add_argument('--input', required=True, help='Feature-ready CSV with Date and seven predictors')
p.add_argument('--output', default='predictions.csv')
a = p.parse_args()
b = joblib.load(a.bundle)
if b['sklearn_version'] != sklearn.__version__:
    raise RuntimeError('Use scikit-learn ' + b['sklearn_version'])
predict_reference(b['model'], pd.read_csv(a.input)).to_csv(a.output, index=False)
print('Saved', a.output)
