import argparse
import joblib
import pandas as pd
import sklearn
from src.consolidation import predict_frame, FEATURES

p=argparse.ArgumentParser(description='KAIROS historical research inference. Load trusted model artifacts only.')
p.add_argument('--bundle',default='results/KAIROS_FINAL.joblib')
p.add_argument('--input',required=True,help='Feature-ready CSV with Date and required features; monthly models also require Flow_month')
p.add_argument('--model',choices=FEATURES,default='reference')
p.add_argument('--output',default='predictions.csv')
a=p.parse_args()
b=joblib.load(a.bundle)
if b['sklearn_version']!=sklearn.__version__:
    raise RuntimeError('Use scikit-learn '+b['sklearn_version'])
predict_frame(b,pd.read_csv(a.input),a.model).to_csv(a.output,index=False)
print('Saved',a.output)
