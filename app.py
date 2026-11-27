"""KAIROS: local Streamlit dashboard for the single reference forest."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
from src.reference import FEATURES, build_reference, predict_reference
from src.baseline import momentum_predict, evaluate_predictions, CLASS_NAMES
from src.explainability import global_importance, explain_rows

st.set_page_config(page_title="KAIROS | Market research", layout="wide")
ROOT = Path(__file__).resolve().parent
NAMES = dict(zip([-1, 0, 1], CLASS_NAMES))
REQUIRED = ['Date', 'Outcome_date', 'Partition', 'Target', 'Forward_return_5d_pct',
            'Open', 'High', 'Low', 'Close'] + FEATURES

@st.cache_data
def read_dataset(raw):
    d = pd.read_csv(io.BytesIO(raw))
    if not set(REQUIRED).issubset(d.columns):
        raise ValueError("Use KAIROS_FINAL_DATASET.csv from the current notebook.")
    d['Date'] = pd.to_datetime(d.Date, errors='raise')
    d['Outcome_date'] = pd.to_datetime(d.Outcome_date, errors='raise')
    if d[['Date','Outcome_date','Partition']].isna().any().any() or d.Date.duplicated().any() or not d.Date.is_monotonic_increasing:
        raise ValueError("Invalid or unordered dates/partitions.")
    cols = list(dict.fromkeys(REQUIRED[3:]))
    d[cols] = d[cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(d[cols].to_numpy()).all():
        raise ValueError("Missing or nonfinite numeric values.")
    if set(d.Partition) != {'train','validation','test'} or not d.Target.isin([-1,0,1]).all():
        raise ValueError("Unexpected partitions or labels.")
    parts = [d[d.Partition.eq(p)] for p in ('train','validation','test')]
    if any(set(p.Target) != {-1,0,1} for p in parts):
        raise ValueError("Each partition must contain all three classes.")
    if any(a.Outcome_date.max() >= b.Date.min() for a,b in zip(parts,parts[1:])) or not d.Outcome_date.gt(d.Date).all():
        raise ValueError("Outcome dates cross partition boundaries.")
    r = d.Forward_return_5d_pct.to_numpy()
    boundary = np.isclose(np.abs(r),3,rtol=0,atol=1e-10)
    expected = np.select([(r < -3)&~boundary,(r > 3)&~boundary],[-1,1],default=0)
    if not np.array_equal(d.Target,expected):
        raise ValueError("Target does not match the five-session threshold.")
    return d

@st.cache_resource
def fitted(raw):
    d = read_dataset(raw)
    train = d[d.Partition.eq('train')]
    return build_reference().fit(train[FEATURES],train.Target)

@st.cache_data
def validation_importance(raw):
    d = read_dataset(raw)
    v = d[d.Partition.eq('validation')]
    return global_importance(fitted(raw),v[FEATURES],v.Target)

def prediction_panel(row, model):
    out = predict_reference(model,row).iloc[0]
    st.subheader(NAMES[int(out.Prediction)])
    cols = st.columns(3)
    for col, label, key in zip(cols,CLASS_NAMES,['Score_Down','Score_Ordinary','Score_Up']):
        col.metric(label,f"{out[key]:.1%}")
    st.caption("Uncalibrated class scores, not guaranteed probabilities. Five trading sessions, ±3% threshold.")
    return out

st.title("KAIROS")
st.caption("NIFTY 50 / India VIX · Historical market movement research · UML501")
st.sidebar.header("Data")
upload = st.sidebar.file_uploader("Combined dataset CSV",type=['csv'])
local = ROOT / 'kairos_datasets' / 'KAIROS_FINAL_DATASET.csv'
if upload is None and not local.is_file():
    st.info("Run notebooks/KAIROS.ipynb with the two original source CSVs. Upload the exported KAIROS_FINAL_DATASET.csv here, or place it in kairos_datasets/ beside app.py.")
    st.stop()
raw = upload.getvalue() if upload is not None else local.read_bytes()
try:
    d = read_dataset(raw)
    model = fitted(raw)
except (ValueError, AssertionError, KeyError, TypeError) as exc:
    st.error(str(exc))
    st.stop()
train = d[d.Partition.eq('train')]
val = d[d.Partition.eq('validation')]
test = d[d.Partition.eq('test')]
st.sidebar.caption("One balanced Random Forest; training partition only.")
st.sidebar.write(f"Training: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")
st.sidebar.warning("Historical research only. Not a live market feed or trading system.")
tabs = st.tabs(["Market overview","Dataset explorer","Technical analysis","Prediction engine",
                "Market replay","Model performance","Explainable AI","Methodology"])

with tabs[0]:
    st.header("Market overview")
    last = d.iloc[-1]
    a,b,c = st.columns(3)
    a.metric("NIFTY close",f"{last.Close:,.2f}")
    b.metric("India VIX",f"{last.VIX_Close:.2f}")
    c.metric("Last labelled date",last.Date.strftime('%d %b %Y'))
    st.caption("Latest eligible labelled row of the uploaded snapshot, not current market data.")
    st.line_chart(d.set_index('Date').Close)
    st.line_chart(d.set_index('Date').VIX_Close)

with tabs[1]:
    st.header("Dataset explorer")
    st.dataframe(d.groupby('Partition',sort=False).size().rename('Rows'))
    partition = st.selectbox("Partition",['all','train','validation','test'],key='explore')
    view = d if partition=='all' else d[d.Partition.eq(partition)]
    st.dataframe(view,hide_index=True)
    st.download_button("Download selected rows",view.to_csv(index=False),"KAIROS_subset.csv","text/csv")
    st.caption("Forward return, outcome date and target are evaluation fields, never model inputs.")

with tabs[2]:
    st.header("Technical analysis")
    feature = st.selectbox("Indicator",FEATURES,key='technical')
    st.line_chart(d.set_index('Date')[feature])
    st.subheader("Training-period feature correlations")
    st.dataframe(train[FEATURES].corr().round(3))
    st.caption("Correlation does not establish causation.")

with tabs[3]:
    st.header("Prediction engine")
    st.caption("Select an actual historical feature row from the test period. The model sees only the seven inputs.")
    day = st.selectbox("Forecast date",test.Date.dt.strftime('%Y-%m-%d').tolist(),key='forecast')
    row = test[test.Date.eq(pd.Timestamp(day))]
    st.dataframe(row[FEATURES].T.rename(columns={row.index[0]:'Value'}))
    prediction_panel(row,model)

with tabs[4]:
    st.header("Historical market replay")
    day = st.selectbox("Replay date",test.Date.dt.strftime('%Y-%m-%d').tolist(),key='replay')
    row = test[test.Date.eq(pd.Timestamp(day))]
    forecast = prediction_panel(row,model)
    st.line_chart(d.loc[d.Date.le(pd.Timestamp(day))].tail(60).set_index('Date').Close)
    if st.checkbox("Reveal actual outcome",key='reveal_'+day,value=False):
        actual = row.iloc[0]
        st.write("Outcome date:",actual.Outcome_date.strftime('%d %b %Y'))
        st.metric("Actual forward return",f"{actual.Forward_return_5d_pct:+.2f}%")
        st.write("Actual class:",NAMES[int(actual.Target)])
        if forecast.Prediction == actual.Target:
            st.success("Correct classification")
        else:
            st.error("Incorrect classification")
    else:
        st.info("Actual outcome hidden. Tick the checkbox to reveal it.")
    st.caption("The chart stops on the selected forecast date. Uncheck the box to hide the outcome again.")

with tabs[5]:
    st.header("Model performance")
    period = st.radio("Evaluation period",['validation','test'],index=1,horizontal=True)
    part = val if period=='validation' else test
    pred = predict_reference(model,part).Prediction
    rows = []
    for name,p in [('Random Forest',pred),('Majority rule',np.full(len(part),train.Target.mode().iloc[0])),
                   ('Momentum rule',momentum_predict(part))]:
        metrics,report,matrix = evaluate_predictions(part.Target,p)
        rows.append({'Method':name,**metrics})
        if name=='Random Forest':
            rf_report,rf_matrix = report,matrix
    st.dataframe(pd.DataFrame(rows).set_index('Method').round(4))
    st.subheader("Confusion matrix")
    st.caption("Rows = actual; columns = predicted.")
    st.dataframe(rf_matrix)
    st.dataframe(rf_report.round(4))
    missed = int(((part.Target==-1)&(pred!=-1)).sum())
    false_alarms = int(((part.Target==0)&(pred!=0)).sum())
    st.write(f"Missed major declines: {missed}. Ordinary windows falsely flagged: {false_alarms}.")
    st.caption("The historical test period has already been inspected; no independent prospective validation.")

with tabs[6]:
    st.header("Explainable AI")
    st.subheader("Impurity-based importance")
    st.bar_chart(pd.Series(model.feature_importances_,index=FEATURES,name='Importance'))
    if st.checkbox("Calculate validation permutation importance",key='perm'):
        imp = validation_importance(raw)
        st.bar_chart(imp.set_index('Feature')['Permutation mean'])
        st.dataframe(imp.round(4),hide_index=True)
    day = st.selectbox("Explanation date",test.Date.dt.strftime('%Y-%m-%d').tolist(),key='shap_date')
    label = st.selectbox("Class to explain",[-1,0,1],format_func=lambda k:NAMES[k])
    if st.checkbox("Show individual SHAP contributions",key='show_shap'):
        row = test[test.Date.eq(pd.Timestamp(day))]
        exp = explain_rows(model,row,FEATURES)
        view = exp[exp.Class.eq(label)]
        st.metric("Baseline score",f"{view.Base_score.iloc[0]:.2%}")
        st.metric("Model score",f"{view.Model_score.iloc[0]:.2%}")
        st.bar_chart(view.set_index('Feature').SHAP)
        st.dataframe(view[['Feature','Value','SHAP']].round(5),hide_index=True)
    st.caption("SHAP explains the forest's score, not the causes of market movements. Permutation importance is sensitive to correlated features and time dependence.")

with tabs[7]:
    st.header("Research methodology")
    st.write("Source CSVs → date alignment and missing-value checks → seven trailing features → chronological partitions → one Random Forest → historical evaluation and XAI.")
    st.write("200 trees, maximum depth 8, minimum leaf size 10, balanced class weighting, Gini criterion, seed 42. No model switching or additional risk classifier.")
    st.write("Target: return after five trading sessions; below −3% is Major Down, above +3% is Major Up, otherwise Ordinary.")
    st.write("Training, validation and test dates are kept in chronological order; future outcomes are excluded from the predictors.")
    st.warning("Class imbalance, overlapping outcomes, previously inspected test data and weak downside recall limit interpretation. Not a trading system.")
    st.caption("AI-assisted development; independently review the notebook and results.")
