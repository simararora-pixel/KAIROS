"""Run from the project root: python -m streamlit run dashboard/app.py"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.core import load_workspace, forecast, outcome, explanation, evaluation, LABELS, NAMES, FEATURES

st.set_page_config(page_title='KAIROS | Market movement research',layout='wide',initial_sidebar_state='expanded')
COLORS = ['#9B514C','#74879A','#387A66']

@st.cache_resource(show_spinner=False)
def workspace():
    return load_workspace(ROOT)

try:
    w = workspace()
except Exception as e:
    st.error(f'Unable to load research files: {e}')
    st.info('Use the downloaded complete package, or generate the notebook exports as described in README.md.')
    st.stop()


def chart(fig,key):
    fig.update_layout(template='plotly_white',height=340,margin=dict(l=12,r=12,t=25,b=12),font=dict(size=13))
    st.plotly_chart(fig,use_container_width=True,key=key)


def csv_download(frame,name,key):
    st.download_button('Download CSV',frame.to_csv(index=False).encode(),file_name=name,mime='text/csv',key=key)


def scores_view(f,key):
    st.subheader(LABELS[f['prediction']])
    s = pd.DataFrame({'Class':list(LABELS.values()),'Score':f['scores']})
    fig = px.bar(s,x='Score',y='Class',orientation='h',color='Class',color_discrete_sequence=COLORS,text_auto='.1%')
    fig.update_xaxes(range=[0,1],tickformat='.0%')
    fig.update_layout(showlegend=False)
    chart(fig,key)
    st.caption('Uncalibrated class scores, not verified event probabilities. Prediction is the largest score.')
    with st.expander('Seven model inputs'):
        st.dataframe(f['features'].rename(index=NAMES).rename('Value').to_frame(),use_container_width=True)


st.title('KAIROS')
st.caption('NIFTY 50 movement classification  |  UML501 machine learning research')
st.sidebar.header('Research snapshot')
st.sidebar.markdown('**One Random Forest**\n\nNIFTY 50 + India VIX\n\nFive observed sessions · ±3% threshold')
st.sidebar.caption(f"Data through {w['market'].Date.iloc[-1]}. Historical research only; no live feed.")
st.sidebar.warning('Not investment advice. The model missed every major decline in the historical test set.')
tabs = st.tabs(['Overview','Datasets','Market analysis','Prediction engine','Historical simulator','Performance','Explainable AI','Methodology'])

with tabs[0]:
    st.header('Market overview')
    m = w['market']; last = m.iloc[-1]
    a,b,c,d = st.columns(4)
    a.metric('NIFTY 50 close',f"{last.Close:,.2f}")
    b.metric('India VIX',f"{last.VIX_Close:.2f}")
    c.metric('Aligned sessions',f'{len(m):,}')
    d.metric('Latest snapshot',last.Date)
    st.caption('Last available observations in the supplied snapshot; not live quotes.')
    lookback = st.selectbox('Chart window (observations)',[60,250,1000,len(m)],index=1,key='overview_window')
    chart(px.line(m.tail(lookback),x='Date',y='Close',labels={'Close':'NIFTY 50 close'}),'overview_close')
    chart(px.line(m.tail(lookback),x='Date',y='VIX_Close',labels={'VIX_Close':'India VIX close'},color_discrete_sequence=[COLORS[1]]),'overview_vix')

with tabs[1]:
    st.header('Dataset explorer')
    choice = st.selectbox('Dataset',['Raw NIFTY 50','Raw India VIX','Aligned prices','Labelled ML dataset','Imputation audit'],key='dataset_choice')
    key = dict(zip(['Raw NIFTY 50','Raw India VIX','Aligned prices','Labelled ML dataset','Imputation audit'],['nifty','vix','market','dataset','imputation']))[choice]
    data=w[key]
    if key=='dataset':
        partition=st.selectbox('Partition',['All','train','validation','test'])
        if partition!='All': data=data.loc[data.Partition.eq(partition)]
    a,b,c=st.columns(3)
    a.metric('Rows',f'{len(data):,}'); b.metric('Columns',len(data.columns)); c.metric('Missing cells',int(data.isna().sum().sum()))
    st.caption(f"Date range: {data.Date.min()} to {data.Date.max()}")
    st.dataframe(data,use_container_width=True,height=380,hide_index=True)
    csv_download(data,f'KAIROS_{key}.csv','dataset_download')
    st.subheader('Chronological partition audit'); st.dataframe(w['audit'],hide_index=True,use_container_width=True)
    st.caption('One missing VIX close is forward-filled from the prior observation; two affected feature rows are excluded.')

with tabs[2]:
    st.header('Market analysis')
    st.caption('Training partition only for exploratory correlations and distributions.')
    train=w['dataset'].loc[w['dataset'].Partition.eq('train')]
    corr=train[FEATURES].rename(columns=NAMES).corr()
    chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto'),'correlations')
    feature=st.selectbox('Feature distribution',FEATURES,format_func=NAMES.get,key='analytics_feature')
    chart(px.histogram(train,x=feature,nbins=45,labels={feature:NAMES[feature]}),'feature_distribution')
    counts=train.Target.map(LABELS).value_counts().reindex(LABELS.values()).rename_axis('Class').reset_index(name='Rows')
    chart(px.bar(counts,x='Class',y='Rows',color='Class',color_discrete_sequence=COLORS,text_auto=True),'training_counts')
    st.caption('Correlation does not establish causation or forecasting ability.')

with tabs[3]:
    st.header('Prediction engine')
    st.caption('Generate a prediction from the saved Random Forest for one of the five unlabelled snapshot observations. These are not current forecasts.')
    date=st.selectbox('Forecast date',w['inputs'].Date.tolist()[::-1],key='latest_date')
    scores_view(forecast(w,date,latest=True),'latest_scores')
    st.info('The five-session outcome is not available in the supplied snapshot. No actual return is invented.')

with tabs[4]:
    st.header('Historical simulator')
    st.caption('Select a held-out historical forecast date, inspect the model output, then reveal the recorded five-session outcome.')
    date=st.selectbox('Historical forecast date',w['test'].Date.tolist(),key='sim_date',on_change=lambda:st.session_state.update(sim_reveal=False))
    f=forecast(w,date)
    scores_view(f,'sim_scores')
    reveal=st.checkbox('Reveal actual outcome',value=False,key='sim_reveal')
    observed=outcome(w,date,reveal)
    if observed is not None:
        a,b,c=st.columns(3)
        a.metric('Actual movement',LABELS[observed['target']]); b.metric('Five-session return',f"{observed['return_pct']:+.2f}%"); c.metric('Outcome date',observed['date'])
        if observed['correct']: st.success('Correct classification')
        else: st.warning('Incorrect classification')
    else:
        st.info('Actual outcome hidden. Check the box to reveal it.')

with tabs[5]:
    st.header('Model performance')
    period=st.radio('Evaluation period',['test','validation'],horizontal=True,key='performance_period')
    met,cm=evaluation(w,period)
    a,b,c=st.columns(3)
    a.metric('Accuracy',f"{met['Accuracy']:.2%}"); b.metric('Macro F1',f"{met['Macro F1']:.4f}"); c.metric('Balanced accuracy',f"{met['Balanced accuracy']:.2%}")
    st.caption(f"{len(w[period])} observations · Major Down recall {met['Major Down recall']:.2%} · Major Up recall {met['Major Up recall']:.2%}")
    chart(px.imshow(cm,x=list(LABELS.values()),y=list(LABELS.values()),text_auto=True,color_continuous_scale='Blues',labels={'x':'Predicted','y':'Actual','color':'Count'}),'confusion')
    comp=w['metrics'].loc[w['metrics'].Period.eq(period)].drop(columns='Period')
    chart(px.bar(comp,x='Model',y='Macro F1',text_auto='.3f',color_discrete_sequence=[COLORS[1]]),'benchmarks')
    st.dataframe(comp.set_index('Model').style.format('{:.4f}'),use_container_width=True)
    st.warning('Majority and momentum are comparison rules, not deployed ML models. Historical test results are not independent validation.')

with tabs[6]:
    st.header('Explainable AI')
    imp=w['importance'].copy(); imp['Feature name']=imp.Feature.map(NAMES)
    mode=st.radio('Global importance',['Permutation mean','Impurity importance'],horizontal=True,key='importance_mode')
    chart(px.bar(imp.sort_values(mode),x=mode,y='Feature name',orientation='h',error_x='Permutation SD' if mode=='Permutation mean' else None,color_discrete_sequence=[COLORS[1]]),'global_importance')
    st.caption('Permutation: validation macro F1, 20 repetitions. Impurity: training split contributions. Neither establishes causality.')
    scope=st.radio('Explanation scope',['Historical test','Latest snapshot'],horizontal=True,key='xai_scope')
    latest=scope=='Latest snapshot'
    dates=w['latest' if latest else 'test'].Date.tolist()
    date=st.selectbox('Explanation date',dates,key='xai_date')
    f=forecast(w,date,latest=latest)
    label=st.selectbox('Explain class',list(LABELS),index=list(LABELS).index(f['prediction']),format_func=LABELS.get,key=f'xai_class_{scope}_{date}')
    ex=explanation(w,date,label,latest=latest)
    a,b=st.columns(2); a.metric('Base score',f"{ex.Base_score.iloc[0]:.2%}"); b.metric('Model score',f"{ex.Model_score.iloc[0]:.2%}")
    ex['Contribution (pp)']=ex.SHAP*100; ex['Effect']=ex.SHAP.ge(0).map({True:'Increases score',False:'Decreases score'})
    chart(px.bar(ex,x='Contribution (pp)',y='Feature name',orientation='h',color='Effect',color_discrete_map={'Increases score':COLORS[2],'Decreases score':COLORS[0]}),'local_shap')
    st.dataframe(ex[['Feature name','Value','Contribution (pp)']],hide_index=True,use_container_width=True)
    st.caption('Tree SHAP: base score plus contributions equals the class score. Explanations describe model behavior, not market causality.')

with tabs[7]:
    st.header('Research methodology')
    st.markdown('**Objective.** Classify the net NIFTY 50 return over the next five observed market sessions: above +3% is Major Up; below -3% is Major Down; otherwise Ordinary.')
    st.markdown('**Pipeline.** Align NIFTY and India VIX by date, check missing values, compute seven trailing features, construct targets within chronological partitions, train one balanced Random Forest, evaluate and explain.')
    st.dataframe(w['audit'],hide_index=True,use_container_width=True)
    st.subheader('Locked model configuration')
    params=w['bundle']['model'].get_params(); keys=['n_estimators','max_depth','min_samples_leaf','class_weight','criterion','max_features','bootstrap','random_state']
    st.dataframe(pd.DataFrame({'Parameter':keys,'Value':[str(params[k]) for k in keys]}),hide_index=True,use_container_width=True)
    st.subheader('Input features'); st.dataframe(pd.DataFrame({'Feature':FEATURES,'Meaning':[NAMES[x] for x in FEATURES]}),hide_index=True,use_container_width=True)
    st.markdown('**Limitations.** Historical snapshot and overlapping outcome windows; previously inspected test period; uncalibrated scores; imbalanced targets; zero test decline recall. Source dates and prices have not been independently reconciled to the official exchange calendar. No trading performance is established.')
    st.caption('AI-assisted development was used. Review the notebook and independently verify the source data before academic submission.')
