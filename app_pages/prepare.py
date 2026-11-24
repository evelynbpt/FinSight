"""Step 4 — user-approved predictors, automatic pipeline preprocessing, warnings."""
import numpy as np
import pandas as pd
import streamlit as st
from src.dataio import recommended_features,feature_warnings,potential_identifier
from src.knowledge import guide
from src.state import needs_problem,jump


def render():
    st.title('4 · Prepare your variables')
    st.write('Choose the columns the model is allowed to look at. The target is never included among predictors.')
    guide('Prepare variables')
    if needs_problem():return
    df=st.session_state.df
    task=st.session_state.task
    target=st.session_state.target
    candidates=[c for c in df.columns if c!=target]
    recommendations=recommended_features(df,target,task)
    current=[x for x in st.session_state.features if x in candidates] or recommendations
    st.caption('Suggested starting features: '+(', '.join(recommendations) if recommendations else 'None — review dataset types.')+'. These are heuristics, NOT a proven optimal variable selection.')
    selected=st.multiselect('Predictor variables (X) — choose one or more',candidates,default=current)
    st.session_state.features=selected
    st.caption(f'{len(selected)} selected · The target is '+(target or 'not needed for clustering'))
    if selected:
        f=df[selected]
        st.dataframe(pd.DataFrame({'Variable':selected,'Type':[str(df[c].dtype) for c in selected],
            'Missing %':[(df[c].isna().mean()*100).round(1) for c in selected],
            'Unique values':[int(df[c].nunique(dropna=True)) for c in selected]}),hide_index=True,use_container_width=True)
    for issue in feature_warnings(df,selected,target):st.warning(issue)
    if task=='clustering' and selected:
        non_numeric=[x for x in selected if not pd.api.types.is_numeric_dtype(df[x])]
        if non_numeric:st.error('K-Means currently requires numeric measurements; remove: '+', '.join(non_numeric))
    st.subheader('What will happen to missing values and categories?')
    a,b=st.columns(2)
    with a:
        st.info('**Numbers:** Fill missing numeric predictor values using the median learned from the training data. Scale numbers for models that need comparable units.')
    with b:
        st.info('**Categories:** Fill missing categories with the most common training value. Convert categories into 0/1 indicator columns.')
    st.caption('These transformations happen *inside* the model pipeline. Cross-validation does not see information from its validation folds during preparation. The target column is never imputed.')
    if st.button('Continue → Choose a model',type='primary',disabled=not bool(selected)):
        st.session_state.result=None
        jump(4)
