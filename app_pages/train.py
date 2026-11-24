"""Step 6 — split, cross-validate and fit sklearn estimators."""
import streamlit as st
from src.modeling import MODELS,run_supervised,run_clustering,compare_k
from src.state import needs_problem,jump
from src.knowledge import guide,contextual_tips


def render():
    st.title('6 · Train and tune your model')
    st.write('This is where Python actually learns patterns from the data. The AI assistant explains settings, but does not invent results.')
    guide('Train & tune')
    if needs_problem():return
    df=st.session_state.df;task=st.session_state.task;name=st.session_state.model_name
    features=st.session_state.features;target=st.session_state.target
    if not features:
        st.warning('Choose predictor variables on Step 4 first.');return
    if name not in MODELS[task]:
        st.warning('Choose a model on Step 5 first.');return
    st.markdown(f'**Selected model:** {name} · **Selected predictors:** {len(features)}'+(f' · **Target:** {target}' if target else ''))
    if task=='clustering':
        st.subheader('Choose how many groups to discover')
        st.caption('We do not know the “correct” number of groups in advance. Compare several K values using actual scores.')
        max_k=max(2,min(8,len(df)-1))
        if st.button('Compare K = 2 through '+str(max_k)):
            with st.spinner('Checking cluster quality for several K values...'):
                try:
                    comparison=compare_k(df,features,max_k)
                    st.session_state.k_comparison=comparison
                    st.session_state.k_comparison_key=(st.session_state.data_version,tuple(features))
                except Exception as exc:st.error(str(exc))
        comp=st.session_state.get('k_comparison') if st.session_state.get('k_comparison_key')==(st.session_state.data_version,tuple(features)) else None
        if comp is not None and not comp.empty:
            st.dataframe(comp.style.format({'Silhouette':'{:.3f}','Within-cluster SS':'{:.2f}'}),use_container_width=True,hide_index=True)
            st.line_chart(comp.set_index('K')[['Silhouette']])
            st.line_chart(comp.set_index('K')[['Within-cluster SS']])
            row=comp.loc[comp['Silhouette'].idxmax()]
            st.info(f'**Data-based starting suggestion: K = {int(row["K"])}**, because it has the highest tested silhouette score ({row["Silhouette"]:.3f}). This is not guaranteed to be the most useful business segmentation.')
        n_clusters=st.slider('Number of groups (K)',min_value=2,max_value=max_k,value=min(3,max_k))
        if st.button('Train K-Means',type='primary'):
            with st.spinner('Learning groups from the selected variables...'):
                try:
                    res=run_clustering(df,features,n_clusters,st.session_state.data_version)
                    _save(res)
                    st.success('Groups calculated using your actual uploaded data.')
                except Exception as exc:st.error(str(exc))
    else:
        if task=='classification':
            labels=df[target].dropna().value_counts().index.tolist()
            preferred=1 if 1 in labels else 'yes' if 'yes' in labels else labels[-1]
            positive=st.selectbox('Which outcome is the positive event we most want to detect?',labels,
                  index=labels.index(st.session_state.positive_class if st.session_state.positive_class in labels else preferred),
                  help='For default prediction, select the label meaning default. This controls precision, recall and the cutoff analysis.')
            st.session_state.positive_class=positive
        else:positive=None
        st.subheader('Training settings')
        st.caption('Choose a simple setup first. Turn on tuning only after the baseline model runs successfully.')
        with st.form('train_form'):
            col1,col2,col3=st.columns(3)
            test_percent=col1.slider('Test set (%)',15,40,25,5,help='This percentage is held aside and not used to train or tune the model.')
            folds=col2.select_slider('Cross-validation folds',options=[3,4,5],value=5,help='5-fold means five rounds, each using a different validation part of the training data.')
            tune=col3.checkbox('Automatically try a few settings',value=False,help='Uses cross-validation on the training set only. Can take longer.')
            params={}
            if 'Tree' in name or 'Forest' in name or 'Boosting' in name:
                params['max_depth']=st.slider('Maximum tree depth',2,15,5,help='Smaller trees are simpler; very deep trees may memorize training data.')
            if 'Forest' in name or 'Boosting' in name:
                params['n_estimators']=st.slider('Number of trees',30,200,100,10)
            if name in ('Ridge Regression','Lasso Regression'):
                params['alpha']=st.select_slider('Regularization strength (alpha)',options=[0.001,0.01,0.1,1.,10.,100.],value=1.0,
                                                  help='A stronger penalty pulls coefficients closer to zero.')
            if name=='K-Nearest Neighbors':
                params['neighbors']=st.slider('Number of nearest neighbors',1,25,5,2)
            if name=='Gradient Boosting Classifier':
                params['learning_rate']=st.select_slider('Learning rate',options=[0.025,0.05,0.1,0.2],value=0.1)
            start=st.form_submit_button('Train and evaluate model',type='primary',use_container_width=True)
        if start:
            with st.spinner('Training the model and checking performance on unseen rows...'):
                try:
                    res=run_supervised(df,task,name,target,features,test_percent/100,folds,tune,params,positive,st.session_state.data_version)
                    _save(res)
                    st.success('Model successfully trained! Continue to Step 7 to read and compare results.')
                except Exception as exc:st.error('Could not train this model: '+str(exc))
    if st.session_state.result is not None:
        res=st.session_state.result
        st.info('Latest result: **'+res.model_name+'**. Check Step 7 for explanation, evaluation and comparison.')
        if st.button('Continue → Results & comparison',type='primary'):
            jump(6)
    contextual_tips('Train & tune')


def _save(res):
    st.session_state.result=res
    st.session_state.history.append(res)
    # A bounded history avoids growing server memory indefinitely.
    st.session_state.history=st.session_state.history[-12:]
