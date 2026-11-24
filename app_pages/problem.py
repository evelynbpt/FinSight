"""Step 3 — define a business question and choose valid target."""
import streamlit as st
from src.dataio import predict_task_from_goal,target_suggestions
from src.state import needs_data,jump
from src.knowledge import guide,contextual_tips

TASK_LABELS={
 'classification':'Classification — predict a category (default / no default)',
 'regression':'Regression — predict a number (price / revenue)',
 'clustering':'Clustering — discover similar groups (no target)'
}

def render():
    st.title('3 · Describe your business problem')
    st.write('You do not need machine learning vocabulary. Tell us what question you want to answer.')
    guide('Define the problem')
    if needs_data():return
    df=st.session_state.df
    goal=st.text_area('What would you like to learn or predict?',value=st.session_state.goal,
        placeholder='Example: Which customers may default on their loans? Or: What groups of customers have similar behavior?',height=90)
    st.session_state.goal=goal
    suggestion=predict_task_from_goal(goal)
    if suggestion:
        st.info(f'💡 **Suggestion: {suggestion.title()}**. We inferred this from the words in your question; you have the final choice.')
    else:
        st.caption('Describe your question using words such as “predict whether”, “predict how much”, or “group similar” for an automatic suggestion.')
    options=list(TASK_LABELS)
    current=st.session_state.task or suggestion or 'classification'
    task=st.radio('What kind of answer do you need?',options,index=options.index(current),format_func=lambda x:TASK_LABELS[x])
    st.session_state.task=task
    if task=='clustering':
        st.session_state.target=None
        st.success('Clustering does not require a target. The model discovers groups using the features you choose in Step 4.')
    else:
        choices=target_suggestions(df,task,goal)
        if not choices:
            st.error('No suitable target found for this task. Choose a different task or upload data with a suitable outcome column.')
            return
        current_target=st.session_state.target if st.session_state.target in choices else choices[0]
        target=st.selectbox('Which column contains the answer you want to predict (Y)?',choices,index=choices.index(current_target),
          help='For loan default, select a historical column like “default payment next month”, not income or customer ID.')
        st.session_state.target=target
        series=df[target]
        if task=='classification':
            counts=series.value_counts(dropna=True).head(15)
            st.dataframe(counts.rename_axis('Class').reset_index(name='Rows'),hide_index=True,use_container_width=True)
            if len(counts)>1 and counts.min()<10:st.warning('Some classes are rare. Model evaluation may be unstable; consider more data or fewer cross-validation folds.')
        else:
            st.caption(f'{int(series.notna().sum()):,} non-empty target values; {series.nunique(dropna=True):,} distinct values.')
        st.warning('Only choose a target that is already observed in your historical data. If the true outcomes are unknown, supervised predictions cannot be trained.')
    contextual_tips('Define the problem')
    st.markdown('**Need help?** Open the persistent AI assistant in the left sidebar and ask “Which model matches my business question?”')
    if st.button('Continue → Choose predictors',type='primary'):
        st.session_state.result=None
        jump(3)
