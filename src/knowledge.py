"""A short, consistent beginner tutorial on every page."""
from __future__ import annotations
import streamlit as st
from src.dataio import feature_warnings, recommended_features, target_suggestions, predict_task_from_goal

LESSONS={
 'Upload data':('Start with a table','Each row should represent one customer, loan, transaction or property. Each column is a fact about that row. Upload only data you have permission to use.','Check whether the number of rows and columns makes sense.'),
 'Explore data':('Get to know the data','A missing value means a cell has no answer. A distribution shows whether values are usually high, low, or spread out. A correlation shows variables that change together; it does **not** prove one causes the other.','Inspect missing values, unusual values, and the column you want to predict.'),
 'Define the problem':('Describe the result you want','**Classification:** predict a category, such as default or no default. **Regression:** predict a number, such as a price. **Clustering:** discover groups when there is no known answer column.','Explain your question in everyday words, then choose the type of answer.'),
 'Prepare variables':('Choose the information the model may use','The **target (Y)** is the answer you want to predict; **predictors (X)** are the clues available before the answer happens. Never include the target as a predictor. Missing-value filling, category encoding and scaling happen in the training pipeline.','Remove IDs and information only known after the outcome; start with a manageable set of meaningful predictors.'),
 'Choose a model':('Choose how the model will learn','Some models are easy to explain (Linear or Logistic Regression, Decision Trees). Others can capture complicated relationships (Random Forest, Gradient Boosting). K-Means groups observations without knowing a target.','Start with one simple baseline, then compare it fairly with a more flexible model.'),
 'Train & tune':('Learn from past cases, then test on unseen cases','The app keeps a separate **test set**. Tuning and cross-validation happen only within the training set, so the test results can better estimate how the model may perform on new cases.','For K-Means, K is the number of groups. For a decision tree, depth controls complexity. Tuning is not the same as guaranteeing a good model.'),
 'Results & comparison':('Understand errors and limitations','**Accuracy** measures overall correctness. **Precision** asks how many predicted positives were correct. **Recall** asks how many actual positives were found. **RMSE** measures typical numerical error; smaller is generally better.','Compare models using the same target, features and test split; choose metrics that fit the real business cost.'),
 'AI advisor':('Turn results into a cautious business explanation','The assistant explains calculated results and limitations; Python performs all statistics. A model finds patterns, not guaranteed causes. Banking decisions require fairness and independent validation.','Ask specific questions such as “Which metric should I focus on and why?”')
}

def guide(step:str):
    title,explain,next_move=LESSONS[step]
    with st.expander('📖 Explain this step in simple words',expanded=True):
        st.markdown(f'**{title}**')
        st.markdown(explain)
        st.caption('Suggested action: '+next_move)


def contextual_tips(step:str):
    df=st.session_state.get('df')
    if df is None: return
    if step=='Upload data':
        st.caption(f'Your table has {len(df):,} rows and {df.shape[1]} columns. This is a dataset, not yet a trained model.')
    if step=='Explore data':
        missing=df.isna().sum().sum()
        if missing: st.warning(f'There are {missing:,} empty cells. Review them before trusting the model.')
    if step=='Define the problem':
        rec=predict_task_from_goal(st.session_state.get('goal',''))
        if rec: st.info(f'Based on your question, **{rec}** is a useful starting point. You may still choose a different task.')
        if st.session_state.get('task'):
            c=target_suggestions(df,st.session_state.task,st.session_state.get('goal',''))
            if c and st.session_state.task!='clustering': st.caption('Potential targets to inspect: '+', '.join(c[:4])+'. These are suggestions, not verified correct answers.')
    if step=='Prepare variables':
        features=st.session_state.get('features',[])
        for warn in feature_warnings(df,features,st.session_state.get('target'))[:6]: st.warning(warn)
    if step=='Train & tune':
        if st.session_state.get('task')=='clustering': st.caption('A silhouette score can compare K values, but meaningful business segments also require interpreting the cluster profiles.')
    if step=='Results & comparison':
        if st.session_state.get('result') is None: st.info('Run a model on Step 6 before evaluating its results.')
