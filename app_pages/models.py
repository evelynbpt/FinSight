"""Step 5 — choose task-compatible algorithm with plain-language tradeoffs."""
import streamlit as st
from src.modeling import MODELS,DESCRIPTIONS
from src.state import needs_problem,jump
from src.knowledge import guide


def render():
    st.title('5 · Choose your machine learning model')
    st.write('Pick a model yourself, or start with a simple baseline. You can compare other models later.')
    guide('Choose a model')
    if needs_problem():return
    task=st.session_state.task
    choices=MODELS[task]
    current=st.session_state.model_name if st.session_state.model_name in choices else choices[0]
    default=choices.index(current)
    st.subheader('Compatible models for '+task)
    options=st.radio('Select a model',choices,index=default,format_func=lambda m:m)
    st.session_state.model_name=options
    st.info(DESCRIPTIONS[options])
    with st.expander('Compare the options in simple terms'):
        for name in choices:
            st.markdown(f'**{name}:** {DESCRIPTIONS[name]}')
    st.markdown('**Recommended learning path:** start with the first model above, then compare it against a more complex model. Better test performance matters more than model complexity.')
    if task=='classification':
        st.warning('For categories like default vs non-default, check class balance and remember that missing a default can be more costly than raising an unnecessary alert.')
    elif task=='regression':
        st.warning('Predictions are estimates. Do not interpret coefficients or feature importance as evidence of cause and effect.')
    else:
        st.warning('K-Means labels are arbitrary. Clustering does not predict loan default or know which segment is financially best.')
    st.caption('💬 Ask the sidebar assistant: “Why should I choose this model over Random Forest?”')
    if st.button('Continue → Train the model',type='primary'):
        jump(5)
