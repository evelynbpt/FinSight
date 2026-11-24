"""Step 8 — full persistent chat and contextual decision support."""
import streamlit as st
from src.chat import submit,api_key,build_context
from src.knowledge import guide
from src.state import needs_data

SUGGESTIONS=['Which model would you recommend and why?',
             'Which target and predictors should I consider?',
             'Explain my latest model scores like I am new to machine learning.',
             'What are the biggest weaknesses or risks of this analysis?',
             'How could a banking analyst use these results responsibly?']

def render():
    st.title('8 · Ask your FinSight AI advisor')
    st.write('A learning assistant for your entire project. The same conversation is kept when you visit other pages during this session.')
    guide('AI advisor')
    if api_key():st.success('Live AI connected. It can explain the dataset summaries and actual model calculations.')
    else:st.info('The live OpenAI API is not connected yet. Built-in guidance still works for common beginner questions. Add the instructor-provided API key in Streamlit Secrets to enable free-form explanations.')
    st.caption('Privacy: the live AI receives dataset column statistics and aggregate modeling results — **not raw customer records**. Chat text you type is sent to the AI provider when enabled. Avoid typing private customer information.')
    st.subheader('Questions you might ask')
    cols=st.columns(2)
    for idx,question in enumerate(SUGGESTIONS):
        if cols[idx%2].button(question,key=f'suggest_{idx}',use_container_width=True):
            with st.spinner('Writing an explanation...'):
                submit(question)
            st.rerun()
    st.divider()
    if not st.session_state.chat_history: st.info('Ask a question to begin your conversation.')
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):st.markdown(message['content'])
    new_message=st.chat_input('Ask me anything about your data or machine learning choices...')
    if new_message:
        with st.spinner('Reviewing the analysis...'):
            submit(new_message)
        st.rerun()
    if st.button('Clear conversation',key='chat_clear'):
        st.session_state.chat_history=[]
        st.rerun()
    with st.expander('What does the assistant know about this analysis?'):
        ctx=build_context()
        st.json({k:v for k,v in ctx.items() if k!='dataset'})
        st.caption('Column-level summaries are also provided to the AI if a dataset is uploaded. Raw customer records are excluded.')
