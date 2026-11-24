"""Central session state: one dataset, one consistent analysis workflow."""
import streamlit as st

STEPS=['Upload data','Explore data','Define the problem','Prepare variables','Choose a model','Train & tune','Results & comparison','AI advisor']

def initialize():
    defaults={'df':None,'filename':'','task':None,'target':None,'features':[],'goal':'','model_name':None,'result':None,'history':[], 'chat_history':[],'selected_page':STEPS[0], 'data_version':0,'selected_sheet':'','positive_class':None}
    for key,value in defaults.items():
        if key not in st.session_state: st.session_state[key]=value

def reset_for_dataset(df,name):
    st.session_state.df=df
    st.session_state.filename=name
    st.session_state.data_version+=1
    st.session_state.task=None
    st.session_state.target=None
    st.session_state.features=[]
    st.session_state.model_name=None
    st.session_state.result=None
    st.session_state.history=[]
    st.session_state.chat_history=[]

def jump(step:int):
    st.session_state.jump_to=STEPS[step]
    st.rerun()

def needs_data():
    if st.session_state.df is None:
        st.info('First upload a CSV or Excel file in **Step 1: Upload data**.')
        if st.button('Go to upload',key='to_upload'):
            jump(0)
        return True
    return False

def needs_problem():
    if needs_data(): return True
    if not st.session_state.task:
        st.info('Define your question in **Step 3** so the app knows what you want to learn.')
        return True
    return False
