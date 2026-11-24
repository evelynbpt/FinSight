"""FinSight: proposal-aligned AI-assisted financial ML toolbox."""
import streamlit as st
from src.state import STEPS,initialize
from src.chat import sidebar_chat,api_key
from app_pages import upload,explore,problem,prepare,models,train,results,advisor

st.set_page_config(page_title='FinSight | AI-assisted analytics',page_icon='📊',layout='wide',initial_sidebar_state='expanded')
initialize()
if 'jump_to' in st.session_state:
    st.session_state.selected_page=st.session_state.pop('jump_to')

st.markdown('''<style>
.block-container {max-width:1260px; padding-top:1.65rem; padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {padding:16px; border-radius:12px; border:1px solid rgba(110,140,160,.32); background:rgba(120,154,170,.10)}
[data-testid="stMetricValue"] {font-weight:750}
[data-testid="stSidebar"] {border-right:1px solid rgba(100,136,163,.22)}
.guide-kicker {font-size:.82rem; font-weight:600; letter-spacing:.06rem; text-transform:uppercase; opacity:.72}
</style>''',unsafe_allow_html=True)

with st.sidebar:
    st.title('📊 FinSight')
    st.caption('AI-powered machine learning for financial analysts')
    st.divider()
    selected=st.radio('Your analysis journey',STEPS,key='selected_page')
    st.divider()
    if st.session_state.df is not None:
        st.success(f'✓ {len(st.session_state.df):,} rows loaded')
        st.caption('File: '+st.session_state.filename)
    else:
        st.info('Start by uploading a CSV or Excel file.')
    if st.session_state.get('result'):
        st.caption('Latest model: '+st.session_state.result.model_name)
    st.caption('Live AI connected' if api_key() else 'AI: built-in guide (API key not configured)')
    sidebar_chat()
    st.divider()
    st.caption('BSAN 750 · Educational decision-support tool. Not for real loan approvals.')

MODULES=[upload,explore,problem,prepare,models,train,results,advisor]
index=STEPS.index(selected)
st.caption(f'FINANCIAL ANALYTICS WORKFLOW  ·  STEP {index+1} OF 8')
MODULES[index].render()
