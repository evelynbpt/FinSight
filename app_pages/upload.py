"""Step 1 — load real data with flexible CSV/Excel options."""
import streamlit as st
from src.dataio import read_dataset,sheet_names,profile
from src.state import reset_for_dataset,jump
from src.knowledge import guide,contextual_tips


def render():
    st.title('1 · Upload your data')
    st.write('Start with a spreadsheet: **each row is one case**, and **each column describes that case**.')
    guide('Upload data')
    st.warning('**Privacy first:** Use public, synthetic-free sample datasets or anonymized, approved data. Do not upload real private customer records or account numbers to an unapproved hosted app.')
    file=st.file_uploader('Choose a CSV or Excel file',type=['csv','xlsx','xls'],help='CSV with commas or semicolons; Excel .xlsx and .xls are supported.')
    if file is not None:
        content=file.getvalue()
        sheet=0
        if file.name.lower().endswith(('.xls','.xlsx')):
            try:
                tabs=sheet_names(content,file.name)
                sheet=st.selectbox('Which Excel sheet should we read?',tabs)
            except Exception as exc:
                st.error('Could not inspect Excel sheets: '+str(exc));return
        if st.button('Use this dataset',type='primary'):
            try:
                df=read_dataset(content,file.name,sheet)
                reset_for_dataset(df,file.name)
                st.success('Dataset ready. Review the preview below, then continue to Step 2.')
            except Exception as exc:st.error(str(exc))
    df=st.session_state.df
    if df is not None:
        st.subheader('Your dataset at a glance')
        a,b,c=st.columns(3)
        a.metric('Rows (observations)',f'{len(df):,}')
        b.metric('Columns (variables)',f'{df.shape[1]:,}')
        c.metric('Empty cells',f'{int(df.isna().sum().sum()):,}')
        st.dataframe(df.head(12),use_container_width=True)
        with st.expander('What types of variables did we find?'):
            st.dataframe(profile(df),hide_index=True,use_container_width=True)
        contextual_tips('Upload data')
        if st.button('Continue → Explore data',type='primary'):
            jump(1)
    else:
        st.info('No dataset selected yet. Download a public dataset using the links in the README, or upload your own CSV/Excel file.')
