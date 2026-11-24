"""Step 2 — visual data exploration; no model fitting."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.state import needs_data,jump
from src.knowledge import guide,contextual_tips
from src.dataio import profile


def render():
    st.title('2 · Explore your data')
    st.write('See what is inside the table before deciding what to predict.')
    guide('Explore data')
    if needs_data():return
    df=st.session_state.df
    st.subheader('Data quality')
    p=profile(df)
    a,b,c=st.columns(3)
    a.metric('Missing cells',f'{int(df.isna().sum().sum()):,}')
    b.metric('Duplicate rows',f'{int(df.duplicated().sum()):,}')
    c.metric('Numeric columns',f'{len(df.select_dtypes(include=np.number).columns):,}')
    st.dataframe(p,hide_index=True,use_container_width=True)
    contextual_tips('Explore data')
    with st.expander('Detailed statistics'):
        st.dataframe(df.describe(include='all').T.astype(str),use_container_width=True)
    st.subheader('Investigate one variable')
    col=st.selectbox('Variable to visualize',df.columns.tolist())
    s=df[col].dropna()
    if s.empty:st.warning('This column contains no available values.')
    elif pd.api.types.is_numeric_dtype(s):
        fig=px.histogram(df,x=col,nbins=35,title=f'Distribution of {col}')
        st.plotly_chart(fig,use_container_width=True)
        st.caption('A distribution shows how frequently values occur. Tall bars mean many observations fall in that range.')
    else:
        counts=s.astype(str).value_counts().head(20).rename_axis('Value').reset_index(name='Count')
        fig=px.bar(counts,x='Value',y='Count',title=f'Most common values of {col}')
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Showing up to 20 most common categories; rare categories may be omitted from this chart.')
    st.subheader('Relationships between variables')
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric)>=2:
        mode=st.radio('Visualize', ['Scatter plot','Correlation heatmap','Boxplot'],horizontal=True)
        if mode=='Scatter plot':
            x=st.selectbox('Horizontal axis',numeric,index=0,key='eda_x')
            y=st.selectbox('Vertical axis',numeric,index=min(1,len(numeric)-1),key='eda_y')
            data=df[[x,y]].dropna()
            st.plotly_chart(px.scatter(data.sample(min(len(data),3500),random_state=750) if len(data) else data,x=x,y=y,opacity=.60),use_container_width=True)
        elif mode=='Correlation heatmap':
            selected=st.multiselect('Numeric variables to compare',numeric,default=numeric[:min(8,len(numeric))])
            if len(selected)>=2:
                cor=df[selected].corr(numeric_only=True)
                st.plotly_chart(px.imshow(cor,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',text_auto='.2f'),use_container_width=True)
                st.caption('Correlation ranges from -1 to 1. Strong correlation does not prove causation.')
        else:
            var=st.selectbox('Numeric variable',numeric)
            cats=[c for c in df.columns if c!=var and 2<=df[c].nunique(dropna=True)<=15]
            if cats:
                by=st.selectbox('Group by',cats)
                st.plotly_chart(px.box(df,x=by,y=var,points=False),use_container_width=True)
            else:st.info('No low-cardinality group column available.')
    else:st.info('At least two numeric columns are needed for correlation or numeric scatter plots.')
    if st.button('Continue → Define your question',type='primary'):
        jump(2)
