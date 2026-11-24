"""Step 7 — holdout metrics, ROC, PR, comparison, profiles and reports."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve,precision_recall_curve
from src.modeling import classification_at_cutoff
from src.knowledge import guide
from src.state import jump
from src.report import report_markdown


def _metric_cols(data:dict,names:list[str]):
    fields=[k for k in names if k in data]
    cols=st.columns(min(4,len(fields))) if fields else []
    for i,k in enumerate(fields):
        v=data[k]
        cols[i%len(cols)].metric(k,f'{v:.3f}' if isinstance(v,(int,float,np.number)) else str(v))


def render():
    st.title('7 · Results & model comparison')
    st.write('See what the model learned, how well it performed, and what its limitations mean for your business question.')
    guide('Results & comparison')
    result=st.session_state.get('result')
    if result is None:
        st.info('No trained model yet. Choose and train a model on Step 6.');return
    if result.data_version!=st.session_state.data_version:
        st.warning('The dataset changed. Please train again.');return
    st.subheader(f'{result.model_name} — verified model output')
    if result.task=='classification':
        binary=len(result.class_names)==2 and result.proba is not None
        cost_fp=cost_fn=1.0
        cutoff=.5
        if binary:
            st.markdown('#### Decide how cautious the classifier should be')
            cutoff=st.slider('Probability cutoff for the positive class',0.05,0.95,0.50,0.05,
                help='Lower cutoff flags more positives. It often increases recall, but also produces more false alarms.')
            with st.expander('Optional business costs: compare false alarms with missed positives'):
                cost_fp=st.number_input('Illustrative cost of a false positive',min_value=0.0,value=1.0,step=1.0)
                cost_fn=st.number_input('Illustrative cost of a missed positive',min_value=0.0,value=5.0,step=1.0)
                st.caption('These are user-defined relative costs, not dollar amounts inferred from the data. They do not automate loan approval.')
        metrics,cm,updated_predictions=classification_at_cutoff(result,cutoff,cost_fp,cost_fn)
        _metric_cols(metrics,['Accuracy','Precision','Recall','F1','ROC-AUC'])
        a,b=st.columns(2)
        with a:
            st.plotly_chart(px.imshow(cm.values,x=cm.columns,y=cm.index,text_auto=True,aspect='auto',color_continuous_scale='Blues',title='Confusion matrix — actual vs predicted'),use_container_width=True)
            st.caption('The diagonal cells are correct predictions. Off-diagonal cells are mistakes.')
        with b:
            if binary:
                pos=result.positive_class
                fpr,tpr,_=roc_curve(np.asarray(result.true_labels)==pos,result.proba)
                graph=go.Figure()
                graph.add_trace(go.Scatter(x=fpr,y=tpr,name='Model',mode='lines'))
                graph.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random reference',mode='lines',line=dict(dash='dash')))
                graph.update_layout(title='ROC curve',xaxis_title='False positive rate',yaxis_title='True positive rate')
                st.plotly_chart(graph,use_container_width=True)
                pr,re,_=precision_recall_curve(np.asarray(result.true_labels)==pos,result.proba)
                st.plotly_chart(px.line(x=re,y=pr,labels={'x':'Recall','y':'Precision'},title='Precision–recall curve'),use_container_width=True)
            else:st.info('ROC and precision–recall curves are shown for binary classification. Macro metrics still describe multiclass performance.')
        st.caption(f'Positive event: **{result.positive_class}**. Precision and recall refer to that event for binary outcomes.')
        if binary:
            st.warning(f'With this cutoff, the model misses **{metrics["FN"]}** actual positive cases and flags **{metrics["FP"]}** false positives in the held-out test data. Consider the real consequences of both error types.')
        if 'CV F1 (weighted)' in result.metrics:st.caption(f'Training-only cross-validation F1 (weighted): {result.metrics["CV F1 (weighted)"]:.3f}. Majority-class test accuracy baseline: {result.metrics["Majority-class baseline"]:.3f}.')
        output=result.prediction_table.copy();output['Predicted']=updated_predictions
    elif result.task=='regression':
        _metric_cols(result.metrics,['RMSE','MAE','R²','CV RMSE'])
        st.caption(f'Baseline RMSE from predicting the training-average every time: **{result.metrics["Baseline RMSE"]:.3f}**. A useful model should generally improve on this baseline.')
        values=result.prediction_table
        chart=px.scatter(values,x='Actual',y='Predicted',title='Observed values vs predictions')
        bottom=float(min(values[['Actual','Predicted']].min()));top=float(max(values[['Actual','Predicted']].max()))
        chart.add_trace(go.Scatter(x=[bottom,top],y=[bottom,top],mode='lines',name='Perfect prediction',line=dict(dash='dash')))
        st.plotly_chart(chart,use_container_width=True)
        st.caption('Points near the dashed line have smaller prediction errors. RMSE and MAE use the same units as your target.')
        output=values
    else:
        _metric_cols(result.metrics,['Silhouette','Within-cluster SS','Groups'])
        a,b=st.columns([2,1])
        with a:
            st.plotly_chart(px.scatter(result.chart,x='PC1',y='PC2',color='Cluster',title='Clusters (two-dimensional PCA view)',opacity=.7),use_container_width=True)
            st.caption('PCA compresses several measurements into two axes for visualization. Some information is lost in this picture.')
        with b:
            st.dataframe(pd.Series(result.metrics['Group sizes'],name='Records'),use_container_width=True)
        st.subheader('Group profiles (original measurement units)')
        st.dataframe(result.profiles,hide_index=True,use_container_width=True)
        st.caption('Interpret the averages, then name the segments in ordinary business language. Group numbers are arbitrary and not rankings.')
        output=result.prediction_table
    if not result.importance.empty:
        st.subheader('Most influential model inputs')
        st.plotly_chart(px.bar(result.importance.head(15).sort_values('Importance'),x='Importance',y='Variable',orientation='h'),use_container_width=True)
        st.caption('Feature importance and coefficient sizes show a model association, not proven causation. Linear coefficients may be standardized or encoded; compare cautiously.')
    st.subheader('Download and inspect results')
    st.dataframe(output.head(25),use_container_width=True)
    c1,c2=st.columns(2)
    c1.download_button('Download predictions / cluster assignments (CSV)',output.to_csv(index=True).encode(),file_name='finsight_results.csv',mime='text/csv',use_container_width=True)
    c2.download_button('Download model report (Markdown)',report_markdown(result,st.session_state.goal),file_name='finsight_report.md',mime='text/markdown',use_container_width=True)
    st.divider()
    st.subheader('Compare previously trained models')
    history=[r for r in st.session_state.history if r.data_version==st.session_state.data_version and r.comparable_signature()==result.comparable_signature()]
    distinct={r.model_name:r for r in history}
    if len(distinct)>1:
        scores=[]
        for name,r in distinct.items():
            row={'Model':name};row.update({k:round(float(v),4) for k,v in r.metrics.items() if isinstance(v,(int,float,np.number)) and k not in ('Groups','Within-cluster SS')})
            scores.append(row)
        st.dataframe(pd.DataFrame(scores),hide_index=True,use_container_width=True)
        st.info('Models shown share the task, target, predictors, and test fraction. For stronger comparisons, avoid choosing a winner based only on the same test set repeatedly.')
    else:
        st.info('Train at least one more compatible model using the same predictors, target, and test percentage to see a comparison table here.')
    st.warning('**Banking caution:** Model scores do not validate a system for automatic credit or loan decisions. Review fairness, representativeness, uncertainty, operating costs, and applicable legal requirements.')
    if st.button('Continue → Ask the AI advisor',type='primary'):
        jump(7)
