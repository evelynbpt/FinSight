import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer,load_diabetes,make_blobs,load_iris
from src.modeling import MODELS,run_supervised,run_clustering,classification_at_cutoff,compare_k

@pytest.fixture(scope='module')
def cls_df():
    data=load_breast_cancer(as_frame=True).frame.sample(180,random_state=750).reset_index(drop=True)
    data=data.rename(columns={'target':'default'})
    data['region']=np.where(data['mean radius']>14,'East','West')
    data.loc[0:4,'mean texture']=np.nan
    return data

@pytest.fixture(scope='module')
def reg_df():
    df=load_diabetes(as_frame=True).frame.sample(180,random_state=750).reset_index(drop=True)
    df['group']=np.where(df['bmi']>0,'High','Low')
    return df

@pytest.mark.parametrize('name',MODELS['classification'])
def test_classification_models(name,cls_df):
    result=run_supervised(cls_df,'classification',name,'default',['mean radius','mean texture','region'],
                          folds=3,params={'n_estimators':30,'max_depth':4},positive_class=1)
    assert 0<=result.metrics['Accuracy']<=1
    assert 0<=result.metrics['Precision']<=1
    assert 0<=result.metrics['Recall']<=1
    assert 0<=result.metrics['F1']<=1
    assert result.training_info['train_rows']+result.training_info['test_rows']==len(cls_df)
    m,cm,pred=classification_at_cutoff(result,.6,1,5)
    assert int(cm.values.sum())==result.training_info['test_rows']
    assert len(pred)==len(result.prediction_table)

@pytest.mark.parametrize('name',MODELS['regression'])
def test_regression_models(name,reg_df):
    result=run_supervised(reg_df,'regression',name,'target',['bmi','bp','s5','group'],folds=3,
                          params={'n_estimators':30,'max_depth':4})
    assert np.isfinite(result.metrics['RMSE'])
    assert np.isfinite(result.metrics['MAE'])
    assert np.isfinite(result.metrics['R²'])
    assert result.metrics['Baseline RMSE']>0


def test_multiclass():
    iris=load_iris(as_frame=True).frame.copy()
    result=run_supervised(iris,'classification','Random Forest Classifier','target',list(iris.columns[:4]),
                          folds=3,params={'n_estimators':30})
    m,cm,pred=classification_at_cutoff(result)
    assert cm.shape==(3,3)
    assert 'ROC-AUC' in result.metrics


def test_clustering_and_k():
    x,_=make_blobs(n_samples=120,centers=3,random_state=750)
    df=pd.DataFrame(x,columns=['Balance','Spending'])
    grid=compare_k(df,['Balance','Spending'],maximum=5)
    assert list(grid.K)==[2,3,4,5]
    result=run_clustering(df,['Balance','Spending'],n_clusters=3)
    assert result.metrics['Groups']==3
    assert len(result.prediction_table)==120
    assert result.profiles.shape[0]==3
    assert -1<=result.metrics['Silhouette']<=1


def test_tuned_ridge(reg_df):
    result=run_supervised(reg_df,'regression','Ridge Regression','target',['bmi','bp','s5'],folds=3,tune=True)
    assert result.training_info['tuned'] is True
    assert 'alpha' in result.training_info['best_parameters']


def test_target_leakage_rejected(cls_df):
    with pytest.raises(ValueError,match='target'):
        run_supervised(cls_df,'classification','Logistic Regression','default',['default','mean radius'],folds=3)
