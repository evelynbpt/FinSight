"""Actual scikit-learn models. Important: preprocessing is inside the train/CV pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    roc_auc_score,confusion_matrix,mean_squared_error,mean_absolute_error,r2_score,
    silhouette_score,roc_curve,precision_recall_curve)
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold,GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from src.dataio import potential_identifier

MODELS={
 'classification':['Logistic Regression','CART Classification Tree','Random Forest Classifier','Gradient Boosting Classifier','K-Nearest Neighbors'],
 'regression':['Linear Regression','Ridge Regression','Lasso Regression','CART Regression Tree','Random Forest Regressor'],
 'clustering':['K-Means Clustering']
}
DESCRIPTIONS={
 'Logistic Regression':'A transparent baseline for yes/no outcomes. Predicts probabilities, not guaranteed outcomes.',
 'CART Classification Tree':'Creates human-readable yes/no rules, but deep trees can overfit.',
 'Random Forest Classifier':'Combines many trees; usually stable, but harder to explain than one tree.',
 'Gradient Boosting Classifier':'Builds trees sequentially to correct mistakes; can perform well but needs tuning.',
 'K-Nearest Neighbors':'Uses similar past cases to classify a new case. Sensitive to feature scales.',
 'Linear Regression':'A simple and interpretable starting point for numeric predictions.',
 'Ridge Regression':'Linear model that shrinks coefficients to reduce instability.',
 'Lasso Regression':'Linear model that can shrink weak coefficients to zero.',
 'CART Regression Tree':'Predicts numeric values using simple rule-based splits.',
 'Random Forest Regressor':'Combines many trees to predict numbers and capture nonlinear patterns.',
 'K-Means Clustering':'Groups similar records without any known target. K is the number of groups.'
}

@dataclass
class Result:
    task:str
    model_name:str
    target:str|None
    features:list[str]
    metrics:dict[str,Any]
    prediction_table:pd.DataFrame
    importance:pd.DataFrame = field(default_factory=pd.DataFrame)
    profiles:pd.DataFrame=field(default_factory=pd.DataFrame)
    chart:pd.DataFrame=field(default_factory=pd.DataFrame)
    cv_scores:list[float]=field(default_factory=list)
    training_info:dict=field(default_factory=dict)
    true_labels:Any=None
    proba:Any=None
    class_names:list=field(default_factory=list)
    positive_class:Any=None
    estimator:Any=None
    data_version:int=0
    def comparable_signature(self):
        return (self.task,self.target,tuple(self.features),self.training_info.get('test_size'))


def _prepare_X(X:pd.DataFrame,scale:bool)->ColumnTransformer:
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[x for x in X.columns if x not in numeric]
    components=[]
    if numeric:
        steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
        if scale: steps.append(('scale',StandardScaler()))
        components.append(('numeric',Pipeline(steps),numeric))
    if categorical:
        components.append(('categorical',Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
            ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=30))
        ]),categorical))
    return ColumnTransformer(components,verbose_feature_names_out=False, remainder='drop')


def make_estimator(model:str,params:dict|None=None):
    p=params or {}
    depth=int(p.get('max_depth',5))
    trees=int(p.get('n_estimators',100))
    alpha=float(p.get('alpha',1.0))
    k=int(p.get('neighbors',5))
    mapping={
        'Linear Regression': lambda:LinearRegression(),
        'Ridge Regression':lambda:Ridge(alpha=alpha),
        'Lasso Regression':lambda:Lasso(alpha=alpha,max_iter=12000,random_state=750),
        'CART Regression Tree':lambda:DecisionTreeRegressor(max_depth=depth,min_samples_leaf=4,random_state=750),
        'Random Forest Regressor':lambda:RandomForestRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=3,random_state=750,n_jobs=-1),
        'Logistic Regression':lambda:LogisticRegression(max_iter=2000,random_state=750),
        'CART Classification Tree':lambda:DecisionTreeClassifier(max_depth=depth,min_samples_leaf=4,random_state=750),
        'Random Forest Classifier':lambda:RandomForestClassifier(n_estimators=trees,max_depth=depth,min_samples_leaf=3,random_state=750,n_jobs=-1),
        'Gradient Boosting Classifier':lambda:GradientBoostingClassifier(n_estimators=trees,learning_rate=float(p.get('learning_rate',0.1)),max_depth=min(4,depth),random_state=750),
        'K-Nearest Neighbors':lambda:KNeighborsClassifier(n_neighbors=k),
    }
    if model not in mapping: raise ValueError('Unknown model. Please choose one listed for your task.')
    return mapping[model]()


def model_grid(model:str,params:dict|None=None)->dict:
    p=params or {};depth=int(p.get('max_depth',5));trees=int(p.get('n_estimators',100));alpha=float(p.get('alpha',1.0))
    if model=='Ridge Regression':return {'estimator__alpha':[0.1,1,10]}
    if model=='Lasso Regression':return {'estimator__alpha':[0.01,0.1,1]}
    if model=='Logistic Regression':return {'estimator__C':[0.1,1,10]}
    if model=='K-Nearest Neighbors':return {'estimator__n_neighbors':[3,5,9]}
    if model.startswith('CART '): return {'estimator__max_depth':[3,5,8]}
    if model.startswith('Random Forest '): return {'estimator__max_depth':[4,8,None]}
    if model=='Gradient Boosting Classifier':return {'estimator__learning_rate':[0.05,0.1],'estimator__max_depth':[2,3]}
    return {}


def _valid_features(df:pd.DataFrame, features:list[str],target:str|None,task:str):
    if not isinstance(df,pd.DataFrame) or len(df)<20: raise ValueError('Please upload a dataset with at least 20 rows.')
    if not features: raise ValueError('Select at least one predictor in Step 4.')
    if any(f not in df.columns for f in features): raise ValueError('A selected predictor is missing from the dataset.')
    if target in features: raise ValueError('Remove the target from predictors. Otherwise the model already knows the answer.')
    if task!='clustering' and (not target or target not in df): raise ValueError('Choose a target variable in Step 3.')
    for name in features:
        col=df[name]
        if col.nunique(dropna=True)<2: raise ValueError(f'**{name}** is empty or constant; remove it in Step 4.')
        if potential_identifier(name,col): raise ValueError(f'**{name}** looks like an identifier. Remove it from predictors in Step 4.')
        if col.dtype=='object' and col.nunique(dropna=True)>60: raise ValueError(f'**{name}** has many categories. Remove it, or group categories before uploading.')


def _importance(pipe:Pipeline)->pd.DataFrame:
    transformed=pipe.named_steps['prep'].get_feature_names_out()
    est=pipe.named_steps['estimator']
    if hasattr(est,'feature_importances_'):
        signed=np.asarray(est.feature_importances_)
    elif hasattr(est,'coef_'):
        co=np.asarray(est.coef_)
        signed=co if co.ndim==1 else co.mean(axis=0)
    else: return pd.DataFrame(columns=['Variable','Importance','Coefficient'])
    mag=np.abs(co).mean(axis=0) if hasattr(est,'coef_') and co.ndim>1 else np.abs(np.asarray(signed))
    if len(mag)!=len(transformed): return pd.DataFrame(columns=['Variable','Importance','Coefficient'])
    out=pd.DataFrame({'Variable':transformed,'Importance':mag,'Coefficient':signed})
    return out.sort_values('Importance',ascending=False).head(20)


def _classifier_metrics(y_true,y_pred,probabilities,positive,labels):
    binary=len(labels)==2
    precision=precision_score(y_true,y_pred,pos_label=positive,zero_division=0) if binary else precision_score(y_true,y_pred,average='macro',zero_division=0)
    recall=recall_score(y_true,y_pred,pos_label=positive,zero_division=0) if binary else recall_score(y_true,y_pred,average='macro',zero_division=0)
    f1=f1_score(y_true,y_pred,pos_label=positive,zero_division=0) if binary else f1_score(y_true,y_pred,average='macro',zero_division=0)
    metrics={'Accuracy':float(accuracy_score(y_true,y_pred)), 'Precision':float(precision),'Recall':float(recall),'F1':float(f1)}
    if probabilities is not None:
        try:
            if binary:
                metrics['ROC-AUC']=float(roc_auc_score(np.asarray(y_true)==positive,probabilities))
            else:
                # provided full probability matrix, ordered using the fitted estimator's classes
                metrics['ROC-AUC']=float(roc_auc_score(y_true,probabilities,multi_class='ovr',labels=labels,average='macro'))
        except ValueError: pass
    return metrics


def classification_at_cutoff(result:Result,cutoff:float=.5,cost_fp:float=1.0,cost_fn:float=1.0):
    """Re-evaluate the untouched test probabilities without refitting."""
    if result.task!='classification': raise ValueError('This only applies to classification.')
    labels=result.class_names
    y_true=np.asarray(result.true_labels)
    if len(labels)==2 and result.proba is not None:
        negative=next(c for c in labels if c!=result.positive_class)
        y_pred=np.where(np.asarray(result.proba)>=cutoff,result.positive_class,negative)
        metrics=_classifier_metrics(y_true,y_pred,result.proba,result.positive_class,labels)
        tp=int(np.sum((y_true==result.positive_class)&(y_pred==result.positive_class)))
        fp=int(np.sum((y_true!=result.positive_class)&(y_pred==result.positive_class)))
        fn=int(np.sum((y_true==result.positive_class)&(y_pred!=result.positive_class)))
        tn=int(np.sum((y_true!=result.positive_class)&(y_pred!=result.positive_class)))
        metrics.update({'TP':tp,'FP':fp,'FN':fn,'TN':tn,'Cost / case':float((cost_fp*fp+cost_fn*fn)/len(y_true))})
    else:
        y_pred=result.prediction_table['Predicted'].values
        metrics=_classifier_metrics(y_true,y_pred,result.proba,result.positive_class,labels)
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    return metrics,pd.DataFrame(cm,index=[f'Actual {x}' for x in labels],columns=[f'Predicted {x}' for x in labels]),np.asarray(y_pred)


def run_supervised(df:pd.DataFrame,task:str,model_name:str,target:str,features:list[str],
                   test_size:float=.25,folds:int=5,tune:bool=False,params:dict|None=None,
                   positive_class:Any=None,data_version:int=0)->Result:
    if task not in ('regression','classification') or model_name not in MODELS[task]:
        raise ValueError('Please choose a model compatible with your problem type.')
    _valid_features(df,features,target,task)
    frame=df[features+[target]].dropna(subset=[target]).copy()
    if len(frame)<20:raise ValueError('Not enough observations with known target values.')
    X=frame[features]
    if task=='regression':
        y=pd.to_numeric(frame[target],errors='coerce')
        keep=y.notna();X=X[keep];y=y[keep]
        if len(y)<20 or y.nunique()<3:raise ValueError('Regression needs a numeric target with several different values.')
        stratify=None
    else:
        y=frame[target]
        if y.nunique()<2 or y.nunique()>15: raise ValueError('Classification needs a categorical target with 2–15 classes.')
        if y.value_counts().min()<max(3,folds):raise ValueError(f'Every class needs at least {max(3,folds)} observations for {folds}-fold cross-validation. Reduce folds or choose another target.')
        stratify=y
    try:
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=test_size,stratify=stratify,random_state=750)
    except ValueError as exc: raise ValueError('Cannot create a balanced train/test split. Try more rows or fewer target classes. '+str(exc)) from exc
    if task=='classification' and ytrain.value_counts().min()<folds:
        raise ValueError('Not enough examples of one class in training data. Try fewer CV folds.')
    if model_name=='K-Nearest Neighbors' and int((params or {}).get('neighbors',5))>len(xtrain):
        raise ValueError('Number of neighbors must not exceed training rows.')
    scale=model_name in ('Linear Regression','Ridge Regression','Lasso Regression','Logistic Regression','K-Nearest Neighbors')
    pipe=Pipeline([('prep',_prepare_X(xtrain,scale)),('estimator',make_estimator(model_name,params))])
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=750) if task=='classification' else KFold(n_splits=folds,shuffle=True,random_state=750)
    scoring='f1_weighted' if task=='classification' else 'neg_root_mean_squared_error'
    grid=model_grid(model_name,params) if tune else {}
    if grid:
        search=GridSearchCV(pipe,grid,scoring=scoring,cv=splitter,n_jobs=1,error_score='raise')
        search.fit(xtrain,ytrain)
        fitted=search.best_estimator_
        scores=np.asarray([search.best_score_])
        best_params={k.replace('estimator__',''):v for k,v in search.best_params_.items()}
        best_cv=float(search.best_score_)
    else:
        scores=cross_val_score(pipe,xtrain,ytrain,cv=splitter,scoring=scoring,n_jobs=1,error_score='raise')
        fitted=pipe.fit(xtrain,ytrain)
        best_params={}
        best_cv=float(np.mean(scores))
    pred=fitted.predict(xtest)
    info={'test_size':test_size,'train_rows':len(xtrain),'test_rows':len(xtest),'cv_folds':folds,'tuned':bool(grid),'best_parameters':best_params,'cv_mean':best_cv,'seed':750}
    importance=_importance(fitted)
    if task=='regression':
        metrics={'RMSE':float(np.sqrt(mean_squared_error(ytest,pred))), 'MAE':float(mean_absolute_error(ytest,pred)),
                 'R²':float(r2_score(ytest,pred)), 'CV RMSE':float(-best_cv)}
        baseline=float(np.sqrt(mean_squared_error(ytest,np.full(len(ytest),np.mean(ytrain)))))
        metrics['Baseline RMSE']=baseline
        table=pd.DataFrame({'Actual':ytest.values,'Predicted':pred},index=ytest.index)
        return Result(task,model_name,target,features,metrics,table,importance,chart=table.reset_index(drop=True),cv_scores=list(scores),training_info=info,true_labels=np.asarray(ytest),estimator=fitted,data_version=data_version)
    labels=list(fitted.named_steps['estimator'].classes_)
    if positive_class not in labels:
        positive_class=(1 if 1 in labels else 'yes' if 'yes' in labels else labels[-1])
    proba=fitted.predict_proba(xtest) if hasattr(fitted,'predict_proba') else None
    binary_prob=proba[:,labels.index(positive_class)] if proba is not None and len(labels)==2 else None
    actual_probs=binary_prob if len(labels)==2 else proba
    table=pd.DataFrame({'Actual':ytest.values,'Predicted':pred},index=ytest.index)
    if binary_prob is not None: table['Positive probability']=binary_prob
    metrics=_classifier_metrics(ytest,pred,actual_probs,positive_class,labels)
    metrics['CV F1 (weighted)']=best_cv
    metrics['Majority-class baseline']=float(ytest.value_counts(normalize=True).max())
    return Result(task,model_name,target,features,metrics,table,importance,cv_scores=list(scores),training_info=info,true_labels=np.asarray(ytest),proba=actual_probs,class_names=labels,positive_class=positive_class,estimator=fitted,data_version=data_version)


def compare_k(df:pd.DataFrame,features:list[str],maximum:int=8,seed:int=750)->pd.DataFrame:
    _valid_features(df,features,None,'clustering')
    numeric=df[features].select_dtypes(include=np.number)
    if numeric.shape[1]<2: raise ValueError('For an informative K-Means plot, please select at least two numeric variables.')
    x=_prepare_X(numeric,True).fit_transform(numeric)
    if np.unique(x,axis=0).shape[0]<3: raise ValueError('Not enough different patterns to form meaningful groups.')
    cap=min(maximum,len(x)-1,len(np.unique(x,axis=0))-1)
    rows=[]
    for k in range(2,cap+1):
        km=KMeans(n_clusters=k,n_init=10,random_state=seed)
        labels=km.fit_predict(x)
        score=(float(silhouette_score(x,labels,sample_size=min(1000,len(x)),random_state=seed)) if 1<len(np.unique(labels))<len(x) else np.nan)
        rows.append({'K':k,'Silhouette':score,'Within-cluster SS':float(km.inertia_)})
    return pd.DataFrame(rows)


def run_clustering(df:pd.DataFrame,features:list[str],n_clusters:int=3,data_version:int=0)->Result:
    _valid_features(df,features,None,'clustering')
    X=df[features].copy()
    nonnumeric=X.select_dtypes(exclude=np.number).columns.tolist()
    if nonnumeric: raise ValueError('K-Means needs numeric measurements. Remove categorical features: '+', '.join(nonnumeric))
    if n_clusters<2 or n_clusters>=len(df): raise ValueError('Select at least 2 groups, but fewer groups than rows.')
    transformer=_prepare_X(X,True)
    scaled=transformer.fit_transform(X)
    km=KMeans(n_clusters=n_clusters,n_init=10,random_state=750)
    labels=km.fit_predict(scaled)
    if len(np.unique(labels))<2: raise ValueError('K-Means found only one distinct group. Try another set of variables.')
    silhouette=float(silhouette_score(scaled,labels,sample_size=min(1500,len(df)),random_state=750))
    pc=PCA(n_components=2).fit_transform(scaled) if scaled.shape[1]>=2 else np.column_stack([scaled[:,0],np.zeros(len(df))])
    chart=pd.DataFrame({'PC1':pc[:,0],'PC2':pc[:,1],'Cluster':[f'Cluster {i+1}' for i in labels]})
    pred=df[features].copy();pred['Cluster']=labels+1
    means=pred.groupby('Cluster')[features].mean(numeric_only=True).round(3)
    means.insert(0,'Records',pred.groupby('Cluster').size())
    profiles=means.reset_index()
    counts={f'Cluster {k+1}':int(np.sum(labels==k)) for k in range(n_clusters)}
    return Result('clustering','K-Means Clustering',None,features,
                  {'Silhouette':silhouette,'Within-cluster SS':float(km.inertia_),'Groups':n_clusters,'Group sizes':counts},
                  pred,profiles=profiles,chart=chart,training_info={'seed':750,'test_size':None,'method':'unsupervised - no target labels'},estimator=Pipeline([('prep',transformer),('estimator',km)]),data_version=data_version)
