import io
import pandas as pd
import pytest
from src.dataio import read_dataset,sheet_names,predict_task_from_goal,recommended_features,feature_warnings

def test_csv_semicolon():
    raw=b'age;income;default\n24;25000;0\n36;54000;1\n'
    df=read_dataset(raw,'bank.csv')
    assert df.shape==(2,3)
    assert df['income'].iloc[1]==54000

def test_xlsx_sheets():
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as writer:
        pd.DataFrame({'age':[21,22],'default':[0,1]}).to_excel(writer,sheet_name='Customers',index=False)
        pd.DataFrame({'price':[4,5]}).to_excel(writer,sheet_name='Properties',index=False)
    content=out.getvalue()
    assert sheet_names(content,'bank.xlsx')==['Customers','Properties']
    assert read_dataset(content,'bank.xlsx','Customers').shape==(2,2)

def test_invalid_extension():
    with pytest.raises(ValueError,match='upload'):
        read_dataset(b'anything','data.exe')

def test_goal_and_feature_warnings():
    df=pd.DataFrame({'CustomerID':['A','B','C'],'income':[1,2,3],'default':[0,0,1]})
    assert predict_task_from_goal('Which customer might default?')=='classification'
    assert predict_task_from_goal('Group similar customers')=='clustering'
    assert predict_task_from_goal('Predict a house price')=='regression'
    assert 'CustomerID' not in recommended_features(df,'default','classification')
    assert feature_warnings(df,['CustomerID'],'default')
