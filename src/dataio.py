"""File reading and lightweight data profiling; no remote uploads."""
from __future__ import annotations
import io
import re
import pandas as pd
import numpy as np

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
SUPPORTED = ('csv','xlsx','xls')


def sheet_names(data: bytes, filename: str) -> list[str]:
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext not in ('xlsx', 'xls'):
        return []
    return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)


def read_dataset(data: bytes, filename: str, sheet: str | int = 0) -> pd.DataFrame:
    if not data:
        raise ValueError('The selected file is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('File is larger than 30 MB. Please reduce the size before uploading.')
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'csv':
        # Automatic delimiter sniffing; handles many European CSV exports.
        df = None
        last_error = None
        for encoding in ['utf-8-sig', 'utf-8', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine='python', encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
        if df is None:
            raise ValueError(f'Could not read this CSV: {last_error}')
    elif ext in ('xlsx','xls'):
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    else:
        raise ValueError('Please upload a .csv, .xlsx, or .xls file.')
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('This sheet does not contain a usable table.')
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) != len(set(df.columns)):
        raise ValueError('Column names must be unique. Please rename duplicate columns.')
    if df.shape[1] > 200:
        raise ValueError('This prototype supports up to 200 columns; please choose relevant columns first.')
    if df.shape[0] > 100000:
        raise ValueError('This prototype supports up to 100,000 rows; please use a smaller sample.')
    df = df.replace([np.inf,-np.inf],np.nan)
    # Date-like columns stay as their original values until the analyst decides how to use them.
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'Variable': df.columns,
        'Type': df.dtypes.astype(str).values,
        'Missing': df.isna().sum().values.astype(int),
        'Missing %': (df.isna().mean().values*100).round(1),
        'Unique values': df.nunique(dropna=True).values.astype(int),
    })


def potential_identifier(name: str, s: pd.Series) -> bool:
    key = re.sub(r'[^a-z0-9]+','',str(name).lower())
    if re.search(r'(customerid|accountid|userid|recordid|transactionid|^id$|^index$|serial|rownumber|^unnamed)', key):
        return True
    if len(s) > 20 and s.nunique(dropna=True) / max(1,s.notna().sum()) > 0.96:
        # High-uniqueness strings are often customer identifiers; avoid flagging continuous numbers.
        return not pd.api.types.is_numeric_dtype(s)
    return False


def recommended_features(df:pd.DataFrame,target:str|None=None,task:str='classification') -> list[str]:
    suggestions=[]
    for col in df.columns:
        if col == target or potential_identifier(col,df[col]):
            continue
        if df[col].isna().mean()>0.65 or df[col].nunique(dropna=True)<2:
            continue
        if df[col].dtype == 'object' and df[col].nunique(dropna=True)>30:
            continue
        if task=='clustering' and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        suggestions.append(col)
    return suggestions[:12]


def predict_task_from_goal(goal:str) -> str|None:
    g=(goal or '').lower()
    if any(w in g for w in ['segment','cluster','group similar','types of customer','group customer']):
        return 'clustering'
    if any(w in g for w in ['default','churn','leave','yes or no','fraud','subscribe','classify','approval','will they','whether','risk category']):
        return 'classification'
    if any(w in g for w in ['price','sales','spend','revenue','amount','value','profit','income','predict how much','forecast']):
        return 'regression'
    return None


def target_suggestions(df:pd.DataFrame,task:str,goal:str='')->list[str]:
    if task=='clustering':
        return []
    cols=[]
    q=(goal or '').lower()
    keywords=['default','churn','fraud','subscribe','target','y','label','outcome'] if task=='classification' else ['price','value','sales','spend','revenue','profit','amount','income','y','target']
    for c in df.columns:
        s=df[c].dropna(); unique=s.nunique()
        if len(s)<15 or unique<2: continue
        valid=(unique<=20 if task=='classification' else pd.api.types.is_numeric_dtype(s) and unique>10)
        if valid:
            priority=2 if any(x in c.lower() for x in keywords) else 0
            priority+=2 if c.lower() in q else 0
            cols.append((priority,c))
    return [c for _,c in sorted(cols,key=lambda pair:(-pair[0],pair[1]))]


def feature_warnings(df:pd.DataFrame,features:list[str],target:str|None=None)->list[str]:
    notes=[]
    if target and target in features: notes.append('The outcome (target) cannot also be a predictor. That would reveal the answer.')
    for f in features:
        if f not in df.columns: continue
        series=df[f]
        if potential_identifier(f,series): notes.append(f'**{f}** looks like an ID. IDs usually should not be used to predict outcomes.')
        if series.isna().mean()>0.4: notes.append(f'**{f}** has {series.isna().mean():.0%} missing data. Check whether filling it makes sense.')
        if series.nunique(dropna=True)<2: notes.append(f'**{f}** does not vary; it cannot help a model distinguish observations.')
        if series.dtype=='object' and series.nunique(dropna=True)>30: notes.append(f'**{f}** has many categories. It may be text or an identifier; remove or simplify it.')
        if re.search(r'(?i)(gender|sex|race|ethnic|religion|disab|nationalit|age)',f):
            notes.append(f'**{f}** may involve a sensitive or regulated characteristic. Check fairness, policy and legal requirements; do not use this app for automatic lending decisions.')
        if target and re.search(r'(?i)(after|post|future|next_month|approved|default_paid)',f):
            notes.append(f'**{f}** may contain information only available after the outcome. Check for target leakage.')
    return list(dict.fromkeys(notes))
