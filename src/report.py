"""Create a downloadable report from computed values."""
from __future__ import annotations
from datetime import date

def report_markdown(result,goal:str=''):
    lines=['# FinSight Analysis Report',f'Date: {date.today().isoformat()}',f'Business question: {goal or "Not supplied"}',
           f'Model: {result.model_name}',f'Task: {result.task}',f'Target: {result.target or "None – clustering"}',
           'Predictors: '+', '.join(result.features),'','## Measured results']
    for k,v in result.metrics.items():lines.append(f'- {k}: {v}')
    lines+=['','## Model setup']
    for k,v in result.training_info.items():lines.append(f'- {k}: {v}')
    if not result.importance.empty:
        lines+=['','## Leading features (importance does NOT prove causation)']
        for _,row in result.importance.head(10).iterrows():lines.append(f'- {row["Variable"]}: {row["Importance"]:.4f}')
    if not result.profiles.empty:
        lines+=['','## Group profiles', result.profiles.to_markdown(index=False)]
    lines+=['','## Limitations',
       '- This application is an educational analysis toolbox, not validated production credit-risk software.',
       '- Model performance is an estimate from historical data; future performance can differ.',
       '- Associations do not establish causal relationships.',
       '- Credit and lending decisions require independent validation, fairness checks, governance, and human review.']
    return '\n'.join(lines)+'\n'
