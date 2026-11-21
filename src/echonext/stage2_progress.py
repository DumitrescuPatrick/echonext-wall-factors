"""Aggregate-only progress display; never opens clinical predictions or metadata."""
import json
from pathlib import Path

out=Path('reports/stage2_v1');active=[]
for p in sorted(out.glob('*_history.json')):
    if p.with_name(p.name.replace('_history','_complete')).exists():continue
    h=json.loads(p.read_text())
    active.append({'run':p.stem.replace('_history',''),'epoch':len(h),'best_validation_auc':round(max(r['validation_auc'] for r in h),5),'patience_count':h[-1]['bad_epochs']})
print(json.dumps({'tuning_complete':len(list(out.glob('tune_*_complete.json'))),'final_seeds_complete':len(list(out.glob('seed_*_complete.json'))),'active':active,'stage2_complete':(out/'STAGE2_COMPLETE.json').exists()}))
