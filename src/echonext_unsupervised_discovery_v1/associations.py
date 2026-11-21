"""Post-freeze exploratory validation associations; aggregate exports only."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
from .data import verify_freeze, sha

def association(labels,y,seed=62001,bootstraps=2000):
    if len(labels)!=len(y) or not np.isin(y,[0,1]).all(): raise ValueError('Invalid binary evaluation population')
    rng=np.random.default_rng(seed); rows=[]
    for k in np.unique(labels):
        a=labels==k; n=int(a.sum()); other=~a
        if n<10 or other.sum()<10 or min(int(y[a].sum()),int((1-y[a]).sum()),int(y[other].sum()),int((1-y[other]).sum()))<10:
            rows.append({'cluster':int(k),'suppressed':True}); continue
        pos=int(y[a].sum()); neg=int(y[other].sum()); rd=float(y[a].mean()-y[other].mean()); diffs=[]
        for _ in range(bootstraps):
            i=rng.integers(0,len(y),len(y)); b=a[i]
            if b.any() and (~b).any(): diffs.append(float(y[i][b].mean()-y[i][~b].mean()))
        _,p=fisher_exact([[pos,n-pos],[neg,int(other.sum())-neg]])
        low,high=np.quantile(diffs,[.025,.975]); rows.append({'cluster':int(k),'patients':n,'events':pos if min(pos,n-pos)>=10 else 'suppressed','prevalence':float(y[a].mean()) if min(pos,n-pos)>=10 else 'suppressed','risk_difference_vs_rest':rd,'ci_low':float(low),'ci_high':float(high),'p_value':float(p),'suppressed':False})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    manifest=verify_freeze(a.run)
    if sha(a.metadata)!=manifest['details']['metadata_sha256']: raise ValueError('Metadata provenance changed')
    # This is the first and only outcome-access step; never access test targets.
    outcomes=[]
    with a.metadata.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split']=='val': outcomes.append(float(row['shd_moderate_or_greater_flag']))
    results=[]
    for path in sorted(a.run.glob('*.assignments.npy')):
        if str(path.resolve()) not in manifest['files']: raise ValueError('Unfrozen assignments')
        for result in association(np.load(path,allow_pickle=False),np.array(outcomes)):
            results.append({'analysis':path.name.replace('.assignments.npy',''),**result})
    # BH correction across all non-suppressed clusters / representations / seeds.
    valid=[i for i,r in enumerate(results) if 'p_value' in r]; order=sorted(valid,key=lambda i:results[i]['p_value']); running=1.
    for rank,i in reversed(list(enumerate(order,1))):
        running=min(running,results[i]['p_value']*len(order)/rank); results[i]['q_value_BH']=running
    a.report.mkdir(parents=True,exist_ok=False)
    (a.report/'aggregate_associations.json').write_text(json.dumps({'exploratory':True,'test_sealed':True,'freeze_sha256':sha(a.run/'FREEZE.json'),'results':results},indent=2))
    print('Aggregate exploratory associations saved; patient assignments remain private.')
if __name__=='__main__': main()
