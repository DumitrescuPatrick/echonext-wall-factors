"""Optional fixed logistic probe AFTER discovery freeze; explicitly supervised."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from .data import verify_freeze,sha,linkage,roles

def scores(y,p): return {'AUROC':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--metadata',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
    m=verify_freeze(a.run)
    if sha(a.metadata)!=m['details']['metadata_sha256']:raise ValueError('Metadata changed')
    _,_,representatives=roles(linkage(a.metadata)['train']); train=[];val=[]
    with a.metadata.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split']=='train':train.append(float(r['shd_moderate_or_greater_flag']))
            elif r['split']=='val':val.append(float(r['shd_moderate_or_greater_flag']))
    y=np.array(train)[representatives]; v=np.array(val); results=[]
    for path in sorted(a.run.glob('*62001.npz')):
        if str(path.resolve()) not in m['files']: raise ValueError('Unfrozen embedding')
        z=np.load(path,allow_pickle=False); model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=62001));model.fit(z['fit'],y);p=model.predict_proba(z['val'])[:,1]
        result={'representation':path.stem,**scores(v,p)}; rng=np.random.default_rng(62001);boot=[]
        for _ in range(2000):
            i=rng.integers(0,len(v),len(v));boot.append(list(scores(v[i],p[i]).values()))
        ci=np.quantile(boot,[.025,.975],axis=0)
        result['confidence_intervals']={name:[float(ci[0,j]),float(ci[1,j])] for j,name in enumerate(scores(v,p))}; results.append(result)
    a.report.write_text(json.dumps({'supervised_after_freeze':True,'exploratory_validation':True,'test_sealed':True,'results':results},indent=2));print('Aggregate fixed logistic probe results saved.')
if __name__=='__main__':main()
