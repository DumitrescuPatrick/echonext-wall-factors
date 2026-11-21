"""Locked train/monitor fits and paired development-only validation comparisons."""
import csv,json,platform
from pathlib import Path
import numpy as np,joblib,torch
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
from echonext_unsupervised_discovery_v1.questions_common import load_data,number
from echonext_unsupervised_discovery_v1.questions_euclidean import euclidean
from echonext_unsupervised_discovery_v1.data import freeze,verify_freeze,sha
from echonext_medical_discovery_v2.latent import PRIVATE as SOURCE

ROOT=Path('private/wall_factor_validation_v1')
OUT=Path('reports/wall_factor_validation_v1')
CONFIG=Path('config/wall_factor_validation_v1/protocol.json')
TARGET='lvwt_gte_13_flag'

def metrics(y,p):
    d=p>=.2;positive=y==1;negative=~positive
    return dict(AUROC=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),sensitivity=float(d[positive].mean()),specificity=float((~d[negative]).mean()),net_benefit_020=float(np.mean(d&positive)-.25*np.mean(d&negative)),mean_risk=float(p.mean()),prevalence=float(y.mean()))

def paired(y,a,b,reps=1000,seed=71001):
    rng=np.random.default_rng(seed);point={k:metrics(y,b)[k]-v for k,v in metrics(y,a).items()};samples={k:[] for k in point}
    for _ in range(reps):
        ix=rng.integers(len(y),size=len(y));z=y[ix]
        if np.unique(z).size<2:continue
        ma=metrics(z,a[ix]);mb=metrics(z,b[ix])
        for k in samples:samples[k].append(mb[k]-ma[k])
    return {k:{'difference':v,'paired_95_CI':np.quantile(samples[k],[.025,.975]).tolist()} for k,v in point.items()}

def calibration(y,p):
    z=logit(np.clip(p,1e-6,1-1e-6));m=LogisticRegression(C=1e6,max_iter=2000).fit(z[:,None],y)
    bins=[]
    for ix in np.array_split(np.argsort(p),10):bins.append({'patients':len(ix),'mean_risk':float(p[ix].mean()),'event_fraction':float(y[ix].mean())})
    return {'intercept':float(m.intercept_[0]),'slope':float(m.coef_[0,0]),'descriptive_deciles':bins}

def subgroups(rows,fitrows):
    age=np.array([number(r['age_at_ecg']) for r in rows]);qrs=np.array([number(r['qrs_duration']) for r in rows]);hr=np.array([number(r['ventricular_rate']) for r in rows]);year=np.array([number(r['acquisition_year']) for r in rows]);cut=float(np.nanmedian([number(r['acquisition_year']) for r in fitrows]));out={}
    for field in ['sex','location_setting']:
        a=np.array([r[field] or 'missing' for r in rows])
        for value in sorted(set(a)):out[field+':'+value]=a==value
    out.update({'age:<65':age<65,'age:65+':age>=65,'QRS:<120':qrs<120,'QRS:120+':qrs>=120,'rate:<60':hr<60,'rate:60-100':(hr>=60)&(hr<=100),'rate:>100':hr>100,'year:earlier':year<=cut,'year:later':year>cut})
    return out,cut

def main():
    if ROOT.exists():raise FileExistsError('Frozen follow-up already exists; preserve outputs')
    verify_freeze(SOURCE);verify_freeze(SOURCE/'component_reference')
    ROOT.mkdir(mode=0o700);OUT.mkdir(parents=True,exist_ok=True)
    sources=[SOURCE/'FREEZE.json',SOURCE/'pooled_PCA.joblib',SOURCE/'temporal_PCA.joblib',SOURCE/'factor_scores.npz',SOURCE/'frozen_embeddings.npz',Path('private/stage2_v1/tune_3/best.pt'),Path('src/echonext_medical_discovery_v2/latent.py'),CONFIG,Path(__file__)]
    freeze(ROOT,sources,{'purpose':'Prospective lock of follow-up, after exploratory discovery','factors':'pooled column1; temporal column4; native scale and signs unchanged','test_accessed':False})
    # Fit lock is written before new endpoint comparisons. No new PCA or encoder fit.
    data=load_data(validation_targets=True);scores=np.load(SOURCE/'factor_scores.npz',allow_pickle=False);raw=np.load(SOURCE/'frozen_embeddings.npz',allow_pickle=False);inputs={};numeric_columns=None
    for role in ['fit','monitor','val']:
        r=data[role];clinical=r['clinical'];years=np.array([number(v['acquisition_year']) for v in r['records']])[:,None]
        numerical=np.column_stack([clinical,r['waveform'],euclidean(r['spd']),years,raw[role+'_logit']]);numeric_columns=numerical.shape[1]
        cat=np.array([[v[k] or 'missing' for k in ['location_setting','race_ethnicity']] for v in r['records']]);inputs[role]=np.column_stack([numerical,cat])
    y={r:data[r]['targets'][TARGET].astype(int) for r in inputs};predictions={};result=[];models=[]
    for family in ['logistic','HGB']:
        for addition,cols in [('reference',[]),('pooled2',[0]),('temporal5',[1]),('both',[0,1])]:
            xs={}
            for r in inputs:
                f=np.column_stack([scores['pooled_'+r][:,1],scores['temporal_'+r][:,4]])[:,cols];xs[r]=np.column_stack([inputs[r][:,:numeric_columns].astype(float),f,inputs[r][:,numeric_columns:]])
            n=numeric_columns+len(cols);pre=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler()),list(range(n))),('context',OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False),[n,n+1])]);est=LogisticRegression(C=1,max_iter=4000,random_state=71001) if family=='logistic' else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=71001);model=make_pipeline(pre,est).fit(xs['fit'],y['fit']);mp=model.predict_proba(xs['monitor'])[:,1];cal=LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(mp,1e-6,1-1e-6))[:,None],y['monitor']);p=cal.predict_proba(logit(np.clip(model.predict_proba(xs['val'])[:,1],1e-6,1-1e-6))[:,None])[:,1];name=family+'_'+addition;predictions[name]=p
            joblib.dump((model,cal),ROOT/(name+'.joblib'));models.append(ROOT/(name+'.joblib'));result.append({'model':name,'patients':len(p),'events':int(y['val'].sum()),**metrics(y['val'],p),'calibration':calibration(y['val'],p)});print('Saved follow-up model '+name,flush=True)
    np.savez(ROOT/'validation_predictions.npz',**predictions)
    comparisons=[]
    for family in ['logistic','HGB']:
        for addition in ['pooled2','temporal5','both']:
            comparisons.append({'family':family,'addition':addition,**paired(y['val'],predictions[family+'_reference'],predictions[family+'_'+addition])})
    groups,year_cut=subgroups(data['val']['records'],data['fit']['records']);sub=[];factor=np.column_stack([scores['pooled_val'][:,1],scores['temporal_val'][:,4]])
    for name,ix in groups.items():
        yy=y['val'][ix];n=len(yy);e=int(yy.sum())
        if n<100 or min(e,n-e)<20:sub.append({'subgroup':name,'status':'suppressed: insufficient sample/events'});continue
        sub.append({'subgroup':name,'patients':n,'events':e,'factor_means':factor[ix].mean(0).tolist(),'factor_SD':factor[ix].std(0).tolist(),'factor_correlation':float(np.corrcoef(factor[ix].T)[0,1]),'HGB_both_increment':paired(yy,predictions['HGB_reference'][ix],predictions['HGB_both'][ix],reps=300),'status':'exploratory'})
    device={'MPS_built':torch.backends.mps.is_built(),'MPS_available':torch.backends.mps.is_available(),'torch':torch.__version__,'clinical_SSL_training_performed':False,'existing_SSL_complete_files':len(list(Path('private/medical_discovery_next_phase_unsupervised').glob('*/COMPLETE.json'))),'reason':'Separate VAE/SimCLR analysis requires completed training-only checkpoints; no surrogate results substituted'}
    freeze(ROOT/'outputs',models+[ROOT/'validation_predictions.npz'],{'private_only':True})
    report={'endpoint':TARGET,'status':'Exploratory development evaluation, not independent validation','population':{r:{'patients':len(v),'events':int(v.sum())} for r,v in y.items()},'metrics':result,'paired_comparisons':comparisons,'subgroups':sub,'fitting_median_year':year_cut,'unsupervised':device,'test_accessed':False,'bootstrap_scope':'Fixed encoder, selected factors and fitted models; does not capture selection bias or neural training variability','software':{'numpy':np.__version__,'platform':platform.platform()}}
    (OUT/'RESULTS.json').write_text(json.dumps(report,indent=2))
    with (OUT/'MODEL_PERFORMANCE.csv').open('w',newline='') as f:
        fields=['model','patients','events',*metrics(y['val'],predictions['HGB_reference']).keys()];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result)
    with (OUT/'PAIRED_COMPARISONS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['family','added_factors','metric','difference','lower_95','upper_95'])
        for row in comparisons:
            for metric,value in row.items():
                if isinstance(value,dict):w.writerow([row['family'],row['addition'],metric,value['difference'],*value['paired_95_CI']])
    print('Aggregate development follow-up saved; official test untouched.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
