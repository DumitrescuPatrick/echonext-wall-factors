"""Equal-dimension ordinary covariance control for log-SPD prediction features."""
import json,time
from pathlib import Path
import numpy as np
import joblib
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits
from .questions_common import load_data,preprocessing,ROOT,META
from .questions_prediction import PRIVATE as BASE,score
from .data import freeze,verify_freeze,sha

PRIVATE=Path('private/unsupervised_discovery_v1/five_questions_euclidean_v1')
REPORT=Path('reports/unsupervised_discovery_v1/five_questions_v1')
CONFIG=Path('config/unsupervised_discovery_v1/five_questions_v1/euclidean_control.json')

def euclidean(features):
    i,j=np.triu_indices(12);log=np.zeros((len(features),12,12));log[:,i,j]=features[:,:78]/np.where(i==j,1,np.sqrt(2));log[:,j,i]=log[:,i,j];e,u=np.linalg.eigh(log);s=(u*np.exp(e)[:,None,:])@u.transpose(0,2,1)-np.eye(12);return np.column_stack([s[:,i,j]*np.where(i==j,1,np.sqrt(2)),features[:,78]])

def main():
    verify_freeze(BASE)
    if PRIVATE.exists():raise FileExistsError('Preserve control run')
    PRIVATE.mkdir(parents=True,mode=0o700);data=load_data(validation_targets=False);y=data['fit']['targets']['shd_moderate_or_greater_flag'];ym=data['monitor']['targets']['shd_moderate_or_greater_flag'];fold=np.load(BASE/'training_folds.npz',allow_pickle=False)['fold'];splits=[(np.flatnonzero(fold!=j),np.flatnonzero(fold==j)) for j in range(5)];matrices={role:np.column_stack([data[role]['clinical'],data[role]['waveform'],euclidean(data[role]['spd'])]) for role in ['fit','monitor','val']};artifacts=[BASE/'FREEZE.json',CONFIG,Path(__file__)];summary=[]
    for family in ['logistic','boosting']:
        start=time.perf_counter()
        if family=='logistic':model=LogisticRegression(max_iter=3000,random_state=64001);grid={'model__C':[.01,.1,1.,10.]}
        else:model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,l2_regularization=1.,early_stopping=False,random_state=64001);grid={'model__max_leaf_nodes':[7,15]}
        search=GridSearchCV(Pipeline([('preprocessing',preprocessing('clinical_ecg_simple_spd')),('model',model)]),grid,cv=splits,scoring='roc_auc',n_jobs=1,error_score='raise').fit(matrices['fit'],y)
        pm=search.predict_proba(matrices['monitor'])[:,1];pv=search.predict_proba(matrices['val'])[:,1];cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(pm,1e-6,1-1e-6))[:,None],ym)
        if cal.coef_[0,0]<=0:raise RuntimeError('Nonpositive calibration slope')
        pred=cal.predict_proba(logit(np.clip(pv,1e-6,1-1e-6))[:,None])[:,1];modelpath=PRIVATE/f'{family}.joblib';predpath=PRIVATE/f'{family}.npz';joblib.dump({'model':search.best_estimator_,'calibrator':cal},modelpath);np.savez(predpath,calibrated_val=pred);artifacts.extend([modelpath,predpath]);summary.append({'family':family,'best_parameters':search.best_params_,'CV_AUROC':float(search.best_score_),'seconds':time.perf_counter()-start,'input_features':matrices['fit'].shape[1]})
    freeze(PRIVATE,artifacts,{'metadata_sha256':sha(META),'folds_sha256':sha(BASE/'training_folds.npz'),'outcome_access':'No validation outcome decoded during fitting','definition':'Euclidean coordinates of same regularized covariance; same 79 dimensions'})
    verify_freeze(PRIVATE);y=load_data(validation_targets=True)['val']['targets']['shd_moderate_or_greater_flag'];results=[]
    for family in ['logistic','boosting']:
        control=np.load(PRIVATE/f'{family}.npz',allow_pickle=False)['calibrated_val'];logspd=np.load(BASE/f'{family}__clinical_ecg_simple_spd.npz',allow_pickle=False)['calibrated_val'];rng=np.random.default_rng(64002);draws=[]
        for _ in range(2000):
            i=rng.integers(0,len(y),len(y));draws.append(score(y[i],logspd[i])-score(y[i],control[i]))
        draws=np.array(draws);point=score(y,logspd)-score(y,control);results.append({'family':family,'Euclidean_AUROC':float(score(y,control)[0]),'logSPD_AUROC':float(score(y,logspd)[0]),'logSPD_minus_Euclidean':{name:{'difference':float(point[k]),'paired_95_ci':[float(v) for v in np.quantile(draws[:,k],[.025,.975])]} for k,name in enumerate(['AUROC','average_precision','Brier'])}})
    (REPORT/'EUCLIDEAN_CONTROL.json').write_text(json.dumps({'exploratory_added_control':True,'patients':len(y),'same_information_and_dimension':True,'training':summary,'results':results,'interpretation':'Controls covariance information versus matrix-log coordinate choice; not an intrinsic proof of recovered physiological manifold'},indent=2));print('Matched Euclidean control scored, aggregate only.')
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
