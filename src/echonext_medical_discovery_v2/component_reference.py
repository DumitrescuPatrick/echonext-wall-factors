"""Train-only nonlinear component references; exploratory conditional factor sensitivity."""
import csv,json
from pathlib import Path
import numpy as np,joblib
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
from echonext_unsupervised_discovery_v1.questions_common import load_data,matrix,preprocessing
from echonext_unsupervised_discovery_v1.questions_associations import design,estimate
from echonext_unsupervised_discovery_v1.questions_interpretation import correlations_from_logspd
from echonext_unsupervised_discovery_v1.data import verify_freeze,freeze
from .latent import PRIVATE,REPORT,BASE,CACHE
from .associate import BH,independent_columns,number

CONFIG=Path('config/medical_discovery_next_phase/component_reference.json')

def main():
    verify_freeze(PRIVATE);folder=PRIVATE/'component_reference'
    if folder.exists():raise FileExistsError('Preserve reference control')
    folder.mkdir(mode=0o700);data=load_data(validation_targets=False);inputs={r:matrix(data[r],'clinical_ecg_simple_spd') for r in ['fit','monitor','val']};predictions={};files=[]
    # Replace cached log-SPD columns with equal-information ordinary coordinates.
    from echonext_unsupervised_discovery_v1.questions_euclidean import euclidean
    for r in inputs:inputs[r]=np.column_stack([data[r]['clinical'],data[r]['waveform'],euclidean(data[r]['spd'])])
    for endpoint in ['lvef_lte_45_flag','lvwt_gte_13_flag']:
        y=data['fit']['targets'][endpoint];ym=data['monitor']['targets'][endpoint];model=Pipeline([('preprocessing',preprocessing('clinical_ecg_simple_spd')),('model',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=69001))]).fit(inputs['fit'],y);p=model.predict_proba(inputs['monitor'])[:,1];cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],ym);p=model.predict_proba(inputs['val'])[:,1];predictions[endpoint]=cal.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1];path=folder/f'{endpoint}.joblib';joblib.dump((model,cal),path);files.append(path)
    cache=folder/'validation_risks.npz';np.savez(cache,**predictions);files.append(cache);freeze(folder,files+[CONFIG,Path(__file__),PRIVATE/'FREEZE.json'],{'reference_fitting':'training patients only; monitor-only sigmoid','validation_outcomes_used_for_reference_fitting':False})
    rows=[];names=['age_at_ecg','sex','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year','location_setting','race_ethnicity','lvef_lte_45_flag','lvwt_gte_13_flag']
    with (CACHE/'echonext_metadata_100k.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split']=='val':rows.append({n:r.get(n,'') for n in names})
    simple=np.load(BASE/'waveform_62001.npz',allow_pickle=False)['val'];energy=np.sqrt(np.mean(simple[:,12:24].astype(float)**2,axis=1));base,_,_=design(rows,'flexible_plus_energy',energy);corr=correlations_from_logspd(np.load(BASE/'spd_62001.npz',allow_pickle=False)['val'])[:,0,2];proxy=StandardScaler().fit_transform(SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False).fit_transform(corr[:,None]));risk=np.load(PRIVATE/'frozen_embeddings.npz',allow_pickle=False)['val_logit'];factors=np.load(PRIVATE/'factor_scores.npz',allow_pickle=False);result=[];performance=[]
    for endpoint,p in predictions.items():
        y=np.array([number(r[endpoint]) for r in rows]);cov=independent_columns(np.column_stack([base,proxy,risk,logit(np.clip(p,1e-6,1-1e-6))]));performance.append({'endpoint':endpoint,'validation_AUROC':float(roc_auc_score(y,p)),'validation_Brier':float(brier_score_loss(y,p))})
        for space in ['pooled','residual_pooled','temporal']:
            for j in range(8):
                record={'endpoint':endpoint,'space':space,'factor':j+1}
                try:record.update(estimate(y,factors[f'{space}_val'][:,j],cov));record['status']='estimated'
                except (ValueError,RuntimeError,np.linalg.LinAlgError):record['status']='not_estimable'
                result.append(record)
    BH(result,48);(REPORT/'COMPONENT_REFERENCE_SENSITIVITY.json').write_text(json.dumps({'status':'Additional exploratory conditional association sensitivity, not independent predictive value','reference_models':performance,'factor_associations_BH48':result,'limitations':'A fitted risk score is not conditioning on the full underlying distribution; source encoder selection bias and unmeasured axis remain'},indent=2));print('Component-specific conventional/covariance reference sensitivity saved.')
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
