"""Matched train-only tuned clinical models, frozen before validation scoring."""
import argparse,json,time
from pathlib import Path
import numpy as np
import joblib
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
from echonext.stage2_metrics import threshold_for,classification,metrics
from .questions_common import load_data,matrix,preprocessing,PROTOCOL,META,ROOT
from .data import freeze,verify_freeze,sha

PRIVATE=Path('private/unsupervised_discovery_v1/five_questions_v1')
REPORT=Path('reports/unsupervised_discovery_v1/five_questions_v1')
FAMILIES=['logistic','boosting']
FEATURES=['conventional_ecg','clinical_ecg','clinical_ecg_simple','clinical_ecg_spd','clinical_ecg_simple_spd','clinical_ecg_hilbert']

def score(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])

def fit():
    if PRIVATE.exists():raise FileExistsError('Preserve frozen/partial predictions; version a new run')
    PRIVATE.mkdir(parents=True,mode=0o700);data=load_data(validation_targets=False);y=data['fit']['targets']['shd_moderate_or_greater_flag'];ym=data['monitor']['targets']['shd_moderate_or_greater_flag']
    splits=list(StratifiedKFold(5,shuffle=True,random_state=64001).split(np.zeros(len(y)),y));fold=np.empty(len(y),np.int8)
    for j,(_,test) in enumerate(splits):fold[test]=j
    foldfile=PRIVATE/'training_folds.npz';np.savez(foldfile,fold=fold,fit_metadata_rows=data['rep_indices'],monitor_metadata_rows=data['monitor_indices']);artifacts=[foldfile];summary=[]
    for family in FAMILIES:
        for features in FEATURES:
            start=time.perf_counter();x=matrix(data['fit'],features);xm=matrix(data['monitor'],features);xv=matrix(data['val'],features)
            if family=='logistic':model=LogisticRegression(max_iter=3000,random_state=64001);grid={'model__C':[.01,.1,1.,10.]}
            else:model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,l2_regularization=1.,early_stopping=False,random_state=64001);grid={'model__max_leaf_nodes':[7,15]}
            pipeline=Pipeline([('preprocessing',preprocessing(features)),('model',model)])
            search=GridSearchCV(pipeline,grid,scoring='roc_auc',cv=splits,n_jobs=1,refit=True,error_score='raise');search.fit(x,y)
            native_monitor=search.predict_proba(xm)[:,1];native_val=search.predict_proba(xv)[:,1];cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(native_monitor,1e-6,1-1e-6))[:,None],ym)
            if cal.coef_[0,0]<=0:raise RuntimeError('Calibration slope nonpositive; investigate')
            calibrated_monitor=cal.predict_proba(logit(np.clip(native_monitor,1e-6,1-1e-6))[:,None])[:,1];calibrated_val=cal.predict_proba(logit(np.clip(native_val,1e-6,1-1e-6))[:,None])[:,1]
            cuts=[]
            for kind in ['sensitivity','specificity']:
                for target in [.9,.95]:cuts.append({'kind':kind,'target':target,'threshold':threshold_for(ym,calibrated_monitor,kind,target)})
            stem=PRIVATE/f'{family}__{features}';modelpath=stem.with_suffix('.joblib');predpath=stem.with_suffix('.npz');definition=stem.with_suffix('.json')
            joblib.dump({'model':search.best_estimator_,'calibrator':cal},modelpath);np.savez(predpath,native_val=native_val,calibrated_val=calibrated_val)
            record={'family':family,'features':features,'best_parameters':search.best_params_,'CV_AUROC':float(search.best_score_),'thresholds':cuts,'training_patients':len(y),'monitor_patients':len(ym),'validation_patients':len(xv),'seconds':time.perf_counter()-start,'raw_input_features':x.shape[1],'calibration_intercept':float(cal.intercept_[0]),'calibration_slope':float(cal.coef_[0,0])};definition.write_text(json.dumps(record,indent=2));summary.append(record);artifacts.extend([modelpath,predpath,definition]);(PRIVATE/'progress.json').write_text(json.dumps(summary,indent=2));print(f'Fitted {family}: {features}',flush=True)
    for name in ['questions_common.py','questions_prediction.py']:artifacts.append(Path(__file__).with_name(name))
    artifacts.extend([PROTOCOL,ROOT/'FREEZE.json']);freeze(PRIVATE,artifacts,{'metadata_sha256':sha(META),'protocol_sha256':sha(PROTOCOL),'folds_sha256':sha(foldfile),'selection':'Training-only CV, monitor-only sigmoid and thresholds; no validation outcomes read','evaluation_status':'Exploratory project validation previously inspected'})
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'FIT_SUMMARY.json').write_text(json.dumps({'models':summary,'freeze_verified':True,'validation_outcomes_read':False},indent=2));print('All choices/calibrators/thresholds frozen before validation scoring.',flush=True)

def evaluate():
    manifest=verify_freeze(PRIVATE)
    if sha(PROTOCOL)!=manifest['details']['protocol_sha256']:raise ValueError('Protocol changed')
    if (REPORT/'PREDICTION_RESULTS.json').exists():raise FileExistsError('Preserve evaluated results')
    data=load_data(validation_targets=True);y=data['val']['targets']['shd_moderate_or_greater_flag'];models=[];predictions=[];definitions=[]
    for family in FAMILIES:
        for feature in FEATURES:
            stem=PRIVATE/f'{family}__{feature}';z=np.load(stem.with_suffix('.npz'),allow_pickle=False);definition=json.loads(stem.with_suffix('.json').read_text());native=z['native_val'];calibrated=z['calibrated_val'];predictions.append(calibrated);definitions.append((family,feature));points=[]
            for cut in definition['thresholds']:points.append({**cut,**classification(y,calibrated,cut['threshold'])})
            models.append({'family':family,'features':feature,'native':metrics(y,native),'calibrated':metrics(y,calibrated),'operating_points':points})
    predictions=np.array(predictions);rng=np.random.default_rng(64002);bootstrap=[]
    for _ in range(2000):
        i=rng.integers(0,len(y),len(y));bootstrap.append(np.array([score(y[i],p[i]) for p in predictions]))
    bootstrap=np.array(bootstrap);contrasts=[]
    for family in FAMILIES:
        for challenger,reference,primary in [('clinical_ecg_spd','clinical_ecg',True),('clinical_ecg_simple_spd','clinical_ecg_simple',False),('clinical_ecg_hilbert','clinical_ecg_simple',False),('clinical_ecg_simple','clinical_ecg',False)]:
            a=definitions.index((family,challenger));b=definitions.index((family,reference));point=score(y,predictions[a])-score(y,predictions[b]);draws=bootstrap[:,a]-bootstrap[:,b]
            metrics_diff={}
            for j,name in enumerate(['AUROC','average_precision','Brier']):metrics_diff[name]={'difference':float(point[j]),'paired_95_ci':[float(v) for v in np.quantile(draws[:,j],[.025,.975])],'paired_97.5_ci':[float(v) for v in np.quantile(draws[:,j],[.0125,.9875])]}
            contrasts.append({'family':family,'challenger':challenger,'reference':reference,'primary':primary,'metrics':metrics_diff})
    for j,m in enumerate(models):m['calibrated_patient_bootstrap_95_ci']={name:[float(v) for v in np.quantile(bootstrap[:,j,k],[.025,.975])] for k,name in enumerate(['AUROC','average_precision','Brier'])}
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'PREDICTION_RESULTS.json').write_text(json.dumps({'exploratory':True,'test_sealed':True,'patients':len(y),'freeze_sha256':sha(PRIVATE/'FREEZE.json'),'bootstrap_replicates':2000,'interval_scope':'Paired validation patients conditional on fitted models; excludes training, tuning and representation uncertainty. Primary AUROC contrasts use 97.5% intervals across two families.','models':models,'contrasts':contrasts},indent=2));print('Paired prediction comparisons saved, aggregate only.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','evaluate']);args=parser.parse_args()
    with threadpool_limits(limits=4):fit() if args.stage=='fit' else evaluate()
if __name__=='__main__':main()
