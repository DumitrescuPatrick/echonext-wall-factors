"""Nested patient-separated fusion diagnostic with explicit upstream-bias warning."""
import json,time
from pathlib import Path
import numpy as np
import joblib
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
from echonext.stage2_metrics import metrics
from echonext_unsupervised_discovery_v1.questions_common import load_data
from echonext_unsupervised_discovery_v1.questions_euclidean import euclidean
from echonext_unsupervised_discovery_v1.data import freeze,verify_freeze,sha
from .freeze import PRIVATE,CONFIG
from .analysis import net_benefit,check_patient_separation

REPORT=Path('reports/covariance_validation_v1')
NAMES=['ResNet','ResNet_clinical_simple','ResNet_clinical_simple_Euclidean','ResNet_clinical_simple_logSPD']

def score(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])

def main():
    verify_freeze(PRIVATE);folder=PRIVATE/'fusion_diagnostic'
    if folder.exists():raise FileExistsError('Preserve fusion diagnostic')
    folder.mkdir(mode=0o700);config=json.loads(CONFIG.read_text());data=load_data(validation_targets=True)['val'];y=data['targets']['shd_moderate_or_greater_flag'];z=np.load('private/stage2_v1/tune_3/validation_predictions.npz',allow_pickle=False)
    if not np.array_equal(z['y'],y):raise ValueError('ResNet/population mismatch')
    r=z['logits'].astype(float)[:,None];base=np.column_stack([r,data['clinical'],data['waveform']]);matrices=[r,base,np.column_stack([base,euclidean(data['spd'])]),np.column_stack([base,data['spd']])]
    # One validation record per patient was verified by the metadata linkage loader.
    outer=list(StratifiedKFold(5,shuffle=True,random_state=67001).split(base,y));pred=np.zeros((4,len(y)));folds=np.empty(len(y),np.int8);choices=[];artifacts=[]
    for j,(train,test) in enumerate(outer):
        check_patient_separation(test,train);folds[test]=j
        for k,(name,x) in enumerate(zip(NAMES,matrices)):
            start=time.perf_counter();pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(max_iter=3000,random_state=67001));cv=StratifiedKFold(3,shuffle=True,random_state=67001+j);search=GridSearchCV(pipe,{'logisticregression__C':[.01,.1,1.,10.]},scoring='roc_auc',cv=cv,n_jobs=1,error_score='raise').fit(x[train],y[train]);cal=CalibratedClassifierCV(search.best_estimator_,method='sigmoid',cv=cv,ensemble=False).fit(x[train],y[train]);pred[k,test]=cal.predict_proba(x[test])[:,1];path=folder/f'{name}_fold{j}.joblib';joblib.dump(cal,path);artifacts.append(path)
            fitted=cal.calibrated_classifiers_[0].estimator;coefficient=fitted.named_steps['logisticregression'].coef_;choices.append({'model':name,'fold':j,'inner_C':search.best_params_['logisticregression__C'],'input_columns':x.shape[1],'fitted_coefficients':int(coefficient.size+1),'additional_sigmoid_parameters':2,'CV_AUROC':float(search.best_score_),'seconds':time.perf_counter()-start})
        print(f'Fusion diagnostic fold {j+1}/5 complete',flush=True)
    file=folder/'patient_predictions.npz';np.savez(file,probability=pred,fold=folds);artifacts.append(file);freeze(folder,artifacts+[Path(__file__),CONFIG,PRIVATE/'FREEZE.json'],{'base_score_patient_independence':False,'fusion_patient_separation':True,'warning':'All target outcomes influenced historical ResNet model selection; diagnostic only'})
    values=[{'model':name,'metrics':metrics(y,p)} for name,p in zip(NAMES,pred)];thresholds=np.array(config['development_diagnostic']['decision_curves']);nb=np.array([net_benefit(y,p,thresholds) for p in pred]);rng=np.random.default_rng(67002);bootstrap=[];nb_boot=[]
    for _ in range(2000):
        i=rng.integers(0,len(y),len(y));bootstrap.append(np.array([score(y[i],p[i]) for p in pred]));nb_boot.append(np.array([net_benefit(y[i],p[i],thresholds) for p in pred]))
    bootstrap=np.array(bootstrap);nb_boot=np.array(nb_boot);contrasts=[]
    for a,b in [(2,1),(3,1),(3,2),(1,0)]:
        delta=score(y,pred[a])-score(y,pred[b]);sample=bootstrap[:,a]-bootstrap[:,b];dn=nb[a]-nb[b];sn=nb_boot[:,a]-nb_boot[:,b]
        contrasts.append({'challenger':NAMES[a],'reference':NAMES[b],'metric_differences':{name:{'difference':float(delta[k]),'paired_95_ci':[float(v) for v in np.quantile(sample[:,k],[.025,.975])]} for k,name in enumerate(['AUROC','average_precision','Brier'])},'net_benefit_difference':[{'threshold':float(t),'difference':float(dn[k]),'paired_95_ci':[float(v) for v in np.quantile(sn[:,k],[.025,.975])]} for k,t in enumerate(thresholds)]})
    output={'status':'Selection-contaminated exploratory diagnostic, not an independent ResNet increment test','patient_separated_fusion':True,'patient_independent_base_model_selection':False,'test_opened':False,'patients':len(y),'outer_folds':5,'inner_folds':3,'raw_frozen_ResNet_descriptive_metrics':metrics(y,expit(r[:,0])),'models':values,'complexity_and_choices':choices,'contrasts':contrasts,'decision_curves':{'thresholds':thresholds.tolist(),'model_net_benefit':nb.tolist(),'treat_all':(y.mean()-(1-y.mean())*thresholds/(1-thresholds)).tolist(),'treat_none':[0.]*len(thresholds),'threshold_interpretation':'Assumed echo referral harm/benefit ratios, not validated clinical action thresholds'},'uncertainty':'Paired patients conditional on fitted outer models; no neural retraining/selection uncertainty; intervals cannot correct historical base-model selection bias'}
    (REPORT/'FUSION_DIAGNOSTIC.json').write_text(json.dumps(output,indent=2));print('Fusion diagnostic saved with upstream-selection warning.',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
