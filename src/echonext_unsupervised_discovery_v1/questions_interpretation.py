"""Interpretable train-fitted descriptions of existing frozen groups, no SHD fitting."""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from echonext.data import canonical_batch,open_waveforms
from .questions_common import load_data,ROOT
from .phase1_audit import LEADS
from .data import freeze,sha

REPORT=Path('reports/unsupervised_discovery_v1/five_questions_v1')
PRIVATE=Path('private/unsupervised_discovery_v1/five_questions_interpretation_v1')

def correlations_from_logspd(features):
    n=len(features);i,j=np.triu_indices(12);log=np.zeros((n,12,12));log[:,i,j]=features[:,:78]/np.where(i==j,1,np.sqrt(2));log[:,j,i]=log[:,i,j];e,u=np.linalg.eigh(log);s=(u*np.exp(e)[:,None,:])@u.transpose(0,2,1);c=(s-.1*np.eye(12))/.9;v=np.maximum(np.diagonal(c,axis1=1,axis2=2),0);d=np.sqrt(v[:,:,None]*v[:,None,:]);r=np.divide(c,d,out=np.zeros_like(c),where=d>1e-10);return np.clip(r,-1,1)

def descriptors(role):
    correlations=correlations_from_logspd(role['spd']);i,j=np.triu_indices(12,1);simple=role['waveform'];std=simple[:,12:24];energy=np.sqrt(np.mean(std.astype(float)**2,axis=1));derivative=np.sqrt(np.mean(simple[:,36:48].astype(float)**2,axis=1));x=np.column_stack([correlations[:,i,j],std,energy,derivative]);names=[f'corr_{LEADS[a]}_{LEADS[b]}' for a,b in zip(i,j)]+['normalized_SD_'+lead for lead in LEADS]+['normalized_RMS','lead_average_derivative_RMS'];return x,names

def main():
    if PRIVATE.exists():raise FileExistsError('Preserve interpretation models')
    PRIVATE.mkdir(parents=True,mode=0o700);data=load_data(validation_targets=False);features={role:descriptors(data[role])[0] for role in ['fit','monitor','val']};names=descriptors(data['fit'])[1]
    # Independently verify the matrix-log inversion on 256 validation signals.
    waveform=open_waveforms('private/stage2_v1/development_cache/EchoNext_val_waveforms.npy');raw=canonical_batch(waveform[:256]).astype(float);raw-=raw.mean(-1,keepdims=True);c=raw@raw.transpose(0,2,1);v=np.diagonal(c,axis1=1,axis2=2);r=c/np.sqrt(v[:,:,None]*v[:,None,:]);reconstruction=correlations_from_logspd(data['val']['spd'][:256]);error=float(np.max(np.abs(r-reconstruction)))
    if error>1e-4:raise ValueError('Covariance inversion precision check failed')
    results=[];files=[]
    for method in ['spd','waveform','hilbert']:
        frozen=joblib.load(ROOT/f'{method}_62001.joblib')
        def assignments(role):
            if method=='hilbert':
                scaler,mapper=joblib.load(ROOT/'hilbert_62001.transform.joblib');z=mapper.transform(scaler.transform(data[role]['waveform']))
            else:z=data[role][method]
            return frozen['estimator'].predict(frozen['scaler'].transform(z))
        yfit=assignments('fit');ymon=assignments('monitor');yval=assignments('val')
        if not np.array_equal(yval,data['groups'][method]):raise ValueError('Cached group agreement failed')
        tree=DecisionTreeClassifier(max_depth=2,min_samples_leaf=500,random_state=64001).fit(features['fit'],yfit);path=PRIVATE/f'{method}.joblib';joblib.dump(tree,path);files.append(path)
        rule=export_text(tree,feature_names=names,decimals=4);used=[names[i] for i in tree.tree_.feature if i>=0];result={'representation':method,'rule':rule,'features_used':used,'monitor_accuracy':float(accuracy_score(ymon,tree.predict(features['monitor']))),'monitor_balanced_accuracy':float(balanced_accuracy_score(ymon,tree.predict(features['monitor']))),'validation_accuracy':float(accuracy_score(yval,tree.predict(features['val']))),'validation_balanced_accuracy':float(balanced_accuracy_score(yval,tree.predict(features['val']))),'group_definition_changed':False};results.append(result)
    config=Path('config/unsupervised_discovery_v1/five_questions_v1/interpretation.json');freeze(PRIVATE,files+[Path(__file__),config,ROOT/'FREEZE.json'],{'outcomes_used':False,'purpose':'Explain original groups, not redefine phenotypes','inversion_max_abs_error':error})
    (REPORT/'INTERPRETABLE_RULES.json').write_text(json.dumps({'exploratory':True,'no_SHD_labels_used':True,'inverse_log_covariance_max_error':error,'rules':results},indent=2));print('Group explanations saved; no new SHD classifier/phenotype was defined.')
if __name__=='__main__':main()
