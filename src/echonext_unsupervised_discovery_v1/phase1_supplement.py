"""Reproducible supplemental Phase-1 provenance, model inventory and limb-lead QC."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,brier_score_loss
from echonext.data import open_waveforms,canonical_batch
from .data import sha,verify_freeze

def limb_residuals(x):
    z=x.astype(float);z-=z.mean(-1,keepdims=True);c=z@z.transpose(0,2,1)/(z.shape[-1]-1);inverse=np.linalg.pinv(c[:,:2,:2],hermitian=True);result=[]
    for j in range(2,6):
        v=c[:,j,:2];explained=np.einsum('bi,bij,bj->b',v,inverse,v);result.append(np.clip((c[:,j,j]-explained)/np.maximum(c[:,j,j],1e-12),0,1))
    return np.array(result).T

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--root',type=Path,default=Path('private/stage2_v1/development_cache'));parser.add_argument('--run',type=Path,default=Path('private/unsupervised_discovery_v1/classical_v1'));parser.add_argument('--source',type=Path,default=Path.home()/'Downloads');a=parser.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    names=['ORIGINAL_DEVELOPMENT_INTEGRITY.json','EXISTING_MODEL_INVENTORY.json','NUMERICAL_REPRODUCIBILITY.json','LIMB_LEAD_ALGEBRA_QC.json']
    if any((a.output/name).exists() for name in names):raise FileExistsError('Use a fresh supplemental output directory')
    manifest=json.loads(Path('reports/input_integrity.json').read_text());integrity={}
    for name in ['EchoNext_train_waveforms.npy','EchoNext_val_waveforms.npy','echonext_metadata_100k.csv']:
        path=a.source/name;entry=manifest[name];ok=path.exists() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'];integrity[name]={'exists':path.exists(),'bytes':path.stat().st_size if path.exists() else None,'original_source_hash_verified':ok}
        if not ok:raise ValueError('Original development source mismatch')
    (a.output/names[0]).write_text(json.dumps({'original_development_inputs':integrity,'test_file_opened':False},indent=2))
    rows=[];yy=None;z3=None
    for i in range(6):
        info=json.loads(Path(f'reports/stage2_v1/tune_{i}_complete.json').read_text());pred=Path(f'private/stage2_v1/tune_{i}/validation_predictions.npz');best=Path(f'private/stage2_v1/tune_{i}/best.pt')
        if sha(pred)!=info['private_prediction_sha256'] or sha(best)!=info['best_checkpoint_sha256']:raise ValueError('Existing checkpoint or predictions changed')
        with np.load(pred,allow_pickle=False) as data:
            if yy is None:yy=data['y'].copy()
            if not np.array_equal(yy,data['y']):raise ValueError('Model evaluation populations differ')
            logits=data['logits'];probability=expit(logits.astype(float))
            if i==3:z3=logits.copy()
        rows.append({'candidate':i,'configuration':info['configuration'],'seed':info['seed'],'parameters':info['trainable_parameters'],'best_epoch':info['best_epoch'],'epochs_completed':info['epochs_completed'],'AUROC_recomputed':float(roc_auc_score(yy,probability)),'Brier_recomputed':float(brier_score_loss(yy,probability)),'checkpoint_hash_verified':True,'predictions_hash_verified':True})
    (a.output/names[1]).write_text(json.dumps({'models':rows,'repeated_training_seeds_completed':1,'neural_unsupervised_clinical_models_completed':0,'test_accessed':False,'warning':'Six tuning configurations, not six independent seed replications; checkpoint selection used official validation'},indent=2))
    conversions={'torch_sigmoid_float32':torch.sigmoid(torch.from_numpy(z3)).numpy(),'scipy_expit_float32':expit(z3),'scipy_expit_float64':expit(z3.astype(float))}
    numerical={k:{'AUROC':float(roc_auc_score(yy,p)),'Brier':float(brier_score_loss(yy,p))} for k,p in conversions.items()};numerical['interpretation']='Tiny AUC differences reflect finite-precision sigmoid ties; saved Stage-2 metrics remain the historical reference, no checkpoint altered.'
    (a.output/names[2]).write_text(json.dumps(numerical,indent=2));verify_freeze(a.run);wave=open_waveforms(a.root/'EchoNext_val_waveforms.npy');values=[]
    for start in range(0,len(wave),64):values.append(limb_residuals(canonical_batch(wave[start:start+64])))
    values=np.concatenate(values);results=[]
    for method in ['spd','waveform','hilbert']:
        labels=np.load(a.run/f'{method}_62001.assignments.npy',allow_pickle=False)
        for k in [0,1]:
            for j,lead in enumerate(['III','aVR','aVL','aVF']):
                x=values[labels==k,j];results.append({'representation':method,'cluster':k,'lead_predicted_from_I_II':lead,'mean_relative_residual_variance':float(x.mean()),'median':float(np.median(x)),'q95':float(np.quantile(x,.95))})
    (a.output/names[3]).write_text(json.dumps({'scope':'Post hoc within-record linear relation QC, allowing per-record scale coefficients; not a lead-placement diagnosis or a fitted population predictor','interpretation':'Derived limb leads have ideal linear relations; nonzero residual may reflect quantization, clipping, separate baseline processing or acquisition/export effects. It is not new independent cardiac information.','results':results},indent=2));print('Supplemental Phase-1 audits complete; no training or test access.')
if __name__=='__main__':main()
