"""Exploratory mutually adjusted factor and subgroup associations; no selection."""
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.special import logit
from threadpoolctl import threadpool_limits
from echonext_unsupervised_discovery_v1.questions_associations import estimate
from echonext_unsupervised_discovery_v1.questions_common import load_data,number
from echonext_medical_discovery_v2.associate import independent_columns
from echonext_medical_discovery_v2.latent import PRIVATE as SOURCE
from echonext_unsupervised_discovery_v1.data import verify_freeze,freeze
from .run import ROOT,OUT,TARGET,subgroups

def main():
    path=OUT/'ADJUSTED_ASSOCIATIONS.json'
    if path.exists():raise FileExistsError('Preserve follow-up associations')
    verify_freeze(ROOT);verify_freeze(SOURCE/'component_reference');folder=ROOT/'association_lock';folder.mkdir(mode=0o700)
    freeze(folder,[__file__,ROOT/'FREEZE.json'],{'adjustment':'Age, recorded sex, ventricular rate, QRS, year, normalized RMS, frozen source risk, nonlinear endpoint-specific conventional/covariance reference risk; each factor adjusted for the other','status':'Exploratory post-discovery; subgroup results pointwise; no independence claim'})
    d=load_data(validation_targets=True);r=d['val'];scores=np.load(SOURCE/'factor_scores.npz',allow_pickle=False);factors=np.column_stack([scores['pooled_val'][:,1],scores['temporal_val'][:,4]]);raw=np.load(SOURCE/'frozen_embeddings.npz',allow_pickle=False);ref=np.load(SOURCE/'component_reference/validation_risks.npz',allow_pickle=False)[TARGET];energy=np.sqrt(np.mean(r['waveform'][:,12:24].astype(float)**2,axis=1));year=np.array([number(v['acquisition_year']) for v in r['records']]);x=np.column_stack([r['clinical'][:,[5,6,0,3]],year,np.log(energy.clip(1e-8)),raw['val_logit'],logit(np.clip(ref,1e-6,1-1e-6))]);x=StandardScaler().fit_transform(SimpleImputer(strategy='median',add_indicator=True).fit_transform(x));groups,_=subgroups(r['records'],d['fit']['records']);groups={'all':np.ones(len(x),bool),**groups};y=r['targets'][TARGET];results=[]
    for label,ix in groups.items():
        n=int(ix.sum());e=int(y[ix].sum())
        if n<100 or min(e,n-e)<20:continue
        for j,name in enumerate(['pooled2','temporal5']):
            row={'subgroup':label,'factor':name,'patients':n,'events':e}
            try:
                cov=independent_columns(np.column_stack([x[ix],factors[ix,1-j]]));row.update(estimate(y[ix],factors[ix,j],cov));row['events_per_parameter']=e/row['parameters'];row['sparse_or_ill_conditioned']=min(e,n-e)/row['parameters']<10 or row['hessian_condition']>1e9;row['status']='estimated'
            except (ValueError,RuntimeError,np.linalg.LinAlgError):row['status']='not_estimable'
            results.append(row)
    path.write_text(json.dumps({'status':'Exploratory conditional associations; pointwise intervals, no subgroup multiplicity claims','actual_axis_available':False,'validated_voltages_available':False,'results':results},indent=2));print('Adjusted mutually controlled factor/subgroup associations saved.')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
