"""Lead I-III conditional-association sensitivity; not measured electrical axis."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from sklearn.preprocessing import SplineTransformer,StandardScaler
from echonext_unsupervised_discovery_v1.questions_common import load_data
from echonext_unsupervised_discovery_v1.questions_associations import design,estimate
from echonext_unsupervised_discovery_v1.questions_interpretation import correlations_from_logspd
from echonext_unsupervised_discovery_v1.data import verify_freeze
from .freeze import PRIVATE

def main():
    verify_freeze(PRIVATE);out=Path('reports/covariance_validation_v1/LEAD_I_III_SENSITIVITY.json')
    if out.exists():raise FileExistsError('Preserve sensitivity analysis')
    data=load_data(validation_targets=True)['val'];all_data=load_data(validation_targets=False);group=all_data['groups']['spd'];energy=np.sqrt(np.mean(data['waveform'][:,12:24].astype(float)**2,axis=1));correlation=correlations_from_logspd(data['spd'])[:,0,2]
    cov,names,_=design(data['records'],'flexible',energy);spline=StandardScaler().fit_transform(SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False).fit_transform(correlation[:,None]));augmented=np.column_stack([cov,spline]);augmented-=augmented.mean(0);_,r,pivot=qr(augmented,mode='economic',pivoting=True);rank=int((np.abs(np.diag(r))>max(np.abs(np.diag(r)).max(),1)*1e-9).sum());augmented=augmented[:,np.sort(pivot[:rank])];results=[]
    for endpoint in ['lvef_lte_45_flag','lvwt_gte_13_flag']:
        y=data['targets'][endpoint]
        for label,x in [('available_clinical_ECG_context',cov),('same_plus_I_III_spline',augmented)]:results.append({'endpoint':endpoint,'adjustment':label,**estimate(y,group,x)})
    out.write_text(json.dumps({'status':'Additional exploratory conditioning on a group-defining correlation; no actual electrical axis measured','interpretation':'Attenuation or reversal can reflect conditioning on defining features or limited overlap. No causal effect or exclusion of axis confounding is established.','results':results},indent=2));print('Lead I-III conditional sensitivity saved, aggregate only.')
if __name__=='__main__':main()
