"""Separate reduced-LVEF/wall-thickness conditional associations; frozen clusters."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from scipy.linalg import qr
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from .questions_common import load_data,number,PROTOCOL
from .data import sha

OUTPUT=Path('reports/unsupervised_discovery_v1/five_questions_v1')

def design(rows,specification,energy):
    numeric=['age_at_ecg'] if specification=='age_sex' else ['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year']
    categorical=['sex'] if specification=='age_sex' else ['sex','location_setting','race_ethnicity']
    values=np.array([[number(r[n]) for n in numeric] for r in rows]);missing=~np.isfinite(values)
    if specification=='flexible_plus_energy':values=np.column_stack([values,np.log(np.maximum(energy,1e-8))]);numeric=numeric+['log_normalized_RMS'];missing=~np.isfinite(values)
    imputed=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(values)
    if specification=='age_sex':continuous=StandardScaler().fit_transform(imputed);names=numeric
    else:
        spline=SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False,extrapolation='linear');continuous=spline.fit_transform(imputed);names=list(spline.get_feature_names_out(numeric));continuous=StandardScaler().fit_transform(continuous)
    indicators=missing[:,missing.any(0)].astype(float);names=names+[n+'_missing' for j,n in enumerate(numeric) if missing[:,j].any()]
    categories=np.array([[r[n] or 'missing' for n in categorical] for r in rows]);encoder=OneHotEncoder(drop='first',min_frequency=50,sparse_output=False,handle_unknown='infrequent_if_exist');cats=encoder.fit_transform(categories);names+=list(encoder.get_feature_names_out(categorical))
    cov=np.column_stack([continuous,indicators,cats]);cov-=cov.mean(0)
    # Drop redundant nuisance columns while keeping the group coefficient untouched.
    _,r,pivot=qr(cov,mode='economic',pivoting=True);diagonal=np.abs(np.diag(r));rank=int((diagonal>max(diagonal.max(),1)*1e-9).sum());selected=np.sort(pivot[:rank]);return cov[:,selected],[names[i] for i in selected],cov.shape[1]-rank

def estimate(y,group,cov):
    x=np.column_stack([np.ones(len(y)),group,cov])
    if np.linalg.matrix_rank(x)!=x.shape[1]:raise ValueError('Association design is not identifiable')
    def loss(b):return np.logaddexp(0,x@b).sum()-y@(x@b)+1e-8*(b@b)
    def gradient(b):return x.T@(expit(x@b)-y)+2e-8*b
    fit=minimize(loss,np.zeros(x.shape[1]),jac=gradient,method='BFGS',options={'gtol':1e-6,'maxiter':1500});prob=expit(x@fit.x);hessian=(x.T*(prob*(1-prob)))@x+2e-8*np.eye(x.shape[1]);condition=float(np.linalg.cond(hessian));se=np.sqrt(np.linalg.inv(hessian)[1,1]);b=fit.x[1];grad=float(np.abs(gradient(fit.x)).max())
    if grad>1e-4 or not np.isfinite(se):raise RuntimeError('Association optimization did not converge adequately')
    return {'odds_ratio':float(np.exp(b)),'wald_95_ci':[float(np.exp(b-1.96*se)),float(np.exp(b+1.96*se))],'p_value':float(2*norm.sf(abs(b/se))),'optimizer_success':bool(fit.success),'max_gradient':grad,'hessian_condition':condition,'parameters':x.shape[1]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT/'COMPONENT_ADJUSTMENT.json');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve association results')
    data=load_data(validation_targets=True);rows=data['val']['records'];energy=np.sqrt(np.mean(data['val']['waveform'][:,12:24].astype(float)**2,axis=1));results=[]
    for spec in ['age_sex','flexible','flexible_plus_energy']:
        cov,names,dropped=design(rows,spec,energy)
        for endpoint in ['lvef_lte_45_flag','lvwt_gte_13_flag']:
            y=data['val']['targets'][endpoint]
            if not np.isin(y,[0,1]).all():raise ValueError('Nonbinary endpoint')
            for method,group in data['groups'].items():results.append({'representation':method,'endpoint':endpoint,'adjustment':spec,'patients':len(y),'events':int(y.sum()),'covariates':names,'redundant_nuisance_columns_dropped':dropped,**estimate(y,group,cov)})
    primary=[r for r in results if r['adjustment']=='flexible'];order=sorted(primary,key=lambda r:r['p_value']);q=1.
    for rank,r in reversed(list(enumerate(order,1))):q=min(q,r['p_value']*len(order)/rank);r['primary_BH_q']=q
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'exploratory':True,'protocol_sha256':sha(PROTOCOL),'interpretation':'Conditional observational associations, no causal effects; raw-voltage/axis and comorbidities unavailable. Wald intervals are pointwise.','results':results},indent=2));print('Separate endpoint adjusted associations saved, aggregate only.')
if __name__=='__main__':main()
