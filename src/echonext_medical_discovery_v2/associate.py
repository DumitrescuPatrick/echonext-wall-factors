"""Exploratory supervised-reference factor associations, not unsupervised validation."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from echonext_unsupervised_discovery_v1.questions_associations import design,estimate
from echonext_unsupervised_discovery_v1.questions_interpretation import correlations_from_logspd
from echonext_unsupervised_discovery_v1.data import sha,verify_freeze
from threadpoolctl import threadpool_limits
from .latent import PRIVATE,REPORT,CACHE,BASE,CONFIG,number


def independent_columns(x):
    x=x-x.mean(0);_,r,pivot=qr(x,mode='economic',pivoting=True);d=np.abs(np.diag(r));rank=int((d>max(d.max(),1)*1e-9).sum());return x[:,np.sort(pivot[:rank])]

def BH(rows,total):
    order=sorted([r for r in rows if r.get('status')=='estimated'],key=lambda r:r['p_value']);q=1.
    for rank,r in reversed(list(enumerate(order,1))):q=min(q,r['p_value']*total/rank);r['BH_q']=min(1.,q)


def main():
    out=REPORT/'FACTOR_ASSOCIATIONS.json'
    if out.exists():raise FileExistsError('Preserve factor association output')
    frozen=verify_freeze(PRIVATE);config=json.loads(CONFIG.read_text());meta=CACHE/'echonext_metadata_100k.csv'
    if sha(meta)!=frozen['details']['metadata_sha256']:raise ValueError('Metadata changed')
    names=['age_at_ecg','sex','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year','location_setting','race_ethnicity',*config['primary_endpoints'],*config['secondary_endpoints']];rows=[]
    with meta.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split']=='val':rows.append({name:r.get(name,'') for name in names})
    raw=np.load(PRIVATE/'frozen_embeddings.npz',allow_pickle=False);factor=np.load(PRIVATE/'factor_scores.npz',allow_pickle=False);logit=raw['val_logit'].astype(float);old=np.load('private/stage2_v1/tune_3/validation_predictions.npz',allow_pickle=False)['logits'];error=float(np.max(np.abs(logit-old)))
    if error>1e-4:raise ValueError('Frozen model inference does not reproduce saved logits')
    simple=np.load(BASE/'waveform_62001.npz',allow_pickle=False)['val'];energy=np.sqrt(np.mean(simple[:,12:24].astype(float)**2,axis=1));corr=correlations_from_logspd(np.load(BASE/'spd_62001.npz',allow_pickle=False)['val'])[:,0,2];base,_,_=design(rows,'flexible_plus_energy',energy);axisproxy=StandardScaler().fit_transform(SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False).fit_transform(corr[:,None]));primary_cov=independent_columns(np.column_stack([base,axisproxy,logit]))
    age=np.array([number(r['age_at_ecg']) for r in rows])[:,None];ages=StandardScaler().fit_transform(SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False).fit_transform(age));measure=np.array([[number(r[n]) for n in ['ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year']] for r in rows]);numeric=StandardScaler().fit_transform(SimpleImputer(strategy='median',add_indicator=True).fit_transform(measure));category=OneHotEncoder(drop='first',min_frequency=50,sparse_output=False).fit_transform(np.array([[r[n] or 'missing' for n in ['sex','location_setting','race_ethnicity']] for r in rows]));secondary_cov=independent_columns(np.column_stack([ages,numeric,category,np.log(np.maximum(energy,1e-8)),corr,logit]));results=[]
    traits={name:np.array([number(r[name]) for r in rows]) for name in ['age_at_ecg','ventricular_rate','qrs_duration','qt_corrected']};traits.update({'normalized_RMS':energy,'I_III_correlation_proxy':corr,'frozen_ResNet_logit':logit});trait_results=[]
    for space in ['pooled','residual_pooled','temporal']:
        x=factor[f'{space}_val']
        for j in range(8):
            for name,value in traits.items():
                keep=np.isfinite(value);trait_results.append({'space':space,'factor':j+1,'trait':name,'Pearson_r':float(np.corrcoef(x[keep,j],value[keep])[0,1])})
            for endpoint in config['primary_endpoints']+config['secondary_endpoints']:
                family='primary' if endpoint in config['primary_endpoints'] else 'secondary';cov=primary_cov if family=='primary' else secondary_cov;y=np.array([number(r[endpoint]) for r in rows]);keep=np.isin(y,[0,1]);record={'space':space,'factor':j+1,'endpoint':endpoint,'family':family,'patients':int(keep.sum()),'events':int(y[keep].sum()),'scale':'One fitting-patient score SD','upstream_supervised_and_validation_selected':True}
                try:record.update(estimate(y[keep],x[keep,j],cov[keep]));record['status']='estimated';record['numerical_review_required']=record['hessian_condition']>1e9;record['event_count_per_parameter']=record['events']/record['parameters']
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:record.update({'status':'not_estimable','reason':type(exc).__name__})
                results.append(record)
        print('Factor association space complete: '+space,flush=True)
    BH([r for r in results if r['family']=='primary'],48);BH([r for r in results if r['family']=='secondary'],120)
    # Descriptive cluster profiles, never named as newly discovered disease subtypes.
    clusters=[]
    for space in ['pooled','residual_pooled','temporal']:
        labels=np.load(PRIVATE/f'{space}_clusters.npz',allow_pickle=False)['assignments'][0]
        for k in np.unique(labels):
            mask=labels==k;entry={'space':space,'cluster':int(k),'patients':int(mask.sum()),'mean_ResNet_logit':float(logit[mask].mean()),'mean_age':float(age[mask].mean()),'mean_QRS_ms':float(traits['qrs_duration'][mask].mean()),'mean_I_III_proxy':float(corr[mask].mean()),'endpoints':{}}
            for endpoint in config['primary_endpoints']:
                y=np.array([number(r[endpoint]) for r in rows]);event=int(y[mask].sum());n=int(mask.sum());entry['endpoints'][endpoint]={'prevalence':event/n,'events':event} if min(event,n-event)>=10 else {'suppressed':True}
            clusters.append(entry)
    out.write_text(json.dumps({'status':'Hypothesis-generating from supervised and validation-selected ResNet; FDR does not remove upstream selection bias','max_logit_reproduction_error':error,'adjustment':'Primary: available clinical/context splines + normalized RMS + I-III proxy spline + frozen ResNet logit. Actual electrical axis unmeasured.','primary_planned_tests':48,'secondary_planned_tests':120,'associations':results,'trait_correlations':trait_results,'cluster_profiles_descriptive':clusters,'PTB_neural_factor_validation_performed':False,'test_accessed':False},indent=2));print('Exploratory neural-factor associations saved; no independent discovery claim.',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
