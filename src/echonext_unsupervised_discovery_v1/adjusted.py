"""Exploratory post hoc adjusted associations, not a causal or prediction analysis."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .data import verify_freeze,sha
from .audit_patterns import number

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();m=verify_freeze(a.run)
    if sha(a.metadata)!=m['details']['metadata_sha256']:raise ValueError('Metadata changed')
    rows=[]
    with a.metadata.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split']=='val': rows.append({k:r.get(k,'') for k in ['age_at_ecg','sex','ventricular_rate','qrs_duration','pr_interval','qt_corrected','acquisition_year','location_setting','race_ethnicity','shd_moderate_or_greater_flag']})
    y=np.array([float(r['shd_moderate_or_greater_flag']) for r in rows]);out=[]
    for file in sorted(a.run.glob('*62001.assignments.npy')):
        if str(file.resolve()) not in m['files']:raise ValueError('Unfrozen assignment')
        labels=np.load(file,allow_pickle=False); categories=np.unique(labels)
        if len(categories)!=2:continue
        for specification in ['age_sex','age_sex_ECG_setting_year_race']:
            columns=[np.ones(len(y)),(labels==categories[1]).astype(float)];names=['intercept','cluster_1_vs_0']
            numeric=['age_at_ecg'] if specification=='age_sex' else ['age_at_ecg','ventricular_rate','qrs_duration','pr_interval','qt_corrected','acquisition_year']
            categorical=['sex'] if specification=='age_sex' else ['sex','location_setting','race_ethnicity']
            for name in numeric:
                v=np.array([number(r[name]) for r in rows]);missing=~np.isfinite(v)
                if missing.all():continue
                v[missing]=np.median(v[~missing]);sd=v.std()
                if sd>0:columns.append((v-v.mean())/sd);names.append(name)
                if missing.any() and not missing.all():columns.append(missing.astype(float));names.append(name+'_missing')
            for name in categorical:
                v=np.array([r[name] or 'missing' for r in rows]);values,counts=np.unique(v,return_counts=True);rare=set(values[counts<50]);v=np.array(['other' if item in rare else item for item in v])
                for value in sorted(set(v))[1:]:columns.append((v==value).astype(float));names.append(name+':'+value)
            x=np.array(columns).T
            def loss(b):return np.logaddexp(0,x@b).sum()-y@(x@b)+1e-8*(b@b)
            def jac(b):return x.T@(expit(x@b)-y)+2e-8*b
            fit=minimize(loss,np.zeros(x.shape[1]),jac=jac,method='BFGS',options={'gtol':1e-6,'maxiter':1000});prob=expit(x@fit.x);information=(x.T*(prob*(1-prob)))@x+2e-8*np.eye(x.shape[1]);se=np.sqrt(np.diag(np.linalg.pinv(information)));b=fit.x[1]
            out.append({'representation':file.name.split('_62001')[0],'specification':specification,'odds_ratio_cluster_1_vs_0':float(np.exp(b)),'wald_95_ci':[float(np.exp(b-1.96*se[1])),float(np.exp(b+1.96*se[1]))],'max_abs_gradient':float(np.abs(jac(fit.x)).max()),'optimizer_success':bool(fit.success),'design_rank':int(np.linalg.matrix_rank(x)),'design_columns':x.shape[1],'covariates':names[2:]})
    a.report.write_text(json.dumps({'post_hoc_exploratory':True,'caveat':'Association conditioning on measured covariates; no causal or novel-phenotype inference; 1e-8 numerical ridge, model-based Wald intervals','results':out},indent=2));print('Aggregate exploratory adjusted associations saved.')
if __name__=='__main__':main()
