"""Phase 1: audit frozen EchoNext patterns; no fitting or test waveform access.

All new analyses are explicitly exploratory. Only aggregate outputs are written.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,brier_score_loss,adjusted_rand_score
from .data import linkage,roles,sha,verify_freeze
from .audit_patterns import number
from echonext.data import open_waveforms,canonical_batch
from echonext.schema import PRIMARY,SECONDARY

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
NUMERIC=['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected']

def signal_descriptors(x):
    """Normalized-source descriptors, not raw-voltage diagnostic criteria."""
    if x.ndim!=3 or x.shape[1]!=12 or not np.isfinite(x).all():raise ValueError('Invalid ECG array')
    z=x.astype(np.float64);z-=z.mean(-1,keepdims=True)
    covariance=z@z.transpose(0,2,1)/max(1,z.shape[-1]-1)
    variances=np.diagonal(covariance,axis1=1,axis2=2);denom=np.sqrt(variances[:,:,None]*variances[:,None,:])
    correlation=np.divide(covariance,denom,out=np.zeros_like(covariance),where=denom>1e-12)
    rows,cols=np.triu_indices(12,1);std=np.sqrt(variances)
    values=[correlation[:,rows,cols]];names=[f'Pearson correlation {LEADS[i]} / {LEADS[j]}' for i,j in zip(rows,cols)]
    values.append(std);names.extend('Normalized-source SD '+lead for lead in LEADS)
    rms=np.sqrt(np.mean(z*z,axis=(1,2)));window=np.hanning(z.shape[-1]);fft=np.fft.rfft(z*window,axis=-1);power=(fft.real**2+fft.imag**2).mean(1);freq=np.fft.rfftfreq(z.shape[-1],1/250)
    total=power[:,(freq>=.5)&(freq<=100)].sum(1).clip(1e-20)
    bands=[(.5,5),(5,15),(15,40),(40,100),(49,51),(59,61)]
    for low,high in bands:
        values.append((power[:,(freq>=low)&(freq<high)].sum(1)/total)[:,None]);names.append(f'Relative spectral power {low:g}-{high:g} Hz')
    eigen=np.linalg.eigvalsh(covariance);rank=(eigen>np.maximum(eigen[:,-1:],1e-20)*1e-6).sum(1)
    quality=np.column_stack([rms,(std<1e-6).any(1).astype(float),(np.diff(x,axis=-1)==0).mean((1,2)),np.abs(z).max((1,2))/np.maximum(rms,1e-12),rank])
    names.extend(['Normalized-source RMS','Any near-flat lead indicator','Exactly equal adjacent-sample fraction','Maximum absolute amplitude / RMS','Unregularized covariance numerical rank'])
    values.append(quality);out=np.concatenate(values,axis=1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite descriptors')
    return out,names

def difference_ci(a,b):
    """Pointwise large-sample interval; descriptive, not selection-adjusted."""
    delta=float(np.mean(b)-np.mean(a));se=float(np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b)))
    pooled=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return delta,[delta-1.96*se,delta+1.96*se],float(delta/pooled) if pooled>1e-12 else None

def risk_difference(a,y,rng,draws=2000):
    if min(int(a.sum()),int((~a).sum()))<20:return None
    if min(int(y[a].sum()),int((1-y[a]).sum()),int(y[~a].sum()),int((1-y[~a]).sum()))<10:return None
    values=[]
    for _ in range(draws):
        indices=rng.integers(0,len(y),len(y));groups=a[indices]
        if groups.any() and (~groups).any():values.append(float(y[indices][groups].mean()-y[indices][~groups].mean()))
    return {'patients_cluster_0':int((~a).sum()),'patients_cluster_1':int(a.sum()),'prevalence_cluster_0':float(y[~a].mean()),'prevalence_cluster_1':float(y[a].mean()),'risk_difference_cluster_1_vs_0':float(y[a].mean()-y[~a].mean()),'bootstrap_95_ci':[float(q) for q in np.quantile(values,[.025,.975])]}

def export_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('private/stage2_v1/development_cache'));parser.add_argument('--run',type=Path,default=Path('private/unsupervised_discovery_v1/classical_v1'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--draws',type=int,default=2000);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.glob('*.json')):raise FileExistsError('Preserve existing phase-1 results; use a new directory')
    frozen=verify_freeze(a.run);cache_manifest=Path('reports/stage2_v1/development_cache_integrity.json');expected=json.loads(cache_manifest.read_text());verified={}
    for name,entry in expected.items():
        file=a.root/name
        if file.stat().st_size!=entry['bytes'] or sha(file)!=entry['sha256']:raise ValueError('Cache integrity failed')
        verified[name]={'bytes':entry['bytes'],'sha256_verified':True}
    mdpath=a.root/'echonext_metadata_100k.csv';md=linkage(mdpath)
    if sha(mdpath)!=frozen['details']['metadata_sha256']:raise ValueError('Metadata differs from discovery freeze')
    fit,monitor,reps=roles(md['train']);population={'official_train_recordings':len(md['train']),'official_train_patients':len(set(md['train'])),'fitting_recordings':len(fit),'fitting_patients':len(reps),'monitor_patients':len(set(md['train'][i] for i in monitor)),'validation_patients':len(md['val']),'no_split_records_excluded':len(md['no_split']),'official_test_records_linkage_only':len(md['test'])}
    groups={s:set(p) for s,p in md.items()};population['official_split_patient_overlaps']={f'{x}_{y}':len(groups[x]&groups[y]) for x,y in [('train','val'),('train','test'),('val','test')]}
    population['no_split_overlap_validation_patients']=len(groups['no_split']&groups['val']);population['no_split_overlap_test_patients']=len(groups['no_split']&groups['test'])
    validation=[];development_outcome={s:[] for s in ['train','val']};missing={s:{k:0 for k in NUMERIC+[PRIMARY]+SECONDARY} for s in ['train','val']};serial=True
    with mdpath.open(newline='') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if '' in row:serial &= row['']==str(i)
            s=row['split']
            if s not in ('train','val'):continue # Never decode official test/no_split outcomes.
            development_outcome[s].append(number(row[PRIMARY]))
            for name in missing[s]:missing[s][name]+=int(not np.isfinite(number(row.get(name,''))))
            if s=='val':validation.append({k:row.get(k,'') for k in NUMERIC+['sex','location_setting','race_ethnicity',PRIMARY]+SECONDARY})
    y=np.array([number(r[PRIMARY]) for r in validation])
    if not np.isin(y,[0,1]).all():raise ValueError('Primary outcome missing/nonbinary')
    outcome_summary={}
    for s,values in development_outcome.items():
        z=np.array(values);keep=np.isin(z,[0,1]);outcome_summary[s]={'recordings':len(z),'events':int(z[keep].sum()),'prevalence_recording_weighted':float(z[keep].mean()),'missing':int((~keep).sum()),'note':'Training prevalence counts repeated recordings; validation is one recording per patient'}
    # Validate all development cached values without allocating the complete arrays.
    quality={}
    for split in ('train','val'):
        wave=open_waveforms(a.root/f'EchoNext_{split}_waveforms.npy');nonfinite=0;flat_leads=0
        if len(wave)!=len(md[split]):raise ValueError('Partition row count mismatch')
        for start in range(0,len(wave),128):
            batch=np.asarray(wave[start:start+128]);nonfinite+=int((~np.isfinite(batch)).sum());flat_leads+=int((np.ptp(batch[:,0],axis=1)==0).sum())
        if nonfinite:raise ValueError('Nonfinite source values; stop analysis')
        quality[split]={'shape':list(wave.shape),'dtype':str(wave.dtype),'nonfinite_values':nonfinite,'exact_flat_lead_instances':flat_leads}
    report={'population':population,'recording_level_outcomes':outcome_summary,'missing_counts':missing,'input_integrity':verified,'cache_quality':quality,'serial_index_matches_original_position':serial,'test_waveforms_opened':False,'test_outcomes_decoded':False,'labels_used_for_new_fitting':False,'new_models_fitted':False,'alignment_limit':'Provider split-specific row-order guarantee; no embedded waveform identity','amplitude_limit':'Source-normalized signals; no absolute mV criteria','analysis_status':'Exploratory characterization of previously frozen primary-seed patterns'}
    (a.output/'DATA_AUDIT.json').write_text(json.dumps(report,indent=2))
    predictions=Path('private/stage2_v1/tune_3/validation_predictions.npz');complete=json.loads(Path('reports/stage2_v1/tune_3_complete.json').read_text());checkpoint=Path('private/stage2_v1/tune_3/best.pt')
    if sha(predictions)!=complete['private_prediction_sha256'] or sha(checkpoint)!=complete['best_checkpoint_sha256']:raise ValueError('Trained-model provenance mismatch')
    with np.load(predictions,allow_pickle=False) as data:
        if not np.array_equal(y,data['y']):raise ValueError('Prediction/metadata outcome alignment mismatch')
        probability=expit(data['logits'].astype('float64'))
    if not np.isfinite(probability).all():raise ValueError('Nonfinite saved probability')
    labels={}
    for method in ['waveform','spd','hilbert']:
        path=a.run/f'{method}_62001.assignments.npy'
        if str(path.resolve()) not in frozen['files']:raise ValueError('Unfrozen assignment')
        labels[method]=np.load(path,allow_pickle=False)
        if len(labels[method])!=len(y):raise ValueError('Assignment population mismatch')
    waveform=open_waveforms(a.root/'EchoNext_val_waveforms.npy');features=[];names=None
    for start in range(0,len(waveform),64):
        z,names=signal_descriptors(canonical_batch(waveform[start:start+64]));features.append(z)
    features=np.concatenate(features);differences=[];resnet=[];subgroups=[];joint=[];numeric=[];rng=np.random.default_rng(63001)
    qrs=np.array([number(r['qrs_duration']) for r in validation]);age=np.array([number(r['age_at_ecg']) for r in validation]);sex=np.array([r['sex'] for r in validation])
    masks={'all':np.ones(len(y),bool),'QRS_lt_120ms':qrs<120,'QRS_ge_120ms':qrs>=120,'age_lt_65':age<65,'age_ge_65':age>=65,'recorded_female':sex=='female','recorded_male':sex=='male'}
    for method,cluster in labels.items():
        if not np.array_equal(np.unique(cluster),[0,1]):raise ValueError('Primary-seed binary labels required by characterization')
        for k in (0,1):
            keep=cluster==k;yy=y[keep];pp=probability[keep];boot=[]
            for _ in range(a.draws):
                index=rng.integers(0,len(yy),len(yy))
                if len(np.unique(yy[index]))==2:boot.append(roc_auc_score(yy[index],pp[index]))
            resnet.append({'representation':method,'cluster':k,'patients':int(keep.sum()),'SHD_prevalence':float(yy.mean()),'mean_ResNet_risk':float(pp.mean()),'mean_risk_minus_prevalence':float(pp.mean()-yy.mean()),'ResNet_AUROC':float(roc_auc_score(yy,pp)),'ResNet_AUROC_bootstrap_95_ci':[float(q) for q in np.quantile(boot,[.025,.975])],'ResNet_Brier':float(brier_score_loss(yy,pp))})
            for name in NUMERIC:
                v=np.array([number(r[name]) for r in validation]);valid=keep&np.isfinite(v);numeric.append({'representation':method,'cluster':k,'measurement':name,'observed':int(valid.sum()),'missing':int((keep&~np.isfinite(v)).sum()),'mean':float(v[valid].mean()),'median':float(np.median(v[valid])),'q25':float(np.quantile(v[valid],.25)),'q75':float(np.quantile(v[valid],.75))})
        for j,name in enumerate(names):
            x0=features[cluster==0,j];x1=features[cluster==1,j];delta,ci,smd=difference_ci(x0,x1)
            differences.append({'representation':method,'descriptor':name,'mean_cluster_0':float(x0.mean()),'mean_cluster_1':float(x1.mean()),'difference_1_minus_0':delta,'pointwise_95_ci_low':ci[0],'pointwise_95_ci_high':ci[1],'standardized_difference':smd})
        for name,mask in masks.items():
            r=risk_difference(cluster[mask]==1,y[mask],rng,a.draws)
            subgroups.append({'representation':method,'stratum':name,**r} if r else {'representation':method,'stratum':name,'suppressed':True})
    overlaps=[]
    for a0,b0 in [('spd','waveform'),('spd','hilbert'),('waveform','hilbert')]:
        aa=labels[a0]==1;bb=labels[b0]==1;overlaps.append({'first':a0,'second':b0,'ARI':float(adjusted_rand_score(labels[a0],labels[b0])),'high_group_Jaccard':float((aa&bb).sum()/(aa|bb).sum())})
    for first,second in [('spd','waveform'),('spd','hilbert')]:
        for i in (0,1):
            for j in (0,1):
                keep=(labels[first]==i)&(labels[second]==j);n=int(keep.sum());events=int(y[keep].sum())
                if min(n,events,n-events)<10:joint.append({'first':first,'first_cluster':i,'second':second,'second_cluster':j,'suppressed':True});continue
                joint.append({'first':first,'first_cluster':i,'second':second,'second_cluster':j,'patients':n,'SHD_prevalence':float(y[keep].mean())})
    model_summary={'selected_checkpoint_hash_verified':True,'saved_prediction_hash_verified':True,'metadata_outcomes_match_saved_predictions':True,'parameters':complete['trainable_parameters'],'best_epoch':complete['best_epoch'],'overall_validation_AUROC_recomputed':float(roc_auc_score(y,probability)),'overall_validation_Brier_recomputed':float(brier_score_loss(y,probability)),'selection_bias_warning':'Checkpoint and candidate selected on these validation outcomes; descriptive conditional analyses only','groups':resnet}
    for filename,rows in [('SIGNAL_DESCRIPTOR_DIFFERENCES.csv',differences),('MACHINE_MEASUREMENTS.csv',numeric),('SUBGROUP_SHD_ASSOCIATIONS.csv',subgroups),('CROSS_REPRESENTATION_OVERLAP.csv',overlaps),('JOINT_PATTERNS.csv',joint),('RESNET_BY_PATTERN.csv',resnet)]:export_csv(a.output/filename,rows)
    (a.output/'TRAINED_MODEL_AUDIT.json').write_text(json.dumps(model_summary,indent=2));(a.output/'EXPLORATORY_CHARACTERIZATION.json').write_text(json.dumps({'bootstrap_seed':63001,'bootstrap_draws':a.draws,'signal_descriptors':differences,'subgroups':subgroups,'overlap':overlaps,'joint':joint,'warning':'Post hoc analyses; no multiplicity-adjusted subgroup inference; pointwise signal intervals do not account for descriptor selection or representation fitting'},indent=2))
    artifacts=[cache_manifest,Path('reports/stage2_v1/tune_3_complete.json'),predictions,checkpoint,a.run/'FREEZE.json',Path(__file__)]
    private=Path('private/unsupervised_discovery_v1/phase1_audit_v1');private.mkdir(parents=True,exist_ok=True)
    (private/'PROVENANCE.json').write_text(json.dumps({'inputs':{str(p.resolve()):sha(p) for p in artifacts},'aggregate_outputs':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()}},indent=2))
    print('Phase-1 audit and frozen-pattern characterization complete; aggregate outputs only; no training/test access.')
if __name__=='__main__':main()
