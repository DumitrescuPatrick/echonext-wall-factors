"""Aggregate development reporting; individual validation predictions stay private."""
import csv,json,itertools
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from .stage2_train import OUT,PRIVATE,protocol,selected_candidate,sha,dump
from .stage2_metrics import metrics,operating_points,threshold_for,classification,sigmoid_calibration,apply_calibration

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def weighted_core(y,p,w):
    """Bootstrap weights, exact tie groups; returns per-draw AUC/AP/Brier/log loss."""
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),1e-7,1-1e-7);w=np.asarray(w,float)
    n=w.sum(1);pos=w@y;neg=n-pos;order=np.argsort(p,kind='stable')
    _,starts=np.unique(p[order],return_index=True)
    gp=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    gn=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
    auc=(gp*(np.cumsum(gn,axis=1)-gn*.5)).sum(1)/(pos*neg)
    tp=np.cumsum(gp[:,::-1],axis=1);count=np.cumsum((gp+gn)[:,::-1],axis=1)
    precision=np.divide(tp,count,out=np.zeros_like(tp),where=count>0)
    ap=(gp[:,::-1]*precision).sum(1)/pos
    brier=w@((p-y)**2)/n;ll=w@(-(y*np.log(p)+(1-y)*np.log1p(-p)))/n
    return {'auc':auc,'average_precision':ap,'brier':brier,'log_loss':ll}

def bootstrap(y,probs,seed,repeats):
    rng=np.random.default_rng(seed);s=len(probs);sw_rng=np.random.default_rng(seed+1)
    sw=np.vstack([np.bincount(v,minlength=s) for v in sw_rng.integers(0,s,(repeats,s))])/s
    values={k:[] for k in ['auc','average_precision','brier','log_loss']}
    for start in range(0,repeats,64):
        size=min(64,repeats-start);idx=rng.integers(0,len(y),(size,len(y)))
        weights=np.vstack([np.bincount(row,minlength=len(y)) for row in idx])
        scored=[weighted_core(y,p,weights) for p in probs]
        for k in values:values[k].append(np.column_stack([x[k] for x in scored]))
    rows=[]
    for metric,chunks in values.items():
        d=np.concatenate(chunks);points=np.array([weighted_core(y,p,np.ones((1,len(y))))[metric][0] for p in probs])
        pl,ph=np.quantile(d.mean(1),[.025,.975]);cl,ch=np.quantile((d*sw).sum(1),[.025,.975])
        rows.append({'metric':metric,'mean':float(points.mean()),'seed_sd':float(points.std(ddof=1)),'seed_min':float(points.min()),'seed_max':float(points.max()),'patient_low':float(pl),'patient_high':float(ph),'crossed_low':float(cl),'crossed_high':float(ch),'n':len(y),'training_seeds':s,'bootstrap_repeats':repeats})
    return rows

def coarsened_bins(y,p):
    group=np.minimum((p*10).astype(int),9);bins=[];pending=[]
    for j in range(10):
        pending.extend(np.flatnonzero(group==j).tolist())
        if len(pending)>=20 and y[pending].sum()>=5 and (1-y[pending]).sum()>=5:bins.append(pending);pending=[]
    if pending:
        if not bins:raise ValueError('Calibration group support insufficient')
        bins[-1].extend(pending)
    return [{'bin':j,'n':len(idx),'positive':int(y[idx].sum()),'mean_predicted':float(p[idx].mean()),'observed':float(y[idx].mean())} for j,idx in enumerate(bins)]

def bootstrap_operating_points(y,probs,c):
    n=len(y);seeds=len(probs);repeats=c['bootstrap_repeats'];rng=np.random.default_rng(c['bootstrap_seed'])
    swrng=np.random.default_rng(c['bootstrap_seed']+1)
    sw=np.vstack([np.bincount(v,minlength=seeds) for v in swrng.integers(0,seeds,(repeats,seeds))])/seeds
    keys=[(kind,target) for kind in ['sensitivity','specificity'] for target in [.9,.95]]
    decisions={key:np.stack([p>=threshold_for(y,p,*key) for p in probs]) for key in keys}
    draws={(key,metric):[] for key in keys for metric in ['sensitivity','specificity']}
    for start in range(0,repeats,64):
        size=min(64,repeats-start);ix=rng.integers(0,n,(size,n));w=np.vstack([np.bincount(r,minlength=n) for r in ix]);pos=w@y;neg=w@(1-y)
        for key,d in decisions.items():
            draws[key,'sensitivity'].append((w@(d*y).T)/pos[:,None])
            draws[key,'specificity'].append((w@((~d)*(1-y)).T)/neg[:,None])
    rows=[]
    for key,d in decisions.items():
        for metric in ['sensitivity','specificity']:
            values=d[:,y==1].mean(1) if metric=='sensitivity' else (~d[:,y==0]).mean(1)
            bs=np.concatenate(draws[key,metric]);pl,ph=np.quantile(bs.mean(1),[.025,.975]);cl,ch=np.quantile((bs*sw).sum(1),[.025,.975])
            rows.append({'target_kind':key[0],'target':key[1],'metric':metric,'mean':float(values.mean()),'seed_sd':float(values.std(ddof=1)),'patient_low':float(pl),'patient_high':float(ph),'crossed_low':float(cl),'crossed_high':float(ch),'interval_limit':'conditional on validation-selected thresholds/models; no selection refitting'})
    return rows

def report():
    c=protocol();analysis=OUT/'report_source_sha256.json'
    if analysis.exists() and json.loads(analysis.read_text())!={'stage2_report.py':sha(__file__)}:raise ValueError('Report source changed')
    dump(analysis,{'stage2_report.py':sha(__file__)})
    seed_rows=[];op_rows=[];cross_rows=[];cal_rows=[];resources=[];all_probs=[];rules=[];y=None
    for seed in c['final_seeds']:
        receipt=json.loads((OUT/f'seed_{seed}_complete.json').read_text());resources.append(receipt)
        file=PRIVATE/f'seed_{seed}/validation_predictions.npz'
        if sha(file)!=receipt['private_prediction_sha256']:raise ValueError('Validation checkpoint changed')
        with np.load(file) as z:
            if y is None:y=z['y'].copy()
            elif not np.array_equal(y,z['y']):raise ValueError('Validation patients differ')
            logits=z['logits'].copy();p=expit(logits)
        seed_rows.append({'seed':seed,**metrics(y,p)});all_probs.append(p)
        ops=operating_points(y,p)
        op_rows.extend({'seed':seed,**r} for r in ops)
        folds=StratifiedKFold(5,shuffle=True,random_state=56001);pc=np.full(len(y),np.nan)
        decisions={(kind,target):np.zeros(len(y),bool) for kind in ['sensitivity','specificity'] for target in [.9,.95]}
        for tr,te in folds.split(p,y):
            cal=sigmoid_calibration(y[tr],logits[tr]);pc[te]=apply_calibration(logits[te],cal)
            for (kind,target),decision in decisions.items():decision[te]=p[te]>=threshold_for(y[tr],p[tr],kind,target)
        for (kind,target),decision in decisions.items():
            cross_rows.append({'seed':seed,'target_kind':kind,'target':target,**classification(y,decision.astype(float),.5),'method':'five-fold cross-fitted cutoff only; selected risk model already used all validation labels'})
        cal_rows.append({'seed':seed,'method':'five-fold cross-fitted validation sigmoid; model-selection optimism remains',**metrics(y,pc)})
        rules.append({'seed':seed,'native_operating_points':ops,'full_validation_sigmoid':sigmoid_calibration(y,logits)})
    probabilities=np.stack(all_probs);summary=bootstrap(y,probabilities,c['bootstrap_seed'],c['bootstrap_repeats'])
    for name in ['calibration_intercept','calibration_slope','ece_10','mean_risk','risk_variance']:
        v=np.array([r[name] for r in seed_rows]);summary.append({'metric':name,'mean':float(v.mean()),'seed_sd':float(v.std(ddof=1)),'seed_min':float(v.min()),'seed_max':float(v.max()),'patient_low':'','patient_high':'','crossed_low':'','crossed_high':'','n':len(y),'training_seeds':5,'bootstrap_repeats':0})
    write_csv(OUT/'validation_metrics_by_seed.csv',seed_rows);write_csv(OUT/'validation_metrics_summary.csv',summary)
    write_csv(OUT/'validation_operating_points.csv',op_rows);write_csv(OUT/'validation_crossfit_operating_points.csv',cross_rows);write_csv(OUT/'validation_crossfit_calibration.csv',cal_rows)
    op_summary=bootstrap_operating_points(y,probabilities,c)
    write_csv(OUT/'validation_operating_point_summary.csv',op_summary)
    dump(PRIVATE/'final_operating_rules.json',rules)
    ensemble=probabilities.mean(0);write_csv(OUT/'validation_secondary_ensemble.csv',[{'estimand':'uniform five-seed probability ensemble',**metrics(y,ensemble)}])
    bins=coarsened_bins(y,ensemble);write_csv(OUT/'validation_ensemble_calibration_bins.csv',bins)
    tuning=[json.loads((OUT/f'tune_{i}_complete.json').read_text()) for i in range(6)]
    resources=tuning+resources
    write_csv(OUT/'all_runs_resources.csv',[{k:r[k] for k in ['run','candidate_index','seed','epochs_completed','best_epoch','trainable_parameters','total_epoch_seconds','total_cpu_seconds','peak_rss_bytes']} for r in resources])
    write_csv(OUT/'tuning_comparison.csv',[{'candidate_index':r['candidate_index'],**r['configuration'],'epochs_completed':r['epochs_completed'],'best_epoch':r['best_epoch'],**r['best_metrics']} for r in tuning])
    benchmark=[{'source':'Hughes et al. NEJM AI 2026','population':'published official held-out test','auc':.820,'auc_low':.809,'auc_high':.831,'sensitivity':.701,'specificity':.779,'comparison':'context only; published test versus our model-selected validation, waveform/tabular and multitask/architecture differences; no superiority inference','doi':'10.1056/AIdbp2500516'}]
    write_csv(OUT/'published_benchmark_context.csv',benchmark)
    # Aggregate figure inputs only: no individual thresholds or fine ROC export.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    grid=np.linspace(0,1,101)
    for p in probabilities:
        fpr,tpr,_=roc_curve(y,p);axes[0].plot(grid,np.interp(grid,fpr,tpr),alpha=.45)
    axes[0].plot([0,1],[0,1],'k:');axes[0].set(xlabel='False-positive fraction',ylabel='True-positive fraction',title='Five single-seed validation curves')
    axes[1].plot([0,1],[0,1],'k:');axes[1].plot([r['mean_predicted'] for r in bins],[r['observed'] for r in bins],'o-');axes[1].set(xlabel='Mean predicted probability',ylabel='Observed SHD fraction',title='Secondary seed-ensemble calibration')
    fig.suptitle('Development validation: selection and stopping used these patients')
    fig.savefig(OUT/'validation_curves.pdf');fig.savefig(OUT/'validation_curves.png',dpi=300);plt.close(fig)
    chosen=selected_candidate(c);by={r['metric']:r for r in summary}
    text=['# EchoNext Stage 2 — ResNet validation baseline','',f'Trained on all 72,475 official-training ECGs. Model selection, stopping and operating points use 4,626 official-validation patients (1,990 SHD outcomes). Test outcomes/performance remain sealed. Selected candidate {chosen["candidate_index"]}: '+str(chosen['configuration'])+'. Five final training seeds; no best-seed selection.','', '| Metric | Mean single-seed value | Seed SD | Crossed 95% interval |','|---|---:|---:|---|']
    for name in ['auc','average_precision','brier','log_loss']:
        r=by[name];text.append(f'| {dict(auc="ROC AUC",average_precision="Average precision",brier="Brier score",log_loss="Log loss")[name]} | {r["mean"]:.5f} | {r["seed_sd"]:.5f} | [{r["crossed_low"]:.5f}, {r["crossed_high"]:.5f}] |')
    text+=['','| Validation-selected target | Sensitivity, % [crossed 95% interval] | Specificity, % [crossed 95% interval] |','|---|---:|---:|']
    for kind in ['sensitivity','specificity']:
        for target in [.9,.95]:
            ss=[r for r in op_summary if r['target_kind']==kind and r['target']==target]
            parts={r['metric']:f"{100*r['mean']:.2f} [{100*r['crossed_low']:.2f}, {100*r['crossed_high']:.2f}]" for r in ss}
            text.append(f"| {kind.capitalize()} ≥{target:.0%} | {parts['sensitivity']} | {parts['specificity']} |")
    text+=['','| Native calibration descriptor | Five-seed mean | Seed SD |','|---|---:|---:|']
    for name,label in [('calibration_intercept','Calibration intercept'),('calibration_slope','Calibration slope'),('ece_10','10-bin ECE')]:
        r=by[name];text.append(f"| {label} | {r['mean']:.5f} | {r['seed_sd']:.5f} |")
    text+=['','Calibration slope/intercept/ECE are descriptive with seed variability in validation_metrics_summary.csv. All sensitivity/specificity pairs and thresholds are in validation_operating_points.csv; those are apparent estimates at cutoffs chosen on these same predictions. Cross-fitted cutoff/calibration estimates are provided separately, but do not remove model-selection/stopping optimism. The uniform probability ensemble is a secondary, different estimand.','', 'Published EchoNext-Mini AUROC 0.820 (0.809–0.831) is a contextual test benchmark, not a paired validation comparator. Its released inference interface uses seven tabular predictors and 12 outputs; this baseline is waveform-only, single-target, patient-weighted and much smaller, with GroupNorm and a different architecture/optimization. No claim of matching or outperforming the published model is justified from these development results.','', 'Inverse-count training weights equalize patient influence but do not make multiple training ECGs equivalent to the latest-ECG sampling used in validation/test. This residual distribution difference remains.','', 'Binary target completeness does not establish complete component ascertainment: numeric TR measurements are often absent despite populated flags. Outcome linkage and upstream normalization retain provider-dependent limitations; no new clinical adjudication was performed.','', 'All confidence intervals condition on the selected/fitted models and omit rerunning tuning, early stopping, patient sampling and preprocessing provenance. Selection used these validation labels, so this is not independent confirmation. Future confirmatory evidence remains the single sealed test batch.','', 'Resource peak_rss_bytes is the worker-process lifetime maximum up to each completed run, not an independently reset per-fit peak. CPU seconds use per-fit deltas; wall time is reported separately.','', 'The cache amendment changed storage only; every cached development float32 value was verified bitwise against original conversion. Interrupted partial-epoch computation before the cache switch is not recoverable from checkpoint accounting, so reported tuning CPU time is a lower bound for full operational expenditure. Future strict-compute comparisons must report completed optimization work and all overhead separately. No test cache or test performance was created.','', 'Patient-level validation predictions, checkpoints and per-seed calibration objects are private and excluded from the aggregate deliverable.']
    (OUT/'VALIDATION_REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'report_complete':True,'n_validation':len(y),'five_seed_mean_auc':by['auc']['mean'],'test_evaluation':False}))

if __name__=='__main__':report()
