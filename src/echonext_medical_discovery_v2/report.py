"""Aggregate-only evidence tables and plots for the medical discovery report."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .latent import REPORT

def table(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (REPORT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    discovery=json.loads((REPORT/'FACTOR_DISCOVERY.json').read_text());a=json.loads((REPORT/'FACTOR_ASSOCIATIONS.json').read_text());d=json.loads((REPORT/'FACTOR_DIAGNOSTICS.json').read_text());reference=json.loads((REPORT/'COMPONENT_REFERENCE_SENSITIVITY.json').read_text());unit=json.loads((REPORT/'UNIT_INPUT_TRANSPORT_SENSITIVITY.json').read_text());rows=[]
    for r in a['associations']:
        row={'Representation':r['space'],'Factor':r['factor'],'Endpoint':r['endpoint'],'Family':r['family'],'Patients':r['patients'],'Events':r['events'],'Status':r['status']}
        if r['status']=='estimated':row.update({'OR per training SD':r['odds_ratio'],'95% CI lower':r['wald_95_ci'][0],'95% CI upper':r['wald_95_ci'][1],'Exploratory BH q':r['BH_q'],'Information condition number':r['hessian_condition'],'Numerical review required':r['numerical_review_required']})
        rows.append(row)
    table('ALL_FACTOR_ASSOCIATIONS.csv',rows);table('FACTOR_TRAIT_CORRELATIONS.csv',a['trait_correlations']);table('FACTOR_STRESS_DIAGNOSTICS.csv',d['stress'])
    rows=[]
    for space,s in discovery['spaces'].items():
        for r in s['stability']['factors']:rows.append({'Representation':space,**r,'PCA explained variance ratio':s['explained_variance_ratio'][r['factor']-1],'Frozen-neural seed replication':False})
    table('FACTOR_STABILITY.csv',rows);table('CONTINUOUS_FACTOR_REDUNDANCY.csv',d['strong_cross_factor_correlations']);table('CLUSTER_PROFILES_DESCRIPTIVE.csv',[{k:v for k,v in r.items()} for r in a['cluster_profiles_descriptive']])
    for filename,values in [('COMPONENT_REFERENCE_ASSOCIATIONS.csv',reference['factor_associations_BH48']),('UNIT_INPUT_ECHO_SENSITIVITY.csv',unit['source_associations_BH16']),('UNIT_INPUT_PTB_CONSTRUCT_SENSITIVITY.csv',unit['PTB_ECG_associations_BH72'])]:
        rows=[]
        for r in values:
            row={k:v for k,v in r.items() if k not in ['wald_95_ci']};ci=r.get('wald_95_ci',['not estimable','not estimable']);row.update({'95% CI lower':ci[0],'95% CI upper':ci[1]});rows.append(row)
        table(filename,rows)
    hypotheses=[{'Priority':1,'Hypothesis':'Wall-thickness-associated supervised neural morphology factor','Candidate':'pooled PC2','Reproducibility':'Bootstrap matched-score lower percentile 0.9997; one source neural seed only','Clinical information':'OR1.48; after nonlinear component reference OR1.36; selected-validation association only','Interpretability':'No validated QRS/ST-T or axis identity; unit-input PTB HYP association reverses direction','Artifact concerns':'Global-gain/noise stable; shift correlation0.908, per-lead-unit-SD correlation0.778','External feasibility':'Requires calibrated ECG/echo or authenticated source preprocessing and new encoder training','Decision':'Highest-priority hypothesis, not a validated biomarker'},
    {'Priority':2,'Hypothesis':'Temporal-activation wall-thickness association beyond average-rate descriptors','Candidate':'temporal PC5','Reproducibility':'Bootstrap lower0.995; not repeated-neural-seed stability','Clinical information':'OR1.29; nonlinear-reference OR1.24; average rate correlation-0.441','Interpretability':'Coarse full-record activation statistics; no beat/QRS/ST-T assignment','Artifact concerns':'100ms shift correlation0.801; beat-phase/rhythm explanation unresolved','External feasibility':'Needs validated fiducials/RR variability and matched encoders','Decision':'Secondary hypothesis; conventional rhythm explanation must be tested'},
    {'Priority':3,'Hypothesis':'Residualization induces scale dependence','Candidate':'residual PC2/PC5','Reproducibility':'PCA directions stable, but counterfactual gain changes scores','Clinical information':'Residual PC2 associations disappear under component reference; PC5 LVEF remains exploratory','Interpretability':'Residual variance is not automatically independent physiology','Artifact concerns':'Gain2 creates RMS change0.893SD in residual PC2 despite near-zero pooled PC2 change','External feasibility':'Fix/audit residualizer before biological interpretation','Decision':'Methodological finding; do not promote to biomarker'},
    {'Priority':4,'Hypothesis':'Higher pooled/temporal factors mark reduced LVEF','Candidate':'pooled PC3/PC6, temporal PC8','Reproducibility':'Conditional PCA stable','Clinical information':'Initial associations attenuate after nonlinear conventional/covariance reference','Interpretability':'Rate/phase and known-information explanations remain','Artifact concerns':'100ms shift correlations approximately0.49-0.71 for highlighted factors','External feasibility':'Improve invariance and rhythm controls first','Decision':'Deprioritized as new independent medical findings'},
    {'Priority':5,'Hypothesis':'Six categorical disease subtypes','Candidate':'all three KMeans factor spaces','Reproducibility':'Fixed-feature seed ARI high but silhouettes0.10-0.15; selected K at grid maximum','Clinical information':'Descriptive groups reflect supervised reference; no subtype proof','Interpretability':'Continuous axes can yield stable arbitrary partitions','Artifact concerns':'Recording phase and source model selection unresolved','External feasibility':'No direct full-factor replication yet','Decision':'Unsupported categorical disease claim'},
    {'Priority':'Future','Hypothesis':'Label-free QRS/ST-T and whole-record phenotypes','Candidate':'compact VAE/SimCLR; validated segment encoders','Reproducibility':'Synthetic tests only; clinical GPU fitting not run','Clinical information':'No measured VAE/SimCLR clinical result','Interpretability':'Requires validated fiducials and axis evaluation','Artifact concerns':'Input invariance and domain mismatch must be verified','External feasibility':'MPS runner and synthetic smoke prepared; current MPS unavailable','Decision':'Planned experiment, not a discovery'}]
    table('RANKED_HYPOTHESES.csv',hypotheses)
    figs=REPORT/'figures';figs.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(figs/(name+'.png'),dpi=300);plt.savefig(figs/(name+'.pdf'));plt.savefig(figs/(name+'.svg'));plt.close()
    plt.figure(figsize=(7,4))
    for name,s in discovery['spaces'].items():plt.plot(range(1,9),np.cumsum(s['explained_variance_ratio']),'o-',label=name.replace('_',' '))
    plt.xlabel('Fixed PCA factor count');plt.ylabel('Cumulative representation variance');plt.ylim(0,1);plt.legend();plt.title('Frozen supervised ResNet representations');save('figure_1_factor_variance')
    fig,axes=plt.subplots(1,2,figsize=(12,9),sharey=True);order=[(s,j) for s in ['pooled','residual_pooled','temporal'] for j in range(1,9)]
    for ax,endpoint,title in zip(axes,['lvef_lte_45_flag','lvwt_gte_13_flag'],['LVEF ≤45%','Wall thickness ≥13 mm']):
        for i,(space,j) in enumerate(order):
            r=next(r for r in a['associations'] if r['space']==space and r['factor']==j and r['endpoint']==endpoint)
            if r['status']!='estimated':continue
            low,high=r['wald_95_ci'];ax.errorbar(r['odds_ratio'],i,xerr=[[r['odds_ratio']-low],[high-r['odds_ratio']]],fmt='o',capsize=2,color='#28758a' if r['BH_q']<.05 else '.65')
        ax.set_xscale('log');ax.axvline(1,color='.5',ls='--');ax.set_title(title);ax.set_xlabel('Conditional OR / training SD, pointwise 95% CI')
    axes[0].set_yticks(range(24),[f'{s.replace("_"," ")} PC{j}' for s,j in order],fontsize=8);fig.suptitle('Exploratory associations: supervised source and validation-selection bias remain');save('figure_2_all_primary_associations')
    plt.figure(figsize=(9,7));i=0;labels=[]
    for space,s in discovery['spaces'].items():
        for r in s['stability']['factors']:
            mid=r['matched_monitor_correlation_median'];low=r['matched_monitor_correlation_2.5pct'];high=r['matched_monitor_correlation_97.5pct'];plt.errorbar(mid,i,xerr=[[mid-low],[high-mid]],fmt='o',capsize=2);labels.append(space.replace('_',' ')+f' PC{r["factor"]}');i+=1
    plt.yticks(range(24),labels,fontsize=8);plt.xlim(.7,1.005);plt.xlabel('Bootstrap matched monitoring-score correlation');plt.title('Conditional factor stability; no neural retraining');save('figure_3_factor_stability')
    scenarios=['global_gain_2','per_lead_unit_SD','RA_LA_normalized_proxy','independent_noise_5pct','synchronized_reflect_shift_25'];plt.figure(figsize=(10,4))
    for shift,(space,j,label) in zip([-.18,.18],[('pooled',2,'Pooled PC2'),('residual_pooled',2,'Residual PC2')]):
        values=[next(r['RMS_factor_change_in_training_SD'] for r in d['stress'] if r['space']==space and r['factor']==j and r['scenario']==s) for s in scenarios];plt.bar(np.arange(5)+shift,values,width=.36,label=label)
    plt.xticks(range(5),['Global gain ×2','Lead SD normalization','RA/LA proxy','5% noise','100ms common shift'],rotation=10);plt.ylabel('RMS score change / training SD');plt.legend();plt.title('Residualization can introduce counterfactual scale sensitivity');save('figure_4_residualization_stress')
    selected=[r for r in unit['PTB_ECG_associations_BH72'] if r['factor']==2 and r['adjustment']=='complete_axis'];plt.figure(figsize=(8,4))
    for i,r in enumerate(selected):
        low,high=r['wald_95_ci'];plt.errorbar(r['odds_ratio'],i,xerr=[[r['odds_ratio']-low],[high-r['odds_ratio']]],fmt='o',capsize=4)
    plt.yticks(range(3),[r['endpoint'] for r in selected]);plt.axvline(1,color='.5',ls='--');plt.xlabel('Conditional OR per native-source score SD (95% CI)');plt.title('Altered-input PTB PC2 sensitivity: not native-factor validation');save('figure_5_altered_input_PTB')
    (REPORT/'FIGURE_CAPTIONS.md').write_text('''# Figure captions

1. PCA was fitted on original fitting patients without echo targets; the underlying ResNet was supervised and selected using validation outcomes. Variance is neuronal-feature variance, not independent clinical information.

2. All 48 prespecified primary factor/endpoint associations, with available-variable, RMS, I–III proxy and frozen-score adjustment. Pointwise model-based intervals do not cover source network selection/training uncertainty. Blue denotes exploratory BH q<0.05; no novelty or independent discovery is inferred. Risk-dominated PC1 effects are not promoted to biomarkers.

3. One hundred patient-bootstrap factor refits with matching on unchanged monitoring patients. PCA directions can rotate. These are frozen-feature resampling results, not repeated neural seeds, repeat ECG acquisitions or external replication.

4. Fixed normalized-space interventions on 512 outcome-blind validation patients. Machine measurements were held fixed; this is not a full clinical reacquisition simulation. Residualization recomputed derived nuisance features and can induce scale dependence. Lead swaps are proxies; no real reversal prevalence is inferred.

5. Per-record unit-SD input fed through original frozen native-trained weights and pooled PCA. The input change alters the representation, so this does not validate the native EchoNext PC2. Complete recorded-axis PTB subset; coarse axis and selection limitations remain. PTB labels are ECG interpretations, not echo outcomes. All altered-input association tables and separate exploratory multiplicity families are provided.
''');print('Evidence/ranking tables and five figure sets saved.')
if __name__=='__main__':main()
