"""Aggregate tables and figures; no patient-level reads."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('reports/covariance_validation_v1')

def table(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    fusion=json.loads((ROOT/'FUSION_DIAGNOSTIC.json').read_text());stress=json.loads((ROOT/'STRESS_TESTS.json').read_text());lead=json.loads((ROOT/'LEAD_I_III_SENSITIVITY.json').read_text());assoc=json.loads(Path('reports/unsupervised_discovery_v1/five_questions_v1/COMPONENT_ADJUSTMENT.json').read_text())
    table('TABLE_1_FROZEN_DEFINITIONS.csv',[{'Object':'Ordinary covariance','Definition':'svec(0.9 C/e +0.1I -I) plus log e','Dimensions':79,'Status':'Continuous primary prediction feature family'}, {'Object':'Log-SPD','Definition':'svec(log(0.9 C/e+0.1I)) plus log e','Dimensions':79,'Status':'Secondary equal-information coordinate control'},{'Object':'Original cluster','Definition':'Original seed62001 log-SPD scaler/KMeans and unchanged patient assignments','Dimensions':'K=2','Status':'Frozen association/explanation object; not ordinary-covariance re-clustering'}])
    rows=[]
    for r in assoc['results']:
        if r['representation']=='spd' and r['adjustment'] in ['flexible','flexible_plus_energy']:rows.append({'Endpoint':r['endpoint'],'Adjustment':r['adjustment'],'Odds ratio':r['odds_ratio'],'95% CI lower':r['wald_95_ci'][0],'95% CI upper':r['wald_95_ci'][1],'Status':'Earlier exploratory available-variable association'})
    for r in lead['results']:
        if r['adjustment']=='same_plus_I_III_spline':rows.append({'Endpoint':r['endpoint'],'Adjustment':'Flexible plus I-III correlation spline','Odds ratio':r['odds_ratio'],'95% CI lower':r['wald_95_ci'][0],'95% CI upper':r['wald_95_ci'][1],'Status':'Additional defining-feature conditioning; not actual axis adjustment'})
    table('TABLE_2_CLINICAL_ASSOCIATIONS.csv',rows)
    table('TABLE_3_FUSION_DIAGNOSTIC.csv',[{'Model':r['model'],'AUROC':r['metrics']['auc'],'Average precision':r['metrics']['average_precision'],'Brier':r['metrics']['brier'],'Calibration intercept':r['metrics']['calibration_intercept'],'Calibration slope':r['metrics']['calibration_slope'],'ECE (10 bins)':r['metrics']['ece_10'],'Status':'Upstream-selection-contaminated exploratory fusion CV'} for r in fusion['models']])
    rows=[]
    for c in fusion['contrasts']:
        for name,m in c['metric_differences'].items():rows.append({'Challenger':c['challenger'],'Reference':c['reference'],'Metric':name,'Difference':m['difference'],'Paired 95% CI lower':m['paired_95_ci'][0],'Paired 95% CI upper':m['paired_95_ci'][1],'Status':'Diagnostic only; no correction for base model selection bias'})
    table('TABLE_4_PAIRED_DIAGNOSTIC_DIFFERENCES.csv',rows)
    table('TABLE_5_STRESS_SENSITIVITY.csv',[{'Scenario':r['scenario'],'Changed assignments (%)':100*r['assignment_change_fraction'],'Wilson 95% CI lower (%)':100*r['assignment_change_Wilson_95_ci'][0],'Wilson 95% CI upper (%)':100*r['assignment_change_Wilson_95_ci'][1],'ARI':r['ARI'],'Status':'Normalized-space stress; no clinical error prevalence inference'} for r in stress['results']])
    table('TABLE_6_DECISION_CURVE_DIAGNOSTIC.csv',[{'Challenger':c['challenger'],'Reference':c['reference'],'Risk threshold':r['threshold'],'Net-benefit difference':r['difference'],'Paired 95% CI lower':r['paired_95_ci'][0],'Paired 95% CI upper':r['paired_95_ci'][1],'Status':'Potential utility under assumed ratio, selection-contaminated diagnostic'} for c in fusion['contrasts'] for r in c['net_benefit_difference']])
    table('TABLE_7_MODEL_COMPLEXITY.csv',fusion['complexity_and_choices']);table('TABLE_8_FEATURE_CPU_MICROBENCHMARK.csv',stress['CPU_microbenchmark'])
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(figures/(name+'.png'),dpi=300);plt.savefig(figures/(name+'.pdf'));plt.close()
    r=[r for r in stress['results'] if r['scenario']!='baseline'];plt.figure(figsize=(10,6));point=np.array([100*x['assignment_change_fraction'] for x in r]);low=np.array([100*x['assignment_change_Wilson_95_ci'][0] for x in r]);high=np.array([100*x['assignment_change_Wilson_95_ci'][1] for x in r]);plt.barh(range(len(r)),point,xerr=[np.maximum(0,point-low),np.maximum(0,high-point)],color='#39748c');plt.yticks(range(len(r)),[x['scenario'].replace('_',' ') for x in r]);plt.xlabel('Changed frozen assignments (%)');plt.title('Normalized-space sensitivity; lead reversal scenarios are proxies');save('figure_1_frozen_group_stress')
    comparisons=fusion['contrasts'][:3];plt.figure(figsize=(9,4))
    for j,c in enumerate(comparisons):
        m=c['metric_differences']['AUROC'];d=m['difference'];low,high=m['paired_95_ci'];plt.errorbar(d,j,xerr=[[d-low],[high-d]],fmt='o',capsize=4)
    plt.yticks(range(3),['Ordinary covariance increment','Log-SPD increment','Log-SPD − ordinary']);plt.axvline(0,ls='--',color='.5');plt.xlabel('AUROC difference (paired 95% conditional interval)');plt.title('Fusion diagnostic: upstream ResNet selection bias remains');save('figure_2_fusion_diagnostic')
    plt.figure(figsize=(8,4))
    for c,label,color in [(fusion['contrasts'][0],'Ordinary covariance increment','#39748c'),(fusion['contrasts'][1],'Log-SPD increment','#b55a20')]:
        v=c['net_benefit_difference'];t=np.array([r['threshold'] for r in v]);d=np.array([r['difference'] for r in v]);lo=np.array([r['paired_95_ci'][0] for r in v]);hi=np.array([r['paired_95_ci'][1] for r in v]);plt.plot(t,1000*d,label=label,color=color);plt.fill_between(t,1000*lo,1000*hi,alpha=.12,color=color)
    plt.axhline(0,ls='--',color='.5');plt.axvline(.2,ls=':',color='.5');plt.xlabel('Assumed risk threshold');plt.ylabel('Δ net benefit × 1000');plt.title('Potential utility diagnostic; no validated clinical threshold');plt.legend();save('figure_3_potential_net_benefit')
    (ROOT/'FIGURE_CAPTIONS.md').write_text('''# Figure captions

1. Fixed original log-SPD group assignments under deliberately applied normalized-signal transformations. Wilson intervals describe patient assignment-change proportions. Labels do not estimate real artifact prevalence. The RA/LA and channel-swap transformations are post-normalization proxies, not authenticated electrode simulations. Global polarity/shared time permutation invariance reflects covariance mathematics, not comprehensive clinical robustness.

2. Nested five-outer/three-inner fusion diagnostic on 4,626 development validation patients. Base ResNet checkpoint selection used all these outcomes historically. Fusion folds are patient separated; upstream model selection is not. Intervals condition on fitted fusion models and do not correct this bias. No independent ResNet increment or formal equivalence claim is supported.

3. Paired pointwise intervals for net-benefit increments over the full ResNet + clinical/simple reference under assumed threshold harm/benefit ratios. Values scaled by 1000 are utility-equivalent changes, not observed patients helped. The 20% line is provisional and requires clinical ratification. No threshold is selected from these curves, and current diagnostics do not establish real-world clinical utility.
''');print('Eight aggregate tables and three figure sets saved.')
if __name__=='__main__':main()
