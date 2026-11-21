"""Aggregate-only human-readable tables and figures for the five questions."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('reports/unsupervised_discovery_v1/five_questions_v1')
NAMES={'conventional_ecg':'ECG intervals/rates','clinical_ecg':'ECG + age/sex','clinical_ecg_simple':'ECG + age/sex + simple waveform','clinical_ecg_spd':'ECG + age/sex + log-SPD','clinical_ecg_simple_spd':'ECG + age/sex + simple + log-SPD','clinical_ecg_hilbert':'ECG + age/sex + RFF waveform'}

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    associations=json.loads((ROOT/'COMPONENT_ADJUSTMENT.json').read_text())['results'];prediction=json.loads((ROOT/'PREDICTION_RESULTS.json').read_text());control=json.loads((ROOT/'EUCLIDEAN_CONTROL.json').read_text());rules=json.loads((ROOT/'INTERPRETABLE_RULES.json').read_text())['rules']
    rows=[]
    for r in associations:
        low,high=r['wald_95_ci'];rows.append({'Representation':r['representation'],'Outcome':'LVEF ≤45%' if r['endpoint'].startswith('lvef') else 'LV wall thickness ≥13 mm','Adjustment':r['adjustment'],'Odds ratio':r['odds_ratio'],'95% CI lower':low,'95% CI upper':high,'Primary BH q value':r.get('primary_BH_q','Not primary'),'Patients':r['patients'],'Events':r['events']})
    table(ROOT/'TABLE_1_ADJUSTED_COMPONENT_ASSOCIATIONS.csv',rows)
    rows=[]
    for r in prediction['models']:
        m=r['calibrated'];ci=r['calibrated_patient_bootstrap_95_ci']['AUROC'];rows.append({'Model family':r['family'],'Inputs':NAMES[r['features']],'Validation patients':m['n'],'AUROC':m['auc'],'AUROC 95% CI lower':ci[0],'AUROC 95% CI upper':ci[1],'Average precision':m['average_precision'],'Brier score':m['brier'],'Calibration intercept':m['calibration_intercept'],'Calibration slope':m['calibration_slope'],'ECE (10 bins)':m['ece_10'],'Native Brier':r['native']['brier']})
    table(ROOT/'TABLE_2_PREDICTIVE_MODELS.csv',rows)
    rows=[]
    for r in prediction['contrasts']:
        m=r['metrics']['AUROC'];rows.append({'Model family':r['family'],'Challenger':NAMES[r['challenger']],'Reference':NAMES[r['reference']],'Primary comparison':r['primary'],'AUROC difference':m['difference'],'Paired 95% CI lower':m['paired_95_ci'][0],'Paired 95% CI upper':m['paired_95_ci'][1],'Paired 97.5% CI lower':m['paired_97.5_ci'][0],'Paired 97.5% CI upper':m['paired_97.5_ci'][1],'Brier difference':r['metrics']['Brier']['difference']})
    table(ROOT/'TABLE_3_INCREMENTAL_PREDICTION.csv',rows)
    rows=[]
    for r in control['results']:
        m=r['logSPD_minus_Euclidean']['AUROC'];rows.append({'Model family':r['family'],'Ordinary covariance AUROC':r['Euclidean_AUROC'],'Log-SPD AUROC':r['logSPD_AUROC'],'AUROC difference':m['difference'],'Paired 95% CI lower':m['paired_95_ci'][0],'Paired 95% CI upper':m['paired_95_ci'][1]})
    table(ROOT/'TABLE_4_EUCLIDEAN_CONTROL.csv',rows);table(ROOT/'TABLE_5_GROUP_EXPLANATION.csv',[{'Representation':r['representation'],'Approximate rule features':'; '.join(dict.fromkeys(r['features_used'])),'Training-monitor agreement':r['monitor_accuracy'],'Validation agreement':r['validation_accuracy'],'Validation balanced accuracy':r['validation_balanced_accuracy']} for r in rules])
    features=list(csv.DictReader(Path('reports/unsupervised_discovery_v1/phase1_audit_v1/SIGNAL_DESCRIPTOR_DIFFERENCES.csv').open()));selected=[]
    for method in ['spd','waveform','hilbert']:
        candidates=[r for r in features if r['representation']==method and r['standardized_difference'] not in ('','None')];candidates.sort(key=lambda r:abs(float(r['standardized_difference'])),reverse=True);selected.extend(candidates[:10])
    table(ROOT/'TABLE_6_TOP_SIGNAL_DESCRIPTORS.csv',selected)
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=300);plt.savefig(figdir/(name+'.pdf'));plt.close()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,endpoint,title in zip(axes,['lvef_lte_45_flag','lvwt_gte_13_flag'],['Reduced LVEF','Increased wall thickness']):
        for j,method in enumerate(['spd','waveform','hilbert']):
            for shift,spec,color in [(-.08,'flexible','#006080'),(.08,'flexible_plus_energy','#b55a20')]:
                r=next(r for r in associations if r['endpoint']==endpoint and r['representation']==method and r['adjustment']==spec);low,high=r['wald_95_ci'];ax.errorbar(r['odds_ratio'],j+shift,xerr=[[r['odds_ratio']-low],[high-r['odds_ratio']]],fmt='o',color=color,capsize=3,label=spec if j==0 else None)
        ax.set_xscale('log');ax.axvline(1,color='.5',linestyle='--');ax.set_yticks(range(3),['Covariance','Simple waveform','Hilbert/RFF']);ax.set_title(title);ax.set_xlabel('Adjusted odds ratio (pointwise 95% CI)')
    axes[1].legend(fontsize=8);save('figure_1_component_adjustment')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    featureorder=['clinical_ecg','clinical_ecg_simple','clinical_ecg_spd','clinical_ecg_simple_spd','clinical_ecg_hilbert']
    for ax,family in zip(axes,['logistic','boosting']):
        for j,name in enumerate(featureorder):
            r=next(r for r in prediction['models'] if r['family']==family and r['features']==name);auc=r['calibrated']['auc'];low,high=r['calibrated_patient_bootstrap_95_ci']['AUROC'];ax.errorbar(auc,j,xerr=[[auc-low],[high-auc]],fmt='o',capsize=3)
        ax.set_yticks(range(5),['ECG + age/sex','+ simple waveform','+ log-SPD','+ simple + log-SPD','+ RFF waveform']);ax.set_xlim(.70,.84);ax.set_title(family);ax.set_xlabel('Validation AUROC (95% conditional patient CI)')
    save('figure_2_matched_models')
    plt.figure(figsize=(8,4))
    for j,r in enumerate(control['results']):
        m=r['logSPD_minus_Euclidean']['AUROC'];d=m['difference'];low,high=m['paired_95_ci'];plt.errorbar(d,j,xerr=[[d-low],[high-d]],fmt='o',capsize=4)
    plt.yticks(range(2),['Logistic regression','Nonlinear boosting']);plt.axvline(0,color='.5',linestyle='--');plt.xlabel('Log-SPD minus ordinary covariance AUROC (paired 95% CI)');plt.title('Same covariance information and dimension');save('figure_3_geometry_specificity_control')
    (ROOT/'FIGURE_CAPTIONS.md').write_text('''# Figure captions

1. Pointwise model-based intervals for the fixed cluster coefficient in separate reduced-LVEF and increased-wall-thickness models. Primary flexible adjustment uses available demographics, recorded ECG variables and care context. The orange energy-adjusted sensitivity model conditions on a feature closely defining the waveform/RFF groups; attenuation or reversal is not evidence of a causal protective effect. Primary six-test BH values are in Table 1.

2. Matched development comparisons on 4,626 validation patients. All models use the same 23,597 fitting and 2,621 training-monitor patients, with identical five-fold tuning assignments. Imputation/scaling occur inside CV. Calibration uses training-monitor outcomes, never validation. Error bars are 95% bootstrap intervals conditional on fitted models and omit tuning/training uncertainty. Separate overlapping intervals are not paired-difference tests; paired contrasts are in Table 3.

3. Matrix-log SPD versus an equal-dimension Euclidean representation of exactly the same regularized covariance and energy information, each added to clinical and simple waveform variables. This control was added after initial incremental results and is exploratory. Neither family demonstrates a matrix-log AUROC advantage; the intervals are not a prespecified equivalence test.
''');print('Six tables and three aggregate figure sets generated.')
if __name__=='__main__':main()
