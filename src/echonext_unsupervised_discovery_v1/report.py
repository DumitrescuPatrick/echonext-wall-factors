"""Publication-safe tables and figures generated exclusively from aggregate inputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'waveform':'Simple waveform','spd':'SPD covariance','hilbert':'Hilbert / RFF'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);p.add_argument('--discovery-summary',type=Path,required=True);a=p.parse_args();root=a.reports;fig=root/'figures';fig.mkdir(exist_ok=True)
    association=json.loads((root/'classical_v1/aggregate_associations.json').read_text())['results'];primary=[r for r in association if r['analysis'].endswith('62001')]
    table=root/'PRIMARY_PATTERN_ASSOCIATIONS.csv'
    with table.open('w',newline='') as f:
        fields=['analysis','cluster','patients','events','prevalence','risk_difference_vs_rest','ci_low','ci_high','p_value','q_value_BH','suppressed'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(primary)
    probe=json.loads((root/'LOGISTIC_PROBE.json').read_text())['results'];audit=json.loads((root/'MORPHOLOGY_AUDIT.json').read_text())['patterns'];discovery=json.loads(a.discovery_summary.read_text())
    def save(name):
        plt.tight_layout();plt.savefig(fig/(name+'.png'),dpi=300);plt.savefig(fig/(name+'.pdf'));plt.savefig(fig/(name+'.svg'));plt.close()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plt.figure(figsize=(7,4))
    for j,method in enumerate(NAMES):
        rows=[r for r in primary if r['analysis']==method+'_62001'];plt.bar(np.array([0,1])+j*3,[100*r['prevalence'] for r in rows],label=NAMES[method])
    plt.xticks([0.5,3.5,6.5],list(NAMES.values()));plt.ylabel('Validation SHD prevalence (%)');plt.ylim(0,65);plt.title('Frozen patterns: cluster 0 and cluster 1');save('figure_1_pattern_enrichment')
    plt.figure(figsize=(7,4));methods=list(NAMES);v=[discovery['validation_assignment_stability_ARI'][m] for m in methods];plt.boxplot(v,tick_labels=[NAMES[m] for m in methods]);plt.ylim(0,1.05);plt.ylabel('Adjusted Rand index');plt.title('Five-seed pattern-assignment agreement');save('figure_2_seed_stability')
    fig3,axes=plt.subplots(1,2,figsize=(8,4))
    for axis,variable,label in zip(axes,['age_at_ecg','qrs_duration'],['Mean age (years)','Mean QRS duration (ms)']):
        for j,method in enumerate(NAMES):
            rows=audit[method+'_62001.assignments'];axis.plot([j-.1,j+.1],[r['numeric'][variable]['mean'] for r in rows],'o-',label=NAMES[method])
        axis.set_xticks(range(3),list(NAMES.values()),rotation=20);axis.set_ylabel(label)
    save('figure_3_morphology_confounding')
    plt.figure(figsize=(7,4))
    for j,r in enumerate(probe):
        ci=r['confidence_intervals']['AUROC'];plt.errorbar(r['AUROC'],j,xerr=[[r['AUROC']-ci[0]],[ci[1]-r['AUROC']]],fmt='o',capsize=4)
    plt.yticks(range(len(probe)),[NAMES[r['representation'].split('_')[0]] for r in probe]);plt.xlabel('Validation AUROC (95% patient-bootstrap interval)');plt.xlim(.70,.85);plt.title('Fixed supervised probes of frozen representations');save('figure_4_logistic_probes')
    stability=root/'BOOTSTRAP_STABILITY.json'
    if stability.exists():
        rows=json.loads(stability.read_text())['results'];plt.figure(figsize=(7,4))
        for j,r in enumerate(rows):
            low=r['ARI_2.5_percentile'];high=r['ARI_97.5_percentile'];mid=r['ARI_median'];plt.errorbar(mid,j,xerr=[[mid-low],[high-mid]],fmt='o',capsize=4)
        plt.yticks(range(len(rows)),[NAMES[r['representation'].split('_')[0]] for r in rows]);plt.xlabel('ARI, median and 2.5–97.5 percentiles');plt.xlim(0,1.02);plt.title('Patient-bootstrap cluster stability: fixed representations');save('figure_5_patient_bootstrap_stability')
    captions='''# Figure captions

1. **SHD enrichment in frozen patterns.** Primary seed 62001, 4,626 validation patients. Cluster labels are arbitrary; bars show descriptive prevalence, not newly diagnosed disease subtypes. Assignments were frozen before outcome access.

2. **Agreement across five seeds.** Ten pairwise adjusted Rand indices per representation on the same validation patients. Pairs are dependent. Hilbert/RFF seeds change the random feature map as well as centroid initialization; the seed comparison therefore measures different sources of variation across families.

3. **Known morphology and demographic differences.** Mean age and machine-reported QRS duration by frozen cluster. Each line joins cluster 0 (left) to cluster 1 (right) within a representation; connecting lines do not imply temporal change. These differences are alternative explanations for SHD enrichment.

4. **Supervised probes.** Logistic regression C=1 fitted on one ECG per fitting patient after freezing representations. Intervals use 2,000 patient bootstraps of validation predictions and condition on the fitted model. They omit training uncertainty. This is a supervised evaluation, not an unsupervised clustering accuracy measure.

5. **Patient-bootstrap stability.** One hundred refits of the scaler and selected clustering method on resampled fitting patients; assignments compared on training-monitor patients against the original fitted model. Intervals are empirical resampling percentiles and exclude representation fitting and K-selection uncertainty.
'''
    (root/'FIGURE_CAPTIONS.md').write_text(captions);print('Aggregate tables and publication-resolution figures generated.')
if __name__=='__main__':main()
