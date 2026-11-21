"""Phase-1 interpretation figures, exclusively from aggregate result files."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .phase1_audit import LEADS

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);p.add_argument('--components',type=Path,default=Path('reports/unsupervised_discovery_v1/SECONDARY_COMPONENTS.json'));a=p.parse_args();root=a.reports;figs=root/'figures';figs.mkdir(exist_ok=True)
    rows=json.loads((root/'EXPLORATORY_CHARACTERIZATION.json').read_text())['signal_descriptors'];matrices=[np.eye(12),np.eye(12)]
    for r in rows:
        if r['representation']=='spd' and r['descriptor'].startswith('Pearson correlation '):
            first,second=r['descriptor'].replace('Pearson correlation ','').split(' / ');i=LEADS.index(first);j=LEADS.index(second)
            for k in [0,1]:matrices[k][i,j]=matrices[k][j,i]=r[f'mean_cluster_{k}']
    def save(name):
        plt.tight_layout();plt.savefig(figs/(name+'.png'),dpi=300);plt.savefig(figs/(name+'.pdf'));plt.close()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,m,title in zip(axes,[*matrices,matrices[1]-matrices[0]],['Covariance cluster 0','Covariance cluster 1','Difference: cluster 1 − cluster 0']):
        im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1);ax.set_xticks(range(12),LEADS,rotation=90);ax.set_yticks(range(12),LEADS);ax.set_title(title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.65)
    save('figure_1_interlead_relationships')
    components=json.loads(a.components.read_text())['results'];selected=[r for r in components if r['representation']=='spd' and r['cluster']==1 and not r.get('suppressed')]
    labels={'lvef_lte_45_flag':'LVEF ≤45%','lvwt_gte_13_flag':'LV wall thickness ≥13 mm','aortic_stenosis_moderate_or_greater_flag':'Aortic stenosis','aortic_regurgitation_moderate_or_greater_flag':'Aortic regurgitation','mitral_regurgitation_moderate_or_greater_flag':'Mitral regurgitation','tricuspid_regurgitation_moderate_or_greater_flag':'Tricuspid regurgitation','rv_systolic_dysfunction_moderate_or_greater_flag':'RV systolic dysfunction','pericardial_effusion_moderate_large_flag':'Pericardial effusion','pasp_gte_45_flag':'PASP ≥45 mmHg','tr_max_gte_32_flag':'Provider TR velocity flag'}
    plt.figure(figsize=(8,5))
    for j,r in enumerate(selected):
        effect=100*r['risk_difference_vs_rest'];plt.errorbar(effect,j,xerr=[[effect-100*r['ci_low']],[100*r['ci_high']-effect]],fmt='o',color='#006080' if r['q_value_BH']<.05 else '.5',capsize=3)
    plt.yticks(range(len(selected)),[labels[r['outcome']] for r in selected]);plt.axvline(0,color='.5',linestyle='--');plt.xlabel('Component prevalence difference: cluster 1 − cluster 0 (percentage points)');plt.title('Covariance pattern: exploratory echo-component associations');save('figure_2_echo_component_associations')
    models=json.loads((root/'TRAINED_MODEL_AUDIT.json').read_text())['groups'];plt.figure(figsize=(6,5))
    for method,marker in [('spd','o'),('waveform','s'),('hilbert','^')]:
        r=[v for v in models if v['representation']==method];plt.scatter([v['SHD_prevalence'] for v in r],[v['mean_ResNet_risk'] for v in r],marker=marker,s=70,label=method)
        for v in r:plt.annotate(str(v['cluster']),(v['SHD_prevalence'],v['mean_ResNet_risk']),xytext=(5,3),textcoords='offset points')
    plt.plot([.3,.6],[.3,.6],'--',color='.5');plt.xlim(.3,.6);plt.ylim(.3,.6);plt.xlabel('Observed SHD prevalence');plt.ylabel('Mean saved ResNet risk');plt.legend(title='Frozen representation');plt.title('Descriptive model risk by pattern');save('figure_3_resnet_risk_by_pattern')
    (root/'FIGURE_CAPTIONS.md').write_text('''# Phase-1 figure captions

1. Mean whole-ECG Pearson lead correlations in the frozen covariance groups, and their difference. These describe global interlead relationships, not QRS-specific axis angles or conduction diagnoses. Separation in a covariance-derived feature is expected from the clustering construction; clinical enrichment is a separate analysis.

2. Cluster 1 minus cluster 0 component prevalence differences with conditional 95% patient-bootstrap intervals. Blue marks have BH q<0.05 within the previously defined exploratory component-test family; grey marks do not. Suppressed rare components are omitted. Intervals are pointwise, not simultaneous. Categorical valve/function flags use supplied moderate-or-greater definitions. The TR flag retains the provider's original label despite the documented unit discrepancy.

3. Mean saved ResNet probability versus observed SHD prevalence in each frozen group; labels indicate arbitrary cluster numbers. Each patient appears in each representation; points are not independent. The selected ResNet used these validation outcomes for model selection. This descriptive calibration check is not an external validation or a fair geometric-model comparison.
''');print('Three aggregate Phase-1 interpretation figures saved.')
if __name__=='__main__':main()
