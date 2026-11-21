"""Fixed-group stress tests in normalized signal space; no clinical relabeling."""
import json,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import adjusted_rand_score
from echonext.data import canonical_batch,open_waveforms
from echonext_unsupervised_discovery_v1.features import spd_features
from echonext_unsupervised_discovery_v1.data import verify_freeze,freeze
from .freeze import PRIVATE,BASE,CONFIG
from .analysis import change_interval

SCENARIOS=['baseline','global_gain_0.5','global_gain_2','per_lead_unit_SD','global_unit_RMS','unequal_per_lead_gain','global_polarity','I_polarity_proxy','RA_LA_normalized_proxy','V1_V2_swap_proxy','independent_noise_5pct','common_60Hz_10pct','shared_time_permutation']

def transform(x,scenario,rng):
    out=x.copy();centered=out-out.mean(-1,keepdims=True);sd=centered.std(-1,keepdims=True);rms=np.sqrt(np.mean(centered**2,axis=(1,2),keepdims=True))
    if scenario=='baseline':return out
    if scenario=='global_gain_0.5':return out*.5
    if scenario=='global_gain_2':return out*2
    if scenario=='per_lead_unit_SD':return centered/np.maximum(sd,1e-8)
    if scenario=='global_unit_RMS':return centered/np.maximum(rms,1e-8)
    if scenario=='unequal_per_lead_gain':return out*np.geomspace(.5,2,12).astype('float32')[None,:,None]
    if scenario=='global_polarity':return -out
    if scenario=='I_polarity_proxy':out[:,0]*=-1;return out
    if scenario=='RA_LA_normalized_proxy':
        # Ideal electrode algebra represented AFTER provider normalization: proxy only.
        out=out[:,[0,2,1,4,3,5,6,7,8,9,10,11]].copy();out[:,0]*=-1;return out
    if scenario=='V1_V2_swap_proxy':out[:,[6,7]]=out[:,[7,6]];return out
    if scenario=='independent_noise_5pct':return out+(rng.normal(size=out.shape)*(.05*sd)).astype('float32')
    if scenario=='common_60Hz_10pct':
        t=np.arange(out.shape[-1])/250;phase=rng.uniform(0,2*np.pi,(len(out),1,1));return out+(.1*rms*np.sin(2*np.pi*60*t[None,None,:]+phase)).astype('float32')
    if scenario=='shared_time_permutation':return out[:,:,rng.permutation(out.shape[-1])]
    raise ValueError('Unknown perturbation')

def ordinary_features(x):
    z=x.astype(float);z-=z.mean(-1,keepdims=True);c=z@z.transpose(0,2,1)/(z.shape[-1]-1);scale=np.trace(c,axis1=1,axis2=2)/12;s=.9*c/np.maximum(scale[:,None,None],1e-10)+.1*np.eye(12)-np.eye(12);i,j=np.triu_indices(12);return np.column_stack([s[:,i,j]*np.where(i==j,1,np.sqrt(2)),np.log(np.maximum(scale,1e-10))])

def main():
    verify_freeze(PRIVATE);folder=PRIVATE/'stress'
    if folder.exists():raise FileExistsError('Preserve stress analysis')
    folder.mkdir(mode=0o700);wave=open_waveforms('private/stage2_v1/development_cache/EchoNext_val_waveforms.npy');estimator=joblib.load(BASE/'spd_62001.joblib');original=np.load(BASE/'spd_62001.assignments.npy',allow_pickle=False);results=[];files=[]
    for j,scenario in enumerate(SCENARIOS):
        rng=np.random.default_rng(67003+j);labels=[];i_iii=[]
        for start in range(0,len(wave),64):
            x=transform(canonical_batch(wave[start:start+64]),scenario,rng);z=spd_features(x);labels.extend(estimator['estimator'].predict(estimator['scaler'].transform(z)));a=x[:,0].astype(float);b=x[:,2].astype(float);a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True);i_iii.extend((a*b).sum(-1)/np.maximum(np.sqrt((a*a).sum(-1)*(b*b).sum(-1)),1e-12))
        labels=np.array(labels);changed=int((labels!=original).sum())
        if scenario=='baseline' and changed:raise ValueError('Baseline does not reproduce original groups')
        file=folder/f'{scenario}.npz';np.savez(file,assignment=labels);files.append(file);results.append({'scenario':scenario,'patients':len(labels),'assignment_change_fraction':changed/len(labels),'assignment_change_Wilson_95_ci':change_interval(changed,len(labels)),'ARI':float(adjusted_rand_score(original,labels)),'mean_I_III_correlation':float(np.mean(i_iii))});print('Stress scenario complete: '+scenario,flush=True)
    # Feature extraction microbenchmark: in-memory CPU, no inference or I/O timing.
    x=canonical_batch(wave[:256]);bench=[]
    for name,fn in [('ordinary_covariance',ordinary_features),('log_SPD',spd_features)]:
        fn(x);times=[]
        for _ in range(5):
            start=time.perf_counter();fn(x);times.append(time.perf_counter()-start)
        bench.append({'representation':name,'batch':len(x),'CPU_seconds_median':float(np.median(times)),'milliseconds_per_ECG':float(1000*np.median(times)/len(x)),'scope':'In-memory feature function only, CPU; no GPU, I/O or clinical throughput claim'})
    freeze(folder,files+[Path(__file__),CONFIG,PRIVATE/'FREEZE.json'],{'definitions_changed':False,'reversal_interpretation':'Normalized-space proxies, not calibrated electrode simulation'})
    Path('reports/covariance_validation_v1/STRESS_TESTS.json').write_text(json.dumps({'status':'Exploratory stress sensitivity of fixed normalized-space groups','original_group_definitions_changed':False,'clinical_claim':'No simulated scenario verifies or excludes real axis abnormalities, lead misplacement or artifact causality','results':results,'CPU_microbenchmark':bench},indent=2));print('Aggregate stress tests and complexity timings saved.',flush=True)
if __name__=='__main__':main()
