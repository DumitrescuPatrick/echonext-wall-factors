"""Factor redundancy, nuisance explainability and normalized-space stress diagnostics."""
import csv,json
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.metrics import r2_score
from echonext.stage2_models import ResNet1D
from echonext.data import open_waveforms,canonical_batch
from echonext_unsupervised_discovery_v1.data import verify_freeze
from echonext_unsupervised_discovery_v1.features import waveform_features
from echonext_covariance_validation_v1.stress import ordinary_features,transform
from .latent import PRIVATE,REPORT,CACHE,extract_features,scores,number

SCENARIOS=['baseline','global_gain_0.5','global_gain_2','global_unit_RMS','per_lead_unit_SD','unequal_per_lead_gain','RA_LA_normalized_proxy','independent_noise_5pct','shared_time_permutation','synchronized_reflect_shift_25']

def main():
    out=REPORT/'FACTOR_DIAGNOSTICS.json'
    if out.exists():raise FileExistsError('Preserve factor diagnostics')
    verify_freeze(PRIVATE);raw=np.load(PRIVATE/'frozen_embeddings.npz',allow_pickle=False);cache=np.load(PRIVATE/'factor_scores.npz',allow_pickle=False);transforms={name:joblib.load(PRIVATE/f'{name}_PCA.joblib') for name in ['pooled','residual_pooled','temporal']};predicted=raw['monitor_pooled']-raw['monitor_residual_pooled'];pooled_prediction=scores(transforms['pooled'],predicted);explainability=[float(r2_score(cache['pooled_monitor'][:,j],pooled_prediction[:,j])) for j in range(8)];all_scores=np.column_stack([cache[f'{s}_val'] for s in transforms]);redundancy=np.corrcoef(all_scores.T);labels=[f'{s}_PC{j+1}' for s in transforms for j in range(8)];pairs=[{'first':labels[i],'second':labels[j],'Pearson_r':float(redundancy[i,j])} for i in range(24) for j in range(i) if abs(redundancy[i,j])>=.8]
    clinical,extra,cat,ridge=joblib.load(PRIVATE/'available_feature_residualizer.joblib');records=[]
    with (CACHE/'echonext_metadata_100k.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split']=='val':records.append({n:row.get(n,'') for n in ['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year','sex','location_setting','race_ethnicity']})
    index=np.sort(np.random.default_rng(69001).choice(len(records),512,replace=False));selected=[records[i] for i in index];numeric=np.array([[number(r[n]) for n in ['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year']] for r in selected]);sex=np.array([{'female':0.,'male':1.}.get(r['sex'],np.nan) for r in selected])[:,None];categorical=np.array([[r[n] or 'missing' for n in ['location_setting','race_ethnicity']] for r in selected]);static_clinical=clinical.transform(numeric);static_cat=cat.transform(categorical);wave=open_waveforms(CACHE/'EchoNext_val_waveforms.npy');original=canonical_batch(wave[index]);payload=torch.load('private/stage2_v1/tune_3/best.pt',map_location='cpu',weights_only=True);model=ResNet1D(payload['width']);model.load_state_dict(payload['state_dict']);model.eval();torch.set_num_threads(4);results=[];reference=None
    for k,scenario in enumerate(SCENARIOS):
        rng=np.random.default_rng(69001+k)
        if scenario=='synchronized_reflect_shift_25':x=np.pad(original,((0,0),(0,0),(25,25)),mode='reflect')[:,:,50:50+2500].copy()
        else:x=transform(original,scenario,rng)
        pooled=[];temporal=[]
        with torch.no_grad():
            for start in range(0,len(x),64):
                a,b,_=extract_features(model,torch.from_numpy(x[start:start+64]));pooled.append(a.numpy());temporal.append(b.numpy())
        pooled=np.concatenate(pooled);temporal=np.concatenate(temporal);available=np.column_stack([static_clinical,extra.transform(np.column_stack([sex,waveform_features(x),ordinary_features(x)])),static_cat]);residual=pooled-ridge.predict(available);values={'pooled':scores(transforms['pooled'],pooled),'residual_pooled':scores(transforms['residual_pooled'],residual),'temporal':scores(transforms['temporal'],temporal)}
        if reference is None:reference=values
        for space in values:
            for j in range(8):
                a=reference[space][:,j];b=values[space][:,j];results.append({'scenario':scenario,'space':space,'factor':j+1,'patients':len(x),'factor_correlation_to_unperturbed':float(np.corrcoef(a,b)[0,1]),'RMS_factor_change_in_training_SD':float(np.sqrt(np.mean((a-b)**2)))})
        print('Frozen-factor stress complete: '+scenario,flush=True)
    out.write_text(json.dumps({'monitor_pooled_factor_R2_from_available_nuisances':explainability,'strong_cross_factor_correlations':pairs,'stress':results,'limits':'Normalized-space diagnostic with clinical machine measurements held fixed; not a full reacquisition simulation. Permutations/polarity are destructive controls; no real artifact prevalence is inferred.','sample':'512 fixed outcome-blind validation patients, seed69001','neural_seed_stability_measured':False},indent=2));print('Factor redundancy/artifact diagnostics saved.')
if __name__=='__main__':main()
