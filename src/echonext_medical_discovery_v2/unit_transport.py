"""Explicitly altered-input frozen-network transport sensitivity, not native replication."""
import json,zipfile,csv,ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,joblib
from scipy.linalg import qr
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from threadpoolctl import threadpool_limits
from echonext.stage2_models import ResNet1D
from echonext.data import open_waveforms,canonical_batch
from echonext_unsupervised_discovery_v1.data import verify_freeze,freeze,sha
from echonext_unsupervised_discovery_v1.questions_associations import design,estimate
from echonext_unsupervised_discovery_v1.questions_interpretation import correlations_from_logspd
from echonext_ptbxl_transport_v1.data import ARCHIVE,find,manifest,decode_record,csv_rows,endpoint
from echonext_ptbxl_transport_v1.features import prepare
from echonext_ptbxl_transport_v1.associate import design as ptb_design
from .latent import PRIVATE,REPORT,CACHE,BASE,extract_features,scores
from .models import input_view
from .associate import independent_columns,BH,number

CONFIG=Path('config/medical_discovery_next_phase/unit_transport.json')


def main():
    verify_freeze(PRIVATE);ptb=Path('private/ptbxl_transport_v1');external=verify_freeze(ptb);folder=PRIVATE/'unit_input_sensitivity'
    if folder.exists():raise FileExistsError('Preserve altered-input sensitivity')
    if sha(ARCHIVE)!=external['details']['archive_sha256']:raise ValueError('PTB archive changed')
    folder.mkdir(mode=0o700);payload=torch.load('private/stage2_v1/tune_3/best.pt',map_location='cpu',weights_only=True);model=ResNet1D(payload['width']);model.load_state_dict(payload['state_dict']);model.eval();torch.set_num_threads(4);transform=joblib.load(PRIVATE/'pooled_PCA.joblib')
    def project(batch):
        with torch.no_grad():a,_,_=extract_features(model,input_view(torch.from_numpy(batch),'per_record_lead_unit_SD'))
        return scores(transform,a.numpy())
    wave=open_waveforms(CACHE/'EchoNext_val_waveforms.npy');source=np.concatenate([project(canonical_batch(wave[i:i+64])) for i in range(0,len(wave),64)]);print('Altered-input EchoNext projection complete',flush=True)
    admin=json.loads((ptb/'admin.json').read_text());valid=np.load(ptb/'frozen_proxy_features.npz',allow_pickle=False)['valid'];admin=[r for i,r in enumerate(admin) if valid[i]];outputs=[];buffer=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        sums=manifest(z);metadata=find(z,'ptbxl_database.csv');prefix=metadata[:-len('ptbxl_database.csv')]
        def one(row):
            if int(row['strat_fold'])==10:raise PermissionError('PTB test fold sealed')
            x,_,_=decode_record(z,prefix+row['filename_hr'],sums);return prepare(x,'primary')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for j,x in enumerate(pool.map(one,admin)):
                buffer.append(x)
                if len(buffer)==64:outputs.append(project(np.array(buffer)));buffer=[]
                if (j+1)%2000==0:print('PTB altered-input projected patients: '+str(j+1),flush=True)
            if buffer:outputs.append(project(np.array(buffer)))
    target=np.concatenate(outputs);cache=folder/'projection_scores.npz';np.savez(cache,Echo_val=source,PTB=target)
    freeze(folder,[cache,CONFIG,Path(__file__),PRIVATE/'pooled_PCA.joblib',Path('private/stage2_v1/tune_3/best.pt'),ptb/'FREEZE.json'],{'status':'Changed-input sensitivity; original PCA and weights fixed','PTB_labels_used_for_projection':False,'native_transfer_authenticated':False})
    # Clinical associations are decoded only after altered-input projections are frozen.
    fields=['age_at_ecg','sex','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year','location_setting','race_ethnicity','lvef_lte_45_flag','lvwt_gte_13_flag'];rows=[]
    with (CACHE/'echonext_metadata_100k.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split']=='val':rows.append({n:row.get(n,'') for n in fields})
    simple=np.load(BASE/'waveform_62001.npz',allow_pickle=False)['val'];energy=np.sqrt(np.mean(simple[:,12:24].astype(float)**2,axis=1));cov,_,_=design(rows,'flexible_plus_energy',energy);corr=correlations_from_logspd(np.load(BASE/'spd_62001.npz',allow_pickle=False)['val'])[:,0,2];proxy=StandardScaler().fit_transform(SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False).fit_transform(corr[:,None]));risk=np.load(PRIVATE/'frozen_embeddings.npz',allow_pickle=False)['val_logit'];cov=independent_columns(np.column_stack([cov,proxy,risk]));source_results=[]
    for endpoint_name in ['lvef_lte_45_flag','lvwt_gte_13_flag']:
        y=np.array([number(r[endpoint_name]) for r in rows])
        for j in range(8):source_results.append({'factor':j+1,'endpoint':endpoint_name,'status':'estimated',**estimate(y,source[:,j],cov)})
    BH(source_results,16)
    with zipfile.ZipFile(ARCHIVE) as z:
        sums=manifest(z);metadata_rows=csv_rows(z,find(z,'ptbxl_database.csv'),sums);dictionary=csv_rows(z,find(z,'scp_statements.csv'),sums)
    byid={r['ecg_id']:r for r in metadata_rows};rows=[byid[r['ecg_id']] for r in admin];codes=[ast.literal_eval(r['scp_codes']) for r in rows];known=np.array([bool(r['heart_axis'].strip()) for r in rows]);target_results=[]
    for cls in ['CD','HYP','STTC']:
        members={r[''] for r in dictionary if r.get('diagnostic_class')==cls and r.get('diagnostic') in ('1','1.0')};y=np.array([endpoint(c,members) for c in codes])
        for name,mask in [('age_sex',np.ones(len(rows),bool)),('axis_context_quality',np.ones(len(rows),bool)),('complete_axis',known)]:
            keep=mask&np.isin(y,[0,1]);selected=[r for r,k in zip(rows,keep) if k];cov=ptb_design(selected,name=='axis_context_quality')
            if name=='complete_axis':
                axis=OneHotEncoder(drop='first',min_frequency=50,sparse_output=False).fit_transform(np.array([r['heart_axis'] for r in selected])[:,None]);cov=independent_columns(np.column_stack([cov,axis]))
            for j in range(8):
                record={'factor':j+1,'endpoint':cls,'adjustment':name,'patients':int(keep.sum())}
                try:record.update(estimate(y[keep],target[keep,j],cov));record['status']='estimated'
                except (RuntimeError,ValueError,np.linalg.LinAlgError):record['status']='not_estimable'
                target_results.append(record)
    BH(target_results,72)
    native=np.load(PRIVATE/'factor_scores.npz',allow_pickle=False);threshold=float(np.quantile((native['pooled_fit']**2).sum(1),.95));report={'status':'Altered-input sensitivity only; not native-factor or new-encoder validation','source_associations_BH16':source_results,'PTB_ECG_associations_BH72':target_results,'source_native_vs_unit_factor_correlations':[float(np.corrcoef(native['pooled_val'][:,j],source[:,j])[0,1]) for j in range(8)],'native_training_95pct_score_radius_threshold':threshold,'source_unit_outside_native_training_radius_fraction':float(((source**2).sum(1)>threshold).mean()),'PTB_unit_outside_native_training_radius_fraction':float(((target**2).sum(1)>threshold).mean()),'scale':'One original native fitting-patient PC score SD; no PTB scaling fit','echo_outcomes_in_PTB':False,'PTB_fold10_used':False,'new_encoder_training':False}
    (REPORT/'UNIT_INPUT_TRANSPORT_SENSITIVITY.json').write_text(json.dumps(report,indent=2));print('Explicit altered-input external sensitivity saved.',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
