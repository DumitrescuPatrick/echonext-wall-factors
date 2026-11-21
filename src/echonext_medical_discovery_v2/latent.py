"""Frozen ResNet inference and outcome-blind continuous factors/subgroups."""
import csv,json,time
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from scipy.optimize import linear_sum_assignment
from scipy.linalg import subspace_angles
from threadpoolctl import threadpool_limits
from echonext.data import open_waveforms,canonical_batch
from echonext.stage2_models import ResNet1D
from echonext_unsupervised_discovery_v1.data import linkage,roles,sha,freeze,verify_freeze
from echonext_unsupervised_discovery_v1.questions_euclidean import euclidean

PRIVATE=Path('private/medical_discovery_next_phase')
REPORT=Path('reports/medical_discovery_next_phase')
CONFIG=Path('config/medical_discovery_next_phase/protocol.json')
CACHE=Path('private/stage2_v1/development_cache')
BASE=Path('private/unsupervised_discovery_v1/classical_v1')


def extract_features(model,x):
    trunk=model.blocks(model.stem(x));pooled=trunk.mean(-1);sd=trunk.std(-1,unbiased=False);diff=(trunk[:,:,1:]-trunk[:,:,:-1]).square().mean(-1).sqrt()/sd.clamp_min(1e-6);a=trunk[:,:,:-1];b=trunk[:,:,1:];a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True);lag=(a*b).sum(-1)/(a.square().sum(-1)*b.square().sum(-1)).sqrt().clamp_min(1e-6);temporal=torch.cat([sd,diff,lag],dim=1);logit=model.head(pooled).squeeze(-1);return pooled,temporal,logit


def number(v):
    try:return float(v)
    except (ValueError,TypeError):return np.nan


def fit_pca(x):
    scaler=StandardScaler().fit(x);pca=PCA(8,svd_solver='full').fit(scaler.transform(x));sign=np.where(pca.components_[np.arange(8),np.abs(pca.components_).argmax(1)]>=0,1.,-1.);pca.components_*=sign[:,None];return scaler,pca


def scores(transform,x):
    scaler,pca=transform;return pca.transform(scaler.transform(x))/np.sqrt(pca.explained_variance_.clip(1e-10))


def stability(x,monitor,transform):
    original=scores(transform,monitor);rng=np.random.default_rng(69001);aligned=[];angles=[]
    for _ in range(100):
        index=rng.integers(0,len(x),len(x));boot=fit_pca(x[index]);z=scores(boot,monitor);corr=np.corrcoef(original.T,z.T)[:8,8:];left,right=linear_sum_assignment(-np.abs(corr));matched=np.empty(8);matched[left]=np.abs(corr[left,right]);aligned.append(matched);basis=boot[1].components_.T*(transform[0].scale_/boot[0].scale_)[:,None];angles.append(float(np.cos(subspace_angles(transform[1].components_.T,basis)).min()))
    aligned=np.array(aligned);return {'factors':[{'factor':j+1,'matched_monitor_correlation_median':float(np.median(aligned[:,j])),'matched_monitor_correlation_2.5pct':float(np.quantile(aligned[:,j],.025)),'matched_monitor_correlation_97.5pct':float(np.quantile(aligned[:,j],.975))} for j in range(8)],'eight_factor_subspace_min_cosine_median':float(np.median(angles)),'replicates':100,'scope':'Patient bootstrap of frozen features; no neural retraining; matched factors may rotate'}


def main():
    if PRIVATE.exists():raise FileExistsError('Preserve new-phase run')
    verify_freeze(BASE);receipt=json.loads(Path('reports/stage2_v1/tune_3_complete.json').read_text());checkpoint=Path('private/stage2_v1/tune_3/best.pt')
    if sha(checkpoint)!=receipt['best_checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    integrity=json.loads(Path('reports/stage2_v1/development_cache_integrity.json').read_text())
    for name,entry in integrity.items():
        p=CACHE/name
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('Development input changed')
    PRIVATE.mkdir(parents=True,mode=0o700);md=linkage(CACHE/'echonext_metadata_100k.csv');_,mon,reps=roles(md['train']);_,first=np.unique(np.array(md['train'])[mon],return_index=True);mon=mon[np.sort(first)];indices={'fit':reps,'monitor':mon,'val':np.arange(len(md['val']))};records={'train':[],'val':[]}
    # Explicit metadata projection: no echo labels decoded here.
    names=['age_at_ecg','sex','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year','location_setting','race_ethnicity']
    with (CACHE/'echonext_metadata_100k.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] in records:records[row['split']].append({k:row.get(k,'') for k in names})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);payload=torch.load(checkpoint,map_location='cpu',weights_only=True);model=ResNet1D(payload['width']);model.load_state_dict(payload['state_dict']);model.eval();latent={};nuisance={};artifacts=[];start=time.perf_counter()
    for role,index in indices.items():
        split='val' if role=='val' else 'train';wave=open_waveforms(CACHE/f'EchoNext_{split}_waveforms.npy');pooled=[];temporal=[];logits=[]
        with torch.no_grad():
            for begin in range(0,len(index),64):
                x=torch.from_numpy(canonical_batch(wave[index[begin:begin+64]]));a,b,c=extract_features(model,x);pooled.append(a.numpy());temporal.append(b.numpy());logits.append(c.numpy())
        latent[role]={'pooled':np.concatenate(pooled),'temporal':np.concatenate(temporal),'logit':np.concatenate(logits)};row=[records[split][i] for i in index];numeric=np.array([[number(r[n]) for n in ['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year']] for r in row]);sex=np.array([{'female':0.,'male':1.}.get(r['sex'],np.nan) for r in row])[:,None]
        simple=np.load(BASE/'waveform_62001.npz',allow_pickle=False)[role];cov=euclidean(np.load(BASE/'spd_62001.npz',allow_pickle=False)[role]);categorical=np.array([[r[k] or 'missing' for k in ['location_setting','race_ethnicity']] for r in row]);nuisance[role]={'clinical':numeric,'extra':np.column_stack([sex,simple,cov]),'categorical':categorical};print('Frozen ResNet features extracted: '+role,flush=True)
    clinical=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False,extrapolation='linear'),StandardScaler());extra=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler());cat=OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False)
    def combine(role,fit=False):
        r=nuisance[role]
        return np.column_stack([(clinical.fit_transform(r['clinical']) if fit else clinical.transform(r['clinical'])),(extra.fit_transform(r['extra']) if fit else extra.transform(r['extra'])),(cat.fit_transform(r['categorical']) if fit else cat.transform(r['categorical']))])
    fitn=combine('fit',True);ridge=Ridge(alpha=10).fit(fitn,latent['fit']['pooled']);r2={}
    for role in indices:
        prediction=ridge.predict(fitn if role=='fit' else combine(role));latent[role]['residual_pooled']=latent[role]['pooled']-prediction
        if role=='monitor':r2['monitor_embedding_R2_variance_weighted']=float(r2_score(latent[role]['pooled'],prediction,multioutput='variance_weighted'))
    nuisancefile=PRIVATE/'available_feature_residualizer.joblib';joblib.dump((clinical,extra,cat,ridge),nuisancefile);artifacts.append(nuisancefile);summary={};factor_cache={}
    for space in ['pooled','residual_pooled','temporal']:
        transform=fit_pca(latent['fit'][space]);path=PRIVATE/f'{space}_PCA.joblib';joblib.dump(transform,path);artifacts.append(path);z={role:scores(transform,latent[role][space]) for role in indices};factor_cache[space]=z
        quality=stability(latent['fit'][space],latent['monitor'][space],transform);candidates=[]
        for k in [2,3,4,6]:
            km=KMeans(k,n_init=10,random_state=69001).fit(z['fit']);pred=km.predict(z['monitor']);score=silhouette_score(z['monitor'],pred,sample_size=min(2000,len(pred)),random_state=69001);candidates.append((score,k,km))
        best=max(candidates,key=lambda t:t[0]);assign=[]
        for seed in [69001,69002,69003,69004,69005]:
            km=KMeans(best[1],n_init=10,random_state=seed).fit(z['fit']);file=PRIVATE/f'{space}_cluster_{seed}.joblib';joblib.dump(km,file);artifacts.append(file);assign.append(km.predict(z['val']))
        assignments=PRIVATE/f'{space}_clusters.npz';np.savez(assignments,assignments=np.array(assign));artifacts.append(assignments)
        summary[space]={'input_dimensions':latent['fit'][space].shape[1],'explained_variance_ratio':transform[1].explained_variance_ratio_.tolist(),'stability':quality,'selected_clusters':best[1],'monitor_silhouette':float(best[0]),'candidate_silhouettes':[{'k':t[1],'silhouette':float(t[0])} for t in candidates],'validation_cluster_seed_pairwise_ARI':[float(adjusted_rand_score(assign[i],assign[j])) for i in range(5) for j in range(i)],'factor_logit_correlation':[float(np.corrcoef(z['val'][:,j],latent['val']['logit'])[0,1]) for j in range(8)]};print('Outcome-blind factors frozen: '+space,flush=True)
    raw=PRIVATE/'frozen_embeddings.npz';np.savez(raw,**{f'{role}_{name}':value for role,d in latent.items() for name,value in d.items()});artifacts.append(raw);cache=PRIVATE/'factor_scores.npz';np.savez(cache,**{f'{space}_{role}':values for space,d in factor_cache.items() for role,values in d.items()});artifacts.append(cache)
    freeze(PRIVATE,artifacts+[checkpoint,CONFIG,Path(__file__)],{'supervised_representation':True,'source_validation_selected':True,'echo_labels_decoded_for_factor_fitting':False,'test_accessed':False,'factor_count':8,'metadata_sha256':sha(CACHE/'echonext_metadata_100k.csv')})
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'FACTOR_DISCOVERY.json').write_text(json.dumps({'status':'Supervised/validation-selected reference, not unsupervised clinical discovery','patients':{r:len(ix) for r,ix in indices.items()},'inference_and_factor_seconds':time.perf_counter()-start,'device':'cpu inference only','new_neural_training':False,'residualizer':r2,'spaces':summary,'test_accessed':False},indent=2));print('Frozen latent/factor study ready for exploratory association analysis.',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
