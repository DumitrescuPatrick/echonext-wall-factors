"""Shared locked populations and train-only preprocessing for five-question follow-up."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import linkage,roles,sha,verify_freeze

ECG=['ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected']
TARGETS=['shd_moderate_or_greater_flag','lvef_lte_45_flag','lvwt_gte_13_flag']
ROOT=Path('private/unsupervised_discovery_v1/classical_v1')
META=Path('private/stage2_v1/development_cache/echonext_metadata_100k.csv')
PROTOCOL=Path('config/unsupervised_discovery_v1/five_questions_v1/protocol.json')

def number(v):
    try:return float(v)
    except (TypeError,ValueError):return np.nan

def load_data(validation_targets=False):
    manifest=verify_freeze(ROOT)
    if sha(META)!=manifest['details']['metadata_sha256']:raise ValueError('Metadata changed')
    linkage_md=linkage(META);fit,monitor,reps=roles(linkage_md['train']);_,first=np.unique(np.array(linkage_md['train'])[monitor],return_index=True);monitor=monitor[np.sort(first)]
    records={'train':[],'val':[]}
    with META.open(newline='') as f:
        for row in csv.DictReader(f):
            s=row['split']
            if s not in records:continue
            names=ECG+['age_at_ecg','sex','acquisition_year','location_setting','race_ethnicity']
            selected={k:row.get(k,'') for k in names}
            if s=='train' or validation_targets:
                for t in TARGETS:selected[t]=row[t]
            records[s].append(selected)
    arrays={}
    for role,split,index in [('fit','train',reps),('monitor','train',monitor),('val','val',np.arange(len(records['val'])))]:
        rows=[records[split][i] for i in index];clinical=np.array([[*[number(r[n]) for n in ECG],number(r['age_at_ecg']),{'female':0.,'male':1.}.get(r['sex'],np.nan)] for r in rows]);arrays[role]={'clinical':clinical,'records':rows}
        if split=='train' or validation_targets:arrays[role]['targets']={t:np.array([number(r[t]) for r in rows]) for t in TARGETS}
        for method in ['waveform','spd']:
            file=ROOT/f'{method}_62001.npz'
            if str(file.resolve()) not in manifest['files']:raise ValueError('Unfrozen feature cache')
            with np.load(file,allow_pickle=False) as cache:arrays[role][method]=cache[role].copy()
    arrays['rep_indices']=reps;arrays['monitor_indices']=monitor
    arrays['groups']={m:np.load(ROOT/f'{m}_62001.assignments.npy',allow_pickle=False) for m in ['waveform','spd','hilbert']}
    return arrays

class RandomFourier(BaseEstimator,TransformerMixin):
    def __init__(self,n_components=128,seed=62001):self.n_components=n_components;self.seed=seed
    def fit(self,x,y=None):
        rng=np.random.default_rng(self.seed);self.omega_=rng.normal(size=(x.shape[1],self.n_components))/np.sqrt(x.shape[1]);self.phase_=rng.uniform(0,2*np.pi,self.n_components);return self
    def transform(self,x):return np.sqrt(2/self.n_components)*np.cos(x@self.omega_+self.phase_)

def matrix(data,name):
    clinical=data['clinical'];simple=data['waveform'];spd=data['spd']
    if name=='conventional_ecg':return clinical[:,:5]
    if name=='clinical_ecg':return clinical
    if name in ('clinical_ecg_simple','clinical_ecg_hilbert'):return np.column_stack([clinical,simple])
    if name=='clinical_ecg_spd':return np.column_stack([clinical,spd])
    if name=='clinical_ecg_simple_spd':return np.column_stack([clinical,simple,spd])
    raise ValueError('Unknown feature definition')

def preprocessing(name):
    numeric=lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
    if name!='clinical_ecg_hilbert':return numeric()
    return ColumnTransformer([('clinical',numeric(),list(range(7))),('RFF',make_pipeline(numeric(),RandomFourier()),list(range(7,55)))],remainder='drop')
