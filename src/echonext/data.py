"""Read-only memory-mapped development data; test loading is deliberately absent."""
from pathlib import Path
import csv
import numpy as np
from .schema import WAVE_SHAPE, LABELS, PRIMARY, SPLITS


def binary(value):
    if value in ('', 'NaN', 'nan', 'NA', 'None'): return np.nan
    if value in ('0', '0.0'): return 0.
    if value in ('1', '1.0'): return 1.
    raise ValueError('Non-binary outcome encoding; values withheld')


def read_metadata(path):
    """Return private mappings. Never access test outcomes, even for prevalence."""
    result={s:{'rows':[], 'patients':[], 'ecgs':[], 'recent':[], 'labels':[]} for s in SPLITS}
    serial_ok=True;missing_keys=0
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames
        required={'patient_key','ecg_key','split',*LABELS}
        if not required.issubset(columns or []):raise ValueError('Metadata schema mismatch')
        for i,r in enumerate(reader):
            s=r['split']
            if s not in result:raise ValueError('Unrecognized split; value withheld')
            missing_keys += int(not r['patient_key'] or not r['ecg_key'])
            d=result[s];d['rows'].append(i);d['patients'].append(r['patient_key']);d['ecgs'].append(r['ecg_key'])
            d['recent'].append(r.get('most_recent_ecg',''))
            # Integrity-only test operations use keys/splits; no target access.
            if s in ('train','val'):d['labels'].append([binary(r[name]) for name in LABELS])
            if '' in r:serial_ok &= r['']==str(i)
    if missing_keys:raise ValueError('Missing alignment keys; values withheld')
    return result,columns,serial_ok


def validate_splits(metadata):
    groups={s:set(metadata[s]['patients']) for s in SPLITS}
    overlaps={f'{a}_{b}':len(groups[a]&groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    keys=[k for d in metadata.values() for k in d['ecgs']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate ECG keys; values withheld')
    if any(overlaps.values()):raise ValueError('Patient leakage between official splits; values withheld')
    if any(len(metadata[s]['patients'])!=len(groups[s]) for s in ('val','test')):
        raise ValueError('Repeated validation/test patients; values withheld')
    return overlaps


def open_waveforms(path):
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    if a.ndim!=4 or a.shape[1:]!=WAVE_SHAPE:raise ValueError('Waveform shape mismatch')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Waveform dtype mismatch')
    expected=a.offset+a.nbytes
    if Path(path).stat().st_size!=expected:raise ValueError('Truncated or trailing waveform bytes')
    return a


def canonical_batch(x):
    """(N,1,T,12) -> private copied float32 (N,12,T), without modifying mmap."""
    x=np.asarray(x)
    if x.ndim!=4 or x.shape[1:]!=WAVE_SHAPE:raise ValueError('Batch shape mismatch')
    if not np.isfinite(x).all():raise ValueError('Nonfinite waveform; quarantine uniformly before fitting')
    if np.any(np.abs(x)>np.finfo(np.float32).max):raise ValueError('Waveform overflows float32 batch representation')
    return np.array(x[:,0].transpose(0,2,1),dtype=np.float32,copy=True,order='C')


class DevelopmentData:
    def __init__(self,root,split):
        if split not in ('train','val'):raise PermissionError('Official test access is locked in Stage 1')
        self.root=Path(root);self.split=split
        md,_,_=read_metadata(self.root/'echonext_metadata_100k.csv');validate_splits(md)
        self.waveforms=open_waveforms(self.root/f'EchoNext_{split}_waveforms.npy')
        self.labels=np.array(md[split]['labels'],float)
        if len(self.waveforms)!=len(self.labels):raise ValueError('Split row alignment mismatch')

    def batch(self,indices):return canonical_batch(self.waveforms[indices]), self.labels[indices].copy()


class ChannelStandardizer:
    """Optional extra train-fit scaling; primary protocol uses shipped scaling only."""
    def fit(self,batches,role='fit'):
        if role!='fit':raise PermissionError('Scaling may be fitted only on fitting patients')
        n=0;s=np.zeros(12);q=np.zeros(12)
        for x in batches:
            if x.ndim!=3 or x.shape[1]!=12 or not np.isfinite(x).all():raise ValueError('Invalid scaling batch')
            n+=x.shape[0]*x.shape[2];s+=x.sum((0,2),dtype=np.float64);q+=np.square(x,dtype=np.float64).sum((0,2))
        if not n:raise ValueError('No fitting samples')
        self.mean=s/n;self.scale=np.sqrt(np.maximum(q/n-self.mean**2,1e-8));return self

    def transform(self,x):return ((x-self.mean[None,:,None])/self.scale[None,:,None]).astype('float32')


def split_training_patients(patients,seed=41001):
    """Stable outcome-blind 80/10/10 fit/stop/calibration split within official train."""
    ids=np.array(sorted(set(patients)));rng=np.random.default_rng(seed);rng.shuffle(ids)
    a=int(.8*len(ids));b=int(.9*len(ids));roles={}
    for name,group in [('fit',ids[:a]),('stop',ids[a:b]),('calibration',ids[b:])]:
        keep=set(group);roles[name]=np.flatnonzero([p in keep for p in patients])
    if not all(len(v) for v in roles.values()):raise ValueError('Too few patients for development partitions')
    return roles


def patient_weights(patients,indices):
    from collections import Counter
    selected=[patients[int(i)] for i in indices];counts=Counter(selected)
    return np.array([1/counts[p] for p in selected],dtype=np.float32)
