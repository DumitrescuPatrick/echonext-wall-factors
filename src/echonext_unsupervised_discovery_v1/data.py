"""Allowed-column metadata projection. Outcomes are inaccessible during fitting."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from echonext.data import open_waveforms, canonical_batch

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def linkage(path):
    result={s:[] for s in ('train','val','test','no_split')}; keys=set()
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            s=row['split']; p=row['patient_key']; e=row['ecg_key']
            if s not in result or not p or not e or e in keys: raise ValueError('Invalid private linkage; values withheld')
            keys.add(e); result[s].append(p)
    groups={s:set(v) for s,v in result.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('Patient overlap')
    if any(len(result[s])!=len(groups[s]) for s in ('val','test')): raise ValueError('Repeated evaluation patient')
    return result

def roles(patients,seed=62001):
    ids=np.array(sorted(set(patients))); np.random.default_rng(seed).shuffle(ids)
    stop=set(ids[:max(1,int(.1*len(ids)))])
    fit=np.flatnonzero([p not in stop for p in patients]); monitor=np.flatnonzero([p in stop for p in patients])
    if not len(fit): raise ValueError('Insufficient patients')
    # One deterministic recording per patient for cluster fitting, never outcome-selected.
    seen=set(); representatives=[]
    for i in fit:
        if patients[i] not in seen: representatives.append(i); seen.add(patients[i])
    return fit,monitor,np.array(representatives)

class Waves(Dataset):
    def __init__(self,root,split,indices=None):
        if split not in ('train','val'): raise PermissionError('Official test and no_split are inaccessible')
        self.patients=linkage(Path(root)/'echonext_metadata_100k.csv')[split]
        self.array=open_waveforms(Path(root)/f'EchoNext_{split}_waveforms.npy')
        if len(self.array)!=len(self.patients): raise ValueError('Row alignment mismatch')
        self.indices=np.arange(len(self.array)) if indices is None else np.asarray(indices)
        mapping={p:i for i,p in enumerate(sorted(set(self.patients)))}
        self.groups=np.array([mapping[p] for p in self.patients])
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        row=int(self.indices[i]); return canonical_batch(self.array[row:row+1])[0],int(self.groups[row])

def freeze(directory,files,details):
    path=Path(directory)/'FREEZE.json'
    if path.exists(): raise FileExistsError('Definitions already frozen; create a new version')
    manifest={'files':{str(Path(f).resolve()):sha(f) for f in files},'details':details}
    path.write_text(json.dumps(manifest,indent=2)); return sha(path)

def verify_freeze(directory):
    path=Path(directory)/'FREEZE.json'; m=json.loads(path.read_text())
    if any(sha(p)!=h for p,h in m['files'].items()): raise ValueError('Frozen artifact changed')
    return m
