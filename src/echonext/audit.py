"""Local-only aggregate audit. Test waveform QC is structural, never predictive."""
import argparse,hashlib,json,os,platform,shutil,sys
from pathlib import Path
import numpy as np
from .data import read_metadata,validate_splits,open_waveforms,split_training_patients
from .schema import LABELS,PRIMARY,SPLITS

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def scan_waveforms(a,chunk=64):
    bad=[];flat=0
    for start in range(0,len(a),chunk):
        x=a[start:start+chunk];ok=np.isfinite(x).all(axis=(1,2,3))
        bad.extend((start+np.flatnonzero(~ok)).tolist())
        finite=x[ok]
        if len(finite):flat+=int((np.ptp(finite[:,0],axis=1)==0).sum())
    return {'nonfinite_records':len(bad),'flat_lead_instances_in_finite_records':flat},np.array(bad,dtype=np.int64)

def audit(root,project):
    root,project=Path(root).expanduser(),Path(project);private=project/'private';reports=project/'reports'
    private.mkdir(exist_ok=True,mode=0o700);reports.mkdir(exist_ok=True)
    if (reports/'data_audit.json').exists():raise FileExistsError('Audit already frozen; use a new version')
    md,columns,serial_ok=read_metadata(root/'echonext_metadata_100k.csv')
    overlaps=validate_splits(md);groups={};files={};quarantine={};total=0
    for split in SPLITS:
        d=md[split];groups[split]={'records':len(d['rows']),'patients':len(set(d['patients']))}
        if split=='no_split':continue
        file=root/f'EchoNext_{split}_waveforms.npy';a=open_waveforms(file)
        if len(a)!=len(d['rows']):raise ValueError('Metadata/waveform row-count mismatch')
        qc,bad=scan_waveforms(a);quarantine[split]=bad
        groups[split].update(qc);groups[split]['most_recent_flag_not_one']=sum(v not in ('1','1.0') for v in d['recent'])
        files[file.name]={'bytes':file.stat().st_size,'sha256':sha(file),'shape':list(a.shape),'dtype':str(a.dtype),'data_bytes':a.nbytes}
        total+=a.nbytes
        print(json.dumps({'structural_scan_complete':split,'records':len(a),'nonfinite_records':len(bad)}),flush=True)
    metadata=root/'echonext_metadata_100k.csv';files[metadata.name]={'bytes':metadata.stat().st_size,'sha256':sha(metadata)}
    label_summaries=[]
    for split in ('train','val'):
        labels=np.asarray(md[split]['labels'],float);keep=np.ones(len(labels),bool);keep[quarantine[split]]=False
        groups[split]['primary_missing']=int(np.isnan(labels[:,0]).sum())
        for j,name in enumerate(LABELS):
            x=labels[:,j];known=np.isfinite(x)
            label_summaries.append({'split':split,'outcome':name,'records':len(x),'observed':int(known.sum()),'missing':int((~known).sum()),'positive':int(np.nansum(x)),'eligible_finite_waveform_observed':int((known&keep).sum())})
        complete=np.isfinite(labels).all(1)
        groups[split]['complete_component_label_rows']=int(complete.sum())
        groups[split]['composite_vs_component_or_disagreements_on_complete_rows']=int((labels[complete,0] != labels[complete,1:].max(1)).sum())
    roles=split_training_patients(md['train']['patients']);indices={}
    # Partition first, then quarantine: same population definitions for every learner.
    for name,idx in roles.items():
        good=np.setdiff1d(idx,quarantine['train']);indices[name]=good
        groups['train'][name]={'records_before_qc':len(idx),'records_after_qc':len(good),'patients_after_qc':len({md['train']['patients'][j] for j in good})}
    for split in ('train','val','test'):
        indices[split+'_metadata_rows']=np.asarray(md[split]['rows'],dtype=np.int64)
        indices[split+'_quarantine']=quarantine[split]
    np.savez_compressed(private/'alignment_and_partitions.npz',**indices)
    # Hash keys only in a private provenance artifact; never publish their values/digests.
    dump(private/'input_paths.json',{'root':str(root.resolve()),'partition_sha256':sha(private/'alignment_and_partitions.npz')})
    report={'stage':'1','status':'STRUCTURAL_AUDIT_COMPLETE_WITH_ALIGNMENT_LIMITS','metadata_records':sum(len(v['rows']) for v in md.values()),'groups':groups,'patient_overlap_counts':overlaps,'serialized_metadata_index_matches_file_position':serial_ok,'unique_ecg_keys':True,'alignment':'Split counts and original CSV order agree with the provider contract; NPY has no embedded ECG keys. Content-level row alignment cannot be independently proved without an authoritative per-waveform manifest.','test_policy':{'outcomes_accessed':False,'prevalence_computed':False,'model_inference_run':False,'allowed_access':'headers, byte integrity, finite/flat structural QC and patient-key disjointness only'},'lead_order':'12 channel positions verified; anatomical order not authenticated by these arrays','source_version':'Local content is compatible with additional fields documented in 1.1.x; exact release unverified without provider checksum/manifest.','upstream_scaling':'Already filtered/clipped/normalized. PhysioNet says dataset-wide scaling; official inference code says pretrained train parameters. Local normalization provenance unresolved; cannot recover raw amplitudes.','waveform_bytes_total':total,'no_split_excluded':True,'outputs_contain_identifiers':False}
    dump(reports/'data_audit.json',report);dump(reports/'label_audit_development_only.json',label_summaries)
    dump(reports/'input_integrity.json',files)
    import torch,importlib.metadata
    physical=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES') if 'SC_PHYS_PAGES' in os.sysconf_names else None
    resources={'architecture':platform.machine(),'cpu_count':os.cpu_count(),'physical_memory_bytes':physical,'free_disk_bytes':shutil.disk_usage(project).free,'array_data_bytes':total,'float32_cache_bytes_if_created':total//2,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'python':platform.python_version(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','torch','scikit-learn']},'training_run':False}
    dump(reports/'resources.json',resources)
    print(json.dumps({'audit_complete':True,'metadata_records':report['metadata_records'],'patient_overlap_counts':overlaps,'test_outcomes_accessed':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default=str(Path.home()/'Downloads'));p.add_argument('--project',default='.');a=p.parse_args()
    try:audit(a.data_dir,a.project)
    except Exception as e:
        # No traceback or exception payload containing patient values.
        print(json.dumps({'status':'FAILED','error_class':type(e).__name__,'message':'Local audit failed; no patient-level details disclosed.'}),file=sys.stderr);sys.exit(1)
