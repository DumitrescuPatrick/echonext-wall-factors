"""Versioned final audit. Clinical values remain local; outputs are aggregates."""
import csv,json
from pathlib import Path
import numpy as np
from .data import read_metadata,validate_splits,binary
from .stage2_train import sha,dump,OUT,PRIVATE

def main():
    root=Path.home()/'Downloads';old=json.loads(Path('reports/input_integrity.json').read_text())
    verified={}
    for name,info in old.items():
        match=sha(root/name)==info['sha256']
        if not match:raise ValueError('Source file changed from Stage 1')
        verified[name]=True
    md,_,serial=read_metadata(root/'echonext_metadata_100k.csv');overlaps=validate_splits(md)
    used=set().union(*(set(md[s]['rows']) for s in ('train','val','test')))
    excluded=set(range(sum(len(d['rows']) for d in md.values())))-used
    no_split=set(md['no_split']['rows']);assert excluded==no_split
    excluded_groups={s:len(set(md['no_split']['patients'])&set(md[s]['patients'])) for s in ('train','val','test')}
    data={'files_unchanged_from_stage1':verified,'excluded_records':len(excluded),'all_excluded_records_designated_no_split':True,'serial_index_matches_original_csv_position':serial,'split_patient_overlaps':overlaps,'no_split_patient_overlaps_with_used_splits':excluded_groups,'no_split_used_for_any_learning':False,'download_provenance':'All four macOS WhereFroms attributes point to PhysioNet EchoNext 1.1.1. Corroborates original URLs, not a cryptographic provider checksum.','alignment':'Original CSV filter-by-split order plus count/index/uniqueness consistency and provider guarantee; NPY cannot independently identify each ECG. No sorting/reindexing of metadata performed.','lead_order_provider_code':['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],'lead_order_verification':'Official extraction implementation plus original release provenance; no independent per-waveform lead names embedded.','preprocessing_provider_code':{'sampling_hz':250,'seconds':10,'baseline_removal':'two-stage per-lead median baseline (51 and 151 points), subtract baseline','clipping':'per-lead fixed supplied lower/upper bounds','scaling':'subtract supplied per-lead mean; divide supplied per-lead std','local_reprocessing':'none; already normalized delivered arrays'},'normalization_limit':'Public inference code consumes training/pretrained parameters; dataset page says dataset-wide statistics. Exact export parameter provenance is not independently authenticated; no raw-voltage interpretation or end-to-end leakage-free claim.','primary_label':'supplied shd_moderate_or_greater_flag; composite of the 11 supplied condition flags; no relabeling','test_outcomes_decoded':False,'test_predictions_or_metrics_computed':False}
    groups={s:[] for s in ('train','val')}
    with (root/'echonext_metadata_100k.csv').open() as f:
        for r in csv.DictReader(f):
            s=r['split']
            if s not in groups:continue
            try:v=float(r['tr_max_velocity_value'])
            except (TypeError,ValueError):v=np.nan
            flag=binary(r['tr_max_gte_32_flag']);groups[s].append((v,flag))
    tr=[]
    for s,values in groups.items():
        z=np.asarray(values,float);known=np.isfinite(z).all(1);x=z[known,0];flag=z[known,1]
        row={'split':s,'paired_numeric_and_flag':len(x),'numeric_missing':int((~np.isfinite(z[:,0])).sum()),'numeric_quantiles':dict(zip(['min','q25','median','q75','max'],np.quantile(x,[0,.25,.5,.75,1]).tolist()))}
        for cutoff in [.32,3.2,32.]:row[f'flag_disagreements_threshold_{cutoff:g}']=int(((x>=cutoff)!=flag).sum())
        tr.append(row)
    data['tr_velocity_investigation']={'development_only':tr,'source_error':'3.2 m/s equals 320 cm/s, not 32 cm/s. Prose/flag shorthand alone must not silently redefine the target.','decision':'Preserve every provided component and composite flag. Development cross-field consistency informs a likely unit/decimal documentation error, not authority to rewrite labels. Original generator/clinical confirmation remains desirable.'}
    dump(OUT/'final_data_audit.json',data)
    print(json.dumps({'final_audit':'COMPLETE','excluded_no_split_records':len(excluded),'official_partition_patient_overlap':overlaps,'source_files_hash_match':True,'tr_development_aggregates':tr,'test_targets_accessed':False}),flush=True)

if __name__=='__main__':main()
