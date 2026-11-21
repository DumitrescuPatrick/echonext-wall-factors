"""Aggregate development metadata availability; official test fields skipped."""
import argparse,csv,json
from pathlib import Path
from .data import sha
from echonext.schema import LABELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,default=Path('private/stage2_v1/development_cache/echonext_metadata_100k.csv'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve schema audit')
    expected=json.loads(Path('reports/stage2_v1/development_cache_integrity.json').read_text())['echonext_metadata_100k.csv']['sha256']
    if sha(a.metadata)!=expected:raise ValueError('Metadata hash changed')
    totals={'train':0,'val':0};missing={};ecg={'ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected'};demographic={'age_at_ecg','sex','race_ethnicity'}
    with a.metadata.open(newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames
        for name in columns:missing[name]={'train':0,'val':0}
        for r in reader:
            split=r['split']
            if split not in totals:continue
            totals[split]+=1
            for name in columns:
                if r[name] in ('','NaN','nan','NA','None','null'):missing[name][split]+=1
    rows=[]
    for name in columns:
        if name in ('','ecg_key','patient_key'):role='private linkage/index; never model input'
        elif name=='split' or name=='most_recent_ecg':role='partition/administrative; never predictor'
        elif name in LABELS:role='echo-derived outcome; never representation input'
        elif name in ecg:role='ECG machine measurement; descriptive morphology audit'
        elif name in demographic:role='demographic context; confounding/subgroup audit'
        elif name in ('acquisition_year','location_setting'):role='acquisition context; confounding audit'
        else:role='echo reference measurement; exclude ECG representations/predictors'
        for split in totals:rows.append({'field':name or '(serialized index)','role':role,'split':split,'records':totals[split],'missing':missing[name][split],'missing_fraction':missing[name][split]/totals[split]})
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Aggregate metadata schema/availability audit saved; no test outcomes inspected.')
if __name__=='__main__':main()
