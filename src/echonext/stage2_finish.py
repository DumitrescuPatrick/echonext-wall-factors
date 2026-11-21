"""Independent replay, baseline freeze, then preparation of untrained comparisons."""
import csv,json,hashlib,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from .stage2_train import OUT,PRIVATE,protocol,selected_candidate,sha,dump,development_inputs,predict_logits
from .stage2_models import ResNet1D

def freeze_baseline():
    c=protocol();target=OUT/'BASELINE_FROZEN.json'
    if target.exists():raise FileExistsError('Baseline already frozen')
    # Original research and Stage 1 payloads must remain byte-identical.
    previous=json.loads(Path('private/previous_research_before.json').read_text())
    for p,h in previous.items():
        if sha(Path.cwd().parent/p)!=h:raise ValueError('Previous research changed; details withheld')
    stage1=json.loads((PRIVATE/'stage1_preservation.json').read_text())
    for p,h in stage1.items():
        if sha(p)!=h:raise ValueError('Stage 1 frozen artifact changed')
    inputs_before=json.loads(Path('reports/input_integrity.json').read_text())
    for name,info in inputs_before.items():
        if sha(Path.home()/'Downloads'/name)!=info['sha256']:raise ValueError('Original input changed during fitting')
    cache_before=json.loads((OUT/'development_cache_integrity.json').read_text())
    for name,info in cache_before.items():
        if sha(PRIVATE/'development_cache'/name)!=info['sha256']:raise ValueError('Development cache changed during fitting')
    selected=selected_candidate(c);torch.set_num_threads(1)
    inputs=development_inputs(PRIVATE/'development_cache');wave,y=inputs['val']
    with (OUT/'validation_metrics_by_seed.csv').open() as f:rows=list(csv.DictReader(f))
    replayed=0;hashes={}
    for seed in c['final_seeds']:
        r=json.loads((OUT/f'seed_{seed}_complete.json').read_text())
        if r['candidate_index']!=selected['candidate_index']:raise ValueError('Final selected configurations differ')
        folder=PRIVATE/f'seed_{seed}';z=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model=ResNet1D(z['width']);model.load_state_dict(z['state_dict']);logits=predict_logits(model,wave)
        with np.load(folder/'validation_predictions.npz') as stored:
            if not np.array_equal(stored['y'],y) or not np.allclose(stored['logits'],logits,rtol=0,atol=1e-6):raise ValueError('Independent validation replay mismatch')
        row=next(v for v in rows if int(v['seed'])==seed);p=expit(logits)
        for name,value in [('auc',roc_auc_score(y,p)),('average_precision',average_precision_score(y,p)),('brier',brier_score_loss(y,p))]:
            if not np.isclose(float(row[name]),value,atol=1e-8):raise ValueError('Independent aggregate metric mismatch')
        if sha(folder/'best.pt')!=r['best_checkpoint_sha256']:raise ValueError('Final weights changed')
        hashes[str(seed)]=r['best_checkpoint_sha256'];replayed+=1
    tests=[]
    for folder,expected in [('tests',24),('tests/stage2_v1',9)]:
        r=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p','test_*.py'],capture_output=True,text=True)
        if r.returncode or f'Ran {expected} tests' not in r.stderr:raise ValueError('Software verification failed')
        tests.append({'folder':folder,'tests':expected,'status':'PASS','fixtures':'synthetic only'})
    dump(OUT/'independent_validation.json',{'status':'PASS','clinical_checkpoint_replays':replayed,'metric_cells_checked':replayed*3,'n_validation':len(y),'positive_validation':int(y.sum()),'tests':tests,'previous_research_files_unchanged':len(previous),'stage1_artifacts_unchanged':len(stage1),'test_target_analysis':False,'test_predictions':False,'alignment_limit':'replay verifies computational row handling conditional on provider waveform row guarantee, not physiological row authenticity'})
    payload={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.md','.png','.pdf'} and p.name!='BASELINE_FROZEN.json'}
    freeze={'status':'FROZEN_VALIDATION_BASELINE','selected_candidate':selected['candidate_index'],'selected_configuration':selected['configuration'],'final_seeds':c['final_seeds'],'n_train':72475,'n_validation':4626,'model_checkpoint_sha256':hashes,'report_payload_sha256':payload,'fit_source_manifest_sha256':sha(OUT/'fit_source_manifest.json'),'validation_operating_rules_sha256':sha(PRIVATE/'final_operating_rules.json'),'test':'SEALED; NO TARGET ANALYSIS OR MODEL EVALUATION','selection_warning':'Validation used for stopping, tuning and cutoffs; no claim of unbiased independent performance','freeze_source_sha256':sha(__file__)}
    dump(target,freeze)
    print(json.dumps({'baseline_frozen':True,'five_checkpoints_replayed':True,'test_remains_sealed':True}),flush=True)

def package():
    freeze=json.loads((OUT/'BASELINE_FROZEN.json').read_text())
    for name,h in freeze['report_payload_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Frozen baseline report changed')
    plan=OUT/'geometric_comparison_plan.json'
    if not plan.exists():raise ValueError('Prepare comparisons after baseline freeze')
    final={'status':'STAGE2_COMPLETE','baseline_freeze_sha256':sha(OUT/'BASELINE_FROZEN.json'),'geometric_plan_sha256':sha(plan),'test_remains_sealed':True,'geometric_models_trained':False,'publication_claim':'validation baseline only; no external/test superiority result'}
    dump(OUT/'STAGE2_COMPLETE.json',final)
    target=Path('EchoNext_Stage2_ResNet_Validation_Package.zip')
    files=[p for folder in ['src','tests','config','docs'] for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [Path('pyproject.toml'),Path('requirements-lock.txt'),Path('.gitignore')]
    # Share aggregates and code, never private arrays/cache/checkpoints or absolute-path manifests.
    reports=[p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.md','.png','.pdf'} and p.name!='fit_source_manifest.json']
    manifest={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+reports:
            name=str(p);z.write(p,name);manifest[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        z.writestr('START_HERE.md','''# EchoNext Stage 2

Open reports/stage2_v1/VALIDATION_REPORT.md for numerical findings and all caveats. docs/stage2_v1/FINAL_DATA_AUDIT.md records provider-dependent versus independently checked claims. docs/stage2_v1/PROTOCOL_AMENDMENT.md explains changes to Stage 1. Aggregate CSVs/figures, model code, synthetic tests and frozen matching-compute plans are included. No restricted records, metadata, patient predictions, model checkpoints or private caches are included. The baseline used training data for fitting and validation for stopping/selection/cutoffs; this is development performance. The official test remains sealed. Geometric comparison prototypes are prepared, not clinically trained. No database credentials or external telemetry are present. Exact local reproduction needs separately authorized external data and the private cache/row-handling records; public code has synthetic fixtures only.
''')
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Release ZIP integrity failed')
        if any(n.startswith('private/') or Path(n).suffix in {'.npy','.npz','.pt','.pth','.pkl','.parquet'} for n in z.namelist()):raise ValueError('Restricted artifact included')
        for name,entry in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=entry['sha256']:raise ValueError('ZIP manifest mismatch')
    print(json.dumps({'stage2_complete':True,'source_and_aggregate_zip':str(target),'bytes':target.stat().st_size,'test_performance_evaluated':False}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='package':package()
    else:freeze_baseline()
