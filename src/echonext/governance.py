"""Stage 1 source/test lock, preservation verification and explicit source release."""
import hashlib,itertools,json,subprocess,sys,zipfile,re
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_protocol(c):
    from .models import FAMILIES
    if c['families']!=list(FAMILIES):raise ValueError('Unplanned model family')
    candidates=list(itertools.product(c['tuning_grid']['learning_rate'],c['tuning_grid']['weight_decay']))
    if len(set(candidates))!=c['trials_per_family'] or len(candidates)!=6:raise ValueError('Unequal tuning budget')
    if len(set(c['final_training_seeds']))!=5:raise ValueError('Final seeds not distinct')
    if c['official_splits']['test']!='locked' or c['official_splits']['no_split']!='excluded':raise ValueError('Test/no_split policy changed')
    if c['coherence_beta_primary']!=.25:raise ValueError('Primary beta changed; amend protocol')
    return candidates

def freeze():
    root=Path('.');receipt=root/'reports/STAGE1_COMPLETE.json'
    if receipt.exists():raise FileExistsError('Stage 1 already frozen')
    c=json.loads((root/'config/stage1_protocol.json').read_text());validate_protocol(c)
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py'],capture_output=True,text=True)
    if r.returncode:raise ValueError('Synthetic software tests failed; no patient details disclosed')
    n=int(re.search(r'Ran (\d+) tests',r.stderr).group(1))
    (root/'reports/software_tests.json').write_text(json.dumps({'status':'PASS','tests':n,'fixtures':'artificial only; no clinical files opened by tests'},indent=2)+'\n')
    old=json.loads((root/'private/previous_research_before.json').read_text())
    for name,h in old.items():
        if sha(Path.cwd().parent/name)!=h:raise ValueError('Previous research file changed; path/details withheld')
    data=json.loads((root/'reports/data_audit.json').read_text())
    if any(data['patient_overlap_counts'].values()):raise ValueError('Split overlap')
    if data['test_policy']['outcomes_accessed'] or data['test_policy']['model_inference_run']:raise ValueError('Test lock violated')
    files=[p for d in ['src','tests','docs','config'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [root/'README.md',root/'pyproject.toml',root/'.gitignore',root/'requirements-lock.txt']
    manifest={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
    (root/'reports/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lock={'status':'LOCKED_FOR_STAGE1','clinical_training_run':False,'test_evaluation_run':False,'allowed_test_access':'integrity and split separation only','target_handling':'CSV bytes are parsed for row/split keys; test target fields are never requested, decoded or summarized','development_loader_rejects_test':True,'release_requires':'future sealed final-model/config/calibration/threshold manifest and one evaluation batch','source_manifest_sha256':sha(root/'reports/source_manifest.json'),'input_integrity_sha256':sha(root/'reports/input_integrity.json')}
    (root/'reports/test_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    complete={'status':'STAGE1_COMPLETE_WITH_DOCUMENTED_LIMITS','synthetic_software_tests':n,'model_families':8,'clinical_model_training':False,'test_target_analysis':False,'previous_research_files_unchanged':len(old),'unresolved':['local release authenticity','content-level waveform/row alignment beyond provider contract','anatomical channel order of local export','upstream normalization population','TR velocity documentation units','project-specific author ethics/DUA determination'],'source_manifest_sha256':sha(root/'reports/source_manifest.json'),'test_lock_sha256':sha(root/'reports/test_lock.json')}
    receipt.write_text(json.dumps(complete,indent=2)+'\n')
    # Synthetic tests/code/docs only; no local-only reports, mappings, raw data or models.
    target=root/'EchoNext_Stage1_Source_Protocol.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p))
        z.write(root/'reports/source_manifest.json','SOURCE_MANIFEST.json')
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Source ZIP failed integrity check')
        for name,entry in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=entry['sha256']:raise ValueError('Source ZIP hash mismatch')
    print(json.dumps({**complete,'source_zip_bytes':target.stat().st_size}))

if __name__=='__main__':freeze()
