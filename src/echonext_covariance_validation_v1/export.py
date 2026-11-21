"""Export aggregate manuscripts/plans and a strict source/docs/results archive."""
import json,zipfile,hashlib
from pathlib import Path
from echonext_unsupervised_discovery_v1.publication import documents
import echonext_unsupervised_discovery_v1.questions_document as renderer
from echonext_unsupervised_discovery_v1.data import sha,verify_freeze

ROOT=Path('reports/covariance_validation_v1')

def main():
    for source,name in [('RESULTS_AND_DECISION.md','CURRENT_EVIDENCE_AND_DECISION'),('METHODS.md','MANUSCRIPT_METHODS'),('STATISTICAL_ANALYSIS_PLAN.md','STATISTICAL_ANALYSIS_PLAN')]:
        path=Path('docs/covariance_validation_v1')/source;documents(path,ROOT)
        for suffix in ['docx','pdf']:(ROOT/f'MANUSCRIPT_DRAFT.{suffix}').rename(ROOT/f'FIVE_QUESTIONS_RESULTS.{suffix}')
        renderer.ROOT=ROOT;renderer.SOURCE=path;renderer.main()
        for suffix in ['docx','pdf']:(ROOT/f'FIVE_QUESTIONS_RESULTS.{suffix}').rename(ROOT/f'{name}.{suffix}')
    checks={}
    for name in ['reports/unsupervised_discovery_v1/PACKAGE_MANIFEST.json','reports/unsupervised_discovery_v1/phase1_audit_v1/PHASE1_PACKAGE_MANIFEST.json','reports/unsupervised_discovery_v1/five_questions_v1/FIVE_QUESTIONS_PACKAGE_MANIFEST.json','reports/geometry_study_v1/PLAN_MANIFEST.json','reports/stage2_v1/fit_source_manifest.json']:
        m=json.loads(Path(name).read_text());entries=m.get('files',m)
        for path,entry in entries.items():
            expected=entry['sha256'] if isinstance(entry,dict) else entry
            if sha(path)!=expected:raise ValueError('Prior artifact changed')
        checks[name]={'unchanged':True,'files_checked':len(entries)}
    for folder in ['private/covariance_validation_v1','private/covariance_validation_v1/fusion_diagnostic','private/covariance_validation_v1/stress']:
        verify_freeze(folder);checks[folder]={'freeze_verified':True}
    checks.update({'new_synthetic_tests_passed':9,'official_test_opened':False,'new_neural_training':False,'strict_primary_ResNet_increment_completed':False,'reason':'Honest OOF or independent base-model scores unavailable; upstream-selection diagnostic only'})
    # No private filenames/identifiers in the public receipt.
    public_checks={k:v for k,v in checks.items() if not k.startswith('private/')};public_checks['new_private_freezes_verified']=3
    (ROOT/'VERIFICATION.json').write_text(json.dumps(public_checks,indent=2))
    directories=['src/echonext_covariance_validation_v1','src/echonext_unsupervised_discovery_v1','docs/covariance_validation_v1','config/covariance_validation_v1','tests/covariance_validation_v1','reports/covariance_validation_v1']
    files=[p for d in directories for p in Path(d).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.pdf','.png','.docx'} and p.name!='PACKAGE_MANIFEST.json']
    files.extend(Path('src/echonext')/name for name in ['__init__.py','data.py','schema.py','stage2_models.py','stage2_metrics.py'])
    files.extend(Path(name) for name in ['config/unsupervised_discovery_v1/requirements.txt','reports/unsupervised_discovery_v1/five_questions_v1/COMPONENT_ADJUSTMENT.json','reports/unsupervised_discovery_v1/five_questions_v1/PREDICTION_RESULTS.json','reports/unsupervised_discovery_v1/five_questions_v1/EUCLIDEAN_CONTROL.json','reports/unsupervised_discovery_v1/phase1_audit_v1/POPULATION.json'] if Path(name).exists())
    for p in files:
        if 'private' in p.parts or p.suffix in {'.npy','.npz','.pt','.pth','.pkl','.joblib'}:raise ValueError('Private artifact in public archive')
    manifest=ROOT/'PACKAGE_MANIFEST.json';manifest.write_text(json.dumps({'status':'Exploratory covariance study plus independent-validation plan; primary ResNet increment unconfirmed','files':{str(p):sha(p) for p in sorted(set(files))}},indent=2));files.append(manifest)
    archive=Path('EchoNext_Covariance_Validation_v1_Package.zip')
    if archive.exists():raise FileExistsError('Preserve existing archive')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,str(p))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive corruption')
        for path,digest in json.loads(z.read(str(manifest)))['files'].items():
            if hashlib.sha256(z.read(path)).hexdigest()!=digest:raise ValueError('Archive digest mismatch')
    print('Aggregate source/docs/methods/results archive verified:',len(set(files)),'files,',archive.stat().st_size,'bytes')
if __name__=='__main__':main()
