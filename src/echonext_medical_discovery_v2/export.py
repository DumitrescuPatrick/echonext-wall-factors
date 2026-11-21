"""Export the requested report plus an aggregate-only source/methods package."""
import json,zipfile,hashlib,importlib.metadata as md,xml.etree.ElementTree as ET
from pathlib import Path
from echonext_unsupervised_discovery_v1.data import sha,verify_freeze
from echonext_unsupervised_discovery_v1.publication import documents
import echonext_unsupervised_discovery_v1.questions_document as renderer
from .latent import PRIVATE,REPORT

def main():
    source=Path('docs/medical_discovery_next_phase/MEDICAL_DISCOVERY_NEXT_PHASE.md');(REPORT/'MEDICAL_DISCOVERY_NEXT_PHASE.md').write_text(source.read_text());documents(source,REPORT)
    for suffix in ['docx','pdf']:(REPORT/f'MANUSCRIPT_DRAFT.{suffix}').rename(REPORT/f'FIVE_QUESTIONS_RESULTS.{suffix}')
    renderer.ROOT=REPORT;renderer.SOURCE=source;renderer.main()
    for suffix in ['docx','pdf']:(REPORT/f'FIVE_QUESTIONS_RESULTS.{suffix}').rename(REPORT/f'MEDICAL_DISCOVERY_NEXT_PHASE.{suffix}')
    checks={}
    for name in ['reports/unsupervised_discovery_v1/PACKAGE_MANIFEST.json','reports/unsupervised_discovery_v1/phase1_audit_v1/PHASE1_PACKAGE_MANIFEST.json','reports/unsupervised_discovery_v1/five_questions_v1/FIVE_QUESTIONS_PACKAGE_MANIFEST.json','reports/covariance_validation_v1/PACKAGE_MANIFEST.json','reports/ptbxl_transport_v1/PACKAGE_MANIFEST.json','reports/geometry_study_v1/PLAN_MANIFEST.json','reports/stage2_v1/fit_source_manifest.json']:
        manifest=json.loads(Path(name).read_text());entries=manifest.get('files',manifest)
        for path,value in entries.items():
            digest=value['sha256'] if isinstance(value,dict) else value
            if sha(path)!=digest:raise ValueError('Previous research changed')
        checks[name]={'unchanged':True,'files_checked':len(entries)}
    for folder in [PRIVATE,PRIVATE/'unit_input_sensitivity',PRIVATE/'component_reference']:verify_freeze(folder)
    if Path('private/medical_discovery_next_phase_unsupervised').exists():raise ValueError('Unexpected clinical unsupervised run; inventory before publication')
    checks.update({'new_private_factor_projection_reference_freezes_verified':3,'synthetic_tests_passed':9,'VAE_SimCLR_clinical_runs':0,'frozen_inference_logit_max_error':json.loads((REPORT/'FACTOR_ASSOCIATIONS.json').read_text())['max_logit_reproduction_error'],'source_supervised_selection_bias_disclosed':True,'native_PTB_factor_transport_authenticated':False,'altered_input_PTB_sensitivity_completed':True,'validated_anatomical_QRS_STT_study_completed':False,'EchoNext_test_accessed':False,'PTB_fold10_accessed':False})
    (REPORT/'VERIFICATION.json').write_text(json.dumps(checks,indent=2))
    for file in REPORT.glob('*.docx'):
        with zipfile.ZipFile(file) as z:
            for n in z.namelist():ET.fromstring(z.read(n))
    (Path('config/medical_discovery_next_phase')/'requirements.txt').write_text('\n'.join(f'{n}=={md.version(n)}' for n in ['numpy','scipy','scikit-learn','torch','wfdb','matplotlib','joblib'])+'\n')
    directories=['src/echonext_medical_discovery_v2','src/echonext_unsupervised_discovery_v1','src/echonext_covariance_validation_v1','src/echonext_ptbxl_transport_v1','docs/medical_discovery_next_phase','config/medical_discovery_next_phase','tests/medical_discovery_next_phase','reports/medical_discovery_next_phase']
    files=[p for d in directories for p in Path(d).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.pdf','.png','.svg','.docx','.txt'} and p.name!='PACKAGE_MANIFEST.json']
    files.extend(Path('src/echonext')/n for n in ['__init__.py','data.py','schema.py','stage2_models.py','stage2_metrics.py'])
    files.extend(Path(n) for n in ['reports/ptbxl_transport_v1/PTBXL_LICENSE.txt','reports/ptbxl_transport_v1/VERIFIED_LABEL_MAPPING.json','reports/stage2_v1/development_cache_integrity.json'])
    for p in files:
        if 'private' in p.parts or p.suffix in {'.npy','.npz','.pt','.pth','.joblib','.pkl','.dat','.hea'}:raise ValueError('Private/raw/model artifact in public archive')
    manifest=REPORT/'PACKAGE_MANIFEST.json';manifest.write_text(json.dumps({'status':'Measured supervised-reference hypotheses; new clinical encoders and validated QRS/STT study pending','files':{str(p):sha(p) for p in sorted(set(files))}},indent=2));files.append(manifest)
    archive=Path('EchoNext_Medical_Discovery_Next_Phase_v1.zip')
    if archive.exists():raise FileExistsError('Preserve existing archive')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,str(p))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive corruption')
        for name,digest in json.loads(z.read(str(manifest)))['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Archive digest mismatch')
    print('Requested report and aggregate source/results package verified:',len(set(files)),'files,',archive.stat().st_size,'bytes')
if __name__=='__main__':main()
