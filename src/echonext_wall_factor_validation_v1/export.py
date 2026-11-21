"""Explicit public allowlist: source/methods/aggregates only, no private artifacts."""
import json,zipfile
from pathlib import Path
from xml.etree import ElementTree
import importlib.metadata
from echonext_unsupervised_discovery_v1.data import sha
from .run import OUT

def main():
    old=json.loads(Path('reports/medical_discovery_next_phase/PACKAGE_MANIFEST.json').read_text())
    for p,h in old['files'].items():
        if sha(p)!=h:raise ValueError('Previous study changed: '+p)
    verification=json.loads((OUT/'VERIFICATION.json').read_text());verification['previous_next_phase_package_files_unchanged']=len(old['files']);verification['software_versions']={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','matplotlib','joblib','threadpoolctl']}
    with zipfile.ZipFile(OUT/'WALL_FACTOR_VALIDATION_REPORT.docx') as z:
        if z.testzip():raise ValueError('Invalid Word ZIP')
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):ElementTree.fromstring(z.read(name))
    verification['Word_XML_valid']=True;(OUT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2))
    files=[]
    for folder in ['src/echonext_wall_factor_validation_v1','src/echonext_medical_discovery_v2','src/echonext_unsupervised_discovery_v1','src/echonext_covariance_validation_v1']:
        files+=list(Path(folder).glob('*.py'))
    for folder in ['config/wall_factor_validation_v1','docs/wall_factor_validation_v1','reports/wall_factor_validation_v1','tests/wall_factor_validation_v1']:
        files += [p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json']
    files += [Path(p) for p in ['src/echonext/__init__.py','src/echonext/data.py','src/echonext/stage2_models.py','docs/medical_discovery_next_phase/RUN_NATIVE_MPS.md','config/medical_discovery_next_phase/protocol.json','config/medical_discovery_next_phase/requirements.txt']]
    files=sorted(set(files));allowed={'.py','.json','.csv','.md','.pdf','.png','.docx','.txt'}
    if any(p.suffix not in allowed or 'private' in p.parts for p in files):raise ValueError('Nonpublic file rejected')
    manifest={'status':'Exploratory development follow-up; independent echo validation and clinical SSL fitting pending','patient_data_or_weights_included':False,'files':{str(p):sha(p) for p in files}};(OUT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(OUT/'PACKAGE_MANIFEST.json');target=Path('EchoNext_Wall_Factor_Followup_v1.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p))
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Archive corrupt')
        for p,h in manifest['files'].items():
            import hashlib
            if hashlib.sha256(z.read(p)).hexdigest()!=h:raise ValueError('Archive manifest mismatch')
    print(f'Verified aggregate/source archive: {len(files)} files, {target.stat().st_size} bytes; previous package unchanged.')

if __name__=='__main__':main()
