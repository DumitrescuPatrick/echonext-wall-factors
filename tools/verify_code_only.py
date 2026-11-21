"""Offline allowlist, syntax and privacy-pattern checks for this source-only ZIP."""
from pathlib import Path
import json,hashlib,re,ast
ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.venv','venv','__pycache__'}
ALLOWED_EXTENSIONS={'.py','.json','.txt','.md'}
PROHIBITED_DIRS={'private','data','datasets','results','aggregate','tables','figures','publication','manuscript','supplement','weights','checkpoints','embeddings','predictions','cache','caches','logs'}
PATTERNS={
 'patient waveform path':r'(?:files/)?p\d{4}/p\d{8}/s\d{8}',
 'literal identifier assignment':r'''["']?(?:subject_id|patient_id|hadm_id|study_id|ecg_id)["']?\s*[:=]\s*["']?\d{5,}''',
 'credential assignment':r'''["']?(?:password|access_token|api_key|secret_key)["']?\s*[:=]\s*["'][^"']+["']''',
 'provider access key':r'AKIA[0-9A-Z]{16}',
 'personal absolute path':r'/Users/[^\s\"\']+',
}
def files():return sorted(p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&IGNORED and p!=ROOT/'audit/FILE_MANIFEST.json')
def inspect():
 problems=[]
 for p in files():
  rel=p.relative_to(ROOT)
  if p.is_symlink():problems.append((str(rel),'symlink'))
  if p.name!='.gitignore' and p.suffix not in ALLOWED_EXTENSIONS:problems.append((str(rel),'not source-only format'))
  if set(rel.parts)&PROHIBITED_DIRS:problems.append((str(rel),'prohibited directory'))
  t=p.read_text()
  if p.suffix=='.py':
   try:ast.parse(t)
   except SyntaxError:problems.append((str(rel),'Python syntax'))
  if p.resolve()!=Path(__file__).resolve():
   for label,pattern in PATTERNS.items():
    if re.search(pattern,t,re.I):problems.append((str(rel),label))
  if p.suffix=='.json':
   def check(v):
    if isinstance(v,dict):
     for k,x in v.items():
      if k.lower() in {'subject_id','subject_ids','patient_id','patient_ids','hadm_id','hadm_ids','study_ids','ecg_ids','patient_predictions','patient_records'}:problems.append((str(rel),'individual-level key'))
      check(x)
    elif isinstance(v,list):
     for x in v:check(x)
   check(json.loads(t))
 return problems

def main():
 expected={x['file']:x['sha256'] for x in json.loads((ROOT/'audit/FILE_MANIFEST.json').read_text())}
 actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
 if expected!=actual:raise SystemExit('FAIL: allowlist/hash mismatch; review local additions or changes')
 if inspect():raise SystemExit('FAIL: source-only/privacy check; sensitive values withheld')
 print(f'PASS: {len(actual)} code/config/documentation files; hashes, syntax, source-only formats and privacy patterns verified.')
if __name__=='__main__':main()
