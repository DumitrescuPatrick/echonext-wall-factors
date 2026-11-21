from pathlib import Path
import unittest,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import verify_code_only as verify
class CodeRelease(unittest.TestCase):
 def test_source_only_privacy_and_syntax(self):self.assertEqual(verify.inspect(),[])
 def test_original_source_unchanged(self):
  for name,digest in json.loads((ROOT/'audit/ORIGINAL_CODE_HASHES.json').read_text()).items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
 def test_no_publication_or_result_files(self):
  for p in verify.files():
   self.assertIn(p.suffix,{'.py','.md','.json','.txt',''})
   self.assertFalse(set(p.relative_to(ROOT).parts)&verify.PROHIBITED_DIRS)
 def test_json_is_configuration_or_release_metadata(self):
  for p in verify.files():
   if p.suffix=='.json':self.assertIn(p.relative_to(ROOT).parts[0],{'config','audit'})
if __name__=='__main__':unittest.main()
