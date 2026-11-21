"""Check whether source-to-float32 conversion creates waveform plateaus."""
import argparse,json
from pathlib import Path
import numpy as np
from echonext.data import open_waveforms
from .data import verify_freeze

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=Path.home()/'Downloads');p.add_argument('--root',type=Path,default=Path('private/stage2_v1/development_cache'));p.add_argument('--run',type=Path,default=Path('private/unsupervised_discovery_v1/classical_v1'));a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve precision audit')
    verify_freeze(a.run);source=open_waveforms(a.source/'EchoNext_val_waveforms.npy');cached=open_waveforms(a.root/'EchoNext_val_waveforms.npy');original_fraction=[];cached_fraction=[];created=0;removed=0;equal_bits=True
    for start in range(0,len(source),64):
        x=np.asarray(source[start:start+64]);z=np.asarray(cached[start:start+64]);equal_bits &= np.array_equal(x.astype('float32').view('uint32'),z.view('uint32'))
        s=np.diff(x[:,0],axis=1)==0;c=np.diff(z[:,0],axis=1)==0;created+=int((c&~s).sum());removed+=int((s&~c).sum());original_fraction.extend(s.mean((1,2)));cached_fraction.extend(c.mean((1,2)))
    if not equal_bits:raise ValueError('Float32 bit-pattern mismatch')
    result={'source_to_cache_bit_patterns_exact':equal_bits,'new_equal_adjacent_pairs_created_by_float32_conversion':created if created==0 or created>=10 else 'suppressed','equal_adjacent_pairs_removed_by_float32_conversion':removed if removed==0 or removed>=10 else 'suppressed','groups':[]}
    for method in ['spd','waveform','hilbert']:
        labels=np.load(a.run/f'{method}_62001.assignments.npy',allow_pickle=False)
        for k in [0,1]:result['groups'].append({'representation':method,'cluster':k,'source_equal_adjacent_fraction':float(np.array(original_fraction)[labels==k].mean()),'cached_equal_adjacent_fraction':float(np.array(cached_fraction)[labels==k].mean())})
    a.output.write_text(json.dumps(result,indent=2));print('Source/cache precision QC saved, aggregate only.')
if __name__=='__main__':main()
