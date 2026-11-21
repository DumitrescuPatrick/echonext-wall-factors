"""Optional private float32 development cache. Never open/copy test waveforms."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import open_waveforms

def create(root,destination):
    root,destination=Path(root).expanduser(),Path(destination)
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    for split in ['train','val']:
        name=f'EchoNext_{split}_waveforms.npy';target=destination/name
        if target.exists():raise FileExistsError('Cache exists; never overwrite an active input')
        source=open_waveforms(root/name)
        out=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=source.shape)
        for start in range(0,len(source),128):out[start:start+128]=source[start:start+128]
        out.flush();del out;out=open_waveforms(target)
        for start in range(0,len(source),128):
            if not np.array_equal(source[start:start+128].astype('float32').view('uint32'),out[start:start+128].view('uint32')):raise ValueError('Float32 cache mismatch; values withheld')
    (destination/'echonext_metadata_100k.csv').symlink_to((root/'echonext_metadata_100k.csv').resolve())
    print(json.dumps({'private_development_cache_created':True,'all_converted_values_checked':True,'test_cache_created':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path.home()/'Downloads');p.add_argument('--destination',type=Path,default=Path('private/stage2_v1/development_cache'));a=p.parse_args();create(a.data_dir,a.destination)
