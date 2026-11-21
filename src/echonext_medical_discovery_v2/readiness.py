"""Report technical gates without launching clinical training."""
import json,platform,importlib.metadata as md
from pathlib import Path
import torch
from .models import CompactVAE,CompactSimCLR
from .train import device
from .latent import REPORT

def main():
    reason=None
    try:device()
    except RuntimeError as exc:reason=str(exc)
    report={'platform':platform.platform(),'packages':{n:md.version(n) for n in ['torch','numpy','scipy','scikit-learn']},'MPS_available':torch.backends.mps.is_available(),'MPS_block_reason':reason,'new_clinical_encoder_training_launched':False,'model_parameters':{n:sum(p.numel() for p in m.parameters()) for n,m in [('CompactVAE',CompactVAE()),('CompactSimCLR',CompactSimCLR())]},'validated_QRS_STT_reference_available':False,'EchoNext_actual_electrical_axis_available':False,'PTB_provider_axis_available_with_missingness':True,'native_ResNet_PTB_transfer_authenticated':False,'unit_SD_future_transfer':'Removes affine per-lead scale/offset, but filtering/clipping and domain differences remain; requires actual trained encoder and validation'}
    (REPORT/'NEURAL_AND_PHYSIOLOGY_READINESS.json').write_text(json.dumps(report,indent=2));print('Readiness/gates saved; no clinical training launched.')
if __name__=='__main__':main()
