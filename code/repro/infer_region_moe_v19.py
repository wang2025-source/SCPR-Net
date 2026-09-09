#!/usr/bin/env python3
"""MSRS inference for Region-MoE V19."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nets.Ufuser_region_moe_v19 import UfuserRegionMoEV19
from repro.infer_v10_region_detail_v12 import build as build_v12

def build(ck,device):
 cfg=ck['config'];base_ck=torch.load(cfg['init_v12'],map_location='cpu');v12=build_v12(base_ck,device);m=UfuserRegionMoEV19(v12,int(cfg.get('classes',9)),int(cfg.get('channels',48)),int(cfg.get('region',8)),float(cfg.get('max_refine',.1))).to(device);m.load_state_dict(ck['model'],strict=True);m.eval();return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--ir-dir',type=Path,required=True);p.add_argument('--vi-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');o=p.parse_args();device=torch.device(o.device);m=build(torch.load(o.model,map_location='cpu'),device);o.output_dir.mkdir(parents=True,exist_ok=True);paths=sorted(o.ir_dir.glob('*.png'))
 if not paths:raise RuntimeError(f'No PNG under {o.ir_dir}')
 with torch.inference_mode():
  for path in tqdm(paths,desc='V19 inference'):
   ir=cv2.imread(str(path),0);bgr=cv2.imread(str(o.vi_dir/path.name),1)
   if ir is None or bgr is None:raise FileNotFoundError(path.name)
   vi=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb)[...,0];h,w=ir.shape;a=torch.from_numpy(ir).float()[None,None].to(device)/255.;b=torch.from_numpy(vi).float()[None,None].to(device)/255.;a=F.pad(a,(0,(-w)%16,0,(-h)%16),mode='reflect');b=F.pad(b,(0,(-w)%16,0,(-h)%16),mode='reflect');x=m(a,b)[0,0,:h,:w];image=np.rint(x.cpu().numpy()*255).clip(0,255).astype(np.uint8)
   if not cv2.imwrite(str(o.output_dir/path.name),image):raise RuntimeError(path.name)
 print(f'Saved {len(paths)} V19 images to {o.output_dir.resolve()}')
if __name__=='__main__':main()
