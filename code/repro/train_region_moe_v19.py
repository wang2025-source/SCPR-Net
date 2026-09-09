#!/usr/bin/env python3
"""End-to-end hard-target training for semantic Region-MoE V19."""
from __future__ import annotations
import argparse,csv,math,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler,autocast
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nets.Ufuser_region_moe_v19 import UfuserRegionMoEV19
from nets.fusion_losses_v19 import WorstObjectiveFirst,metric_target_losses
from nets.fusion_losses_v14 import semantic_region_losses
from nets.fusion_losses_v8 import VISUAL_NAMES,normalized_pareto_guard,visual_terms
from repro.infer_v10_region_detail_v12 import build as build_v12
from repro.msrs_v2_data import MSRSAlignedV2,aligned_names
from repro.train_pareto_assm_v8 import METRICS,validate

FORMAL_V12=np.array((6.750335,44.628377,12.396784,4.028659,1.655566,1.035681))
FORMAL_GOAL=np.array((6.90,46.0,13.0,4.20,1.68,1.07))
GOAL_RATIOS=FORMAL_GOAL/FORMAL_V12

def args():
 p=argparse.ArgumentParser();p.add_argument('--init-v12',type=Path,default=ROOT/'model/EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt');p.add_argument('--resume',type=Path);p.add_argument('--reset-optimizer',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'model/EMMA_REGION_MOE_V19.ckpt')
 p.add_argument('--reference-v19',type=Path,help='Frozen V19 baseline for safe fine-tuning; defaults to --resume')
 p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--samples-per-epoch',type=int,default=10000);p.add_argument('--crop-size',type=int,default=128);p.add_argument('--workers',type=int,default=8);p.add_argument('--metric-val-images',type=int,default=48)
 p.add_argument('--classes',type=int,default=9);p.add_argument('--channels',type=int,default=48);p.add_argument('--region',type=int,default=8);p.add_argument('--max-refine',type=float,default=.10)
 p.add_argument('--lr-moe',type=float,default=2e-4);p.add_argument('--lr-head',type=float,default=5e-5);p.add_argument('--lr-new',type=float,default=3e-5);p.add_argument('--lr-decoder',type=float,default=1e-5);p.add_argument('--lr-encoder',type=float,default=5e-6);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--warmup-epochs',type=int,default=1)
 p.add_argument('--lambda-visual',type=float,default=.35);p.add_argument('--lambda-metric',type=float,default=2.0);p.add_argument('--lambda-pareto',type=float,default=1.0);p.add_argument('--lambda-semantic',type=float,default=.04);p.add_argument('--lambda-region',type=float,default=.25);p.add_argument('--lambda-route',type=float,default=.02)
 p.add_argument('--entropy-gain',type=float,default=1.037);p.add_argument('--sd-gain',type=float,default=1.031);p.add_argument('--detail-gain',type=float,default=1.049);p.add_argument('--correlation-gain',type=float,default=1.015);p.add_argument('--information-gain',type=float,default=1.033)
 p.add_argument('--save-every',type=int,default=5);p.add_argument('--log-interval',type=int,default=100);p.add_argument('--seed',type=int,default=3407);p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');return p.parse_args()

def schedule(step,warm,total):
 if step<warm:return max((step+1)/max(1,warm),1e-3)
 x=(step-warm)/max(1,total-warm);return .1+.9*.5*(1+math.cos(math.pi*x))

def build(path,device,o):
 ck=torch.load(path,map_location='cpu');v12=build_v12(ck,device)
 return UfuserRegionMoEV19(v12,o.classes,o.channels,o.region,o.max_refine).to(device),ck

def groups(model,o):
 bins={'moe':[],'head':[],'new':[],'decoder':[],'encoder':[]}
 for n,p in model.named_parameters():
  if n.startswith('moe.'):key='moe'
  elif n.startswith('backbone.head.'):key='head'
  elif not n.startswith('backbone.v10.emma.'):key='new'
  elif n.startswith(('backbone.v10.emma.de_','backbone.v10.emma.up','backbone.v10.emma.last','backbone.v10.emma.f_')):key='decoder'
  else:key='encoder'
  bins[key].append(p)
 rates={'moe':o.lr_moe,'head':o.lr_head,'new':o.lr_new,'decoder':o.lr_decoder,'encoder':o.lr_encoder}
 return [{'params':bins[k],'lr':rates[k],'name':k} for k in bins]

def route_loss(out):
 loss=out['fused'].new_zeros(())
 for key in ('low_weights','coarse_weights','fine_weights'):
  w=out[key];usage=w.mean((0,2,3));loss=loss+F.relu(.08-usage[0])+F.relu(.08-usage[1])
 return loss/3

def save(path,epoch,model,opt,sch,metrics,o,ema):
 path.parent.mkdir(parents=True,exist_ok=True);torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'metrics':metrics,'goal_ratios':dict(zip(METRICS,GOAL_RATIOS.tolist())),'objective_ema':None if ema.ema is None else ema.ema.cpu(),'config':vars(o)},path)

def main():
 o=args();random.seed(o.seed);np.random.seed(o.seed);torch.manual_seed(o.seed);torch.backends.cudnn.benchmark=True
 if not o.init_v12.exists():raise FileNotFoundError(o.init_v12)
 device=torch.device(o.device);model,init_ck=build(o.init_v12,device,o)
 cfg=init_ck['config'];v10=torch.load(cfg['reference_v10'],map_location='cpu');bc=v10['config'];root=Path(bc['data_root']);split=torch.load(bc['split_checkpoint'],map_location='cpu');all_val=list(split['val_names']);valset=set(all_val);train=[n for n in aligned_names(root,'train') if n not in valset]
 if 0<o.metric_val_images<len(all_val):ids=np.linspace(0,len(all_val)-1,o.metric_val_images).round().astype(int);val=[all_val[i] for i in dict.fromkeys(ids)]
 else:val=all_val
 tl=DataLoader(MSRSAlignedV2(root,train,crop_size=o.crop_size,augment=True,samples_per_epoch=o.samples_per_epoch),batch_size=o.batch_size,shuffle=True,num_workers=o.workers,pin_memory=True,drop_last=True,persistent_workers=o.workers>0)
 vl=DataLoader(MSRSAlignedV2(root,val,crop_size=None,augment=False),batch_size=1,shuffle=False,num_workers=max(1,o.workers//2),pin_memory=True)
 class_weights=split['class_weights'].to(device);pg=groups(model,o);optimizer=torch.optim.AdamW(pg,weight_decay=o.weight_decay);total=o.epochs*len(tl);warm=o.warmup_epochs*len(tl);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[lambda s:schedule(s,warm,total)]*len(pg));scaler=GradScaler(enabled=o.amp and device.type=='cuda',init_scale=64.,growth_interval=1000)
 dynamic=WorstObjectiveFirst(('entropy','detail_metric','correlation_metric','information'));start=1
 if o.resume:
  ck=torch.load(o.resume,map_location=device);model.load_state_dict(ck['model'],strict=True);start=int(ck['epoch'])+1
  if not o.reset_optimizer:
   optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
   if ck.get('objective_ema') is not None:dynamic.ema=ck['objective_ema'].to(device)
 # Safe continuation compares against an immutable V19, never the older V12.
 if o.resume:
  reference_path=o.reference_v19 or o.resume
  if not reference_path.exists():raise FileNotFoundError(reference_path)
  reference_ck=torch.load(reference_path,map_location=device)
  reference,_=build(o.init_v12,device,o);reference.load_state_dict(reference_ck['model'],strict=True)
 else:
  reference_path=None;reference_ck=None;reference=build_v12(init_ck,device)
 reference.eval()
 for p in reference.parameters():p.requires_grad_(False)
 print(f'V19 params={sum(p.numel() for p in model.parameters()):,} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} train/val={len(train)}/{len(val)} batches={len(tl)}',flush=True);print('hard relative gates '+'/'.join(f'{x:.4f}' for x in GOAL_RATIOS),flush=True)
 names=('total','visual','metric','pareto','semantic','region','route','entropy','detail_metric','correlation_metric','information');hist=o.output.with_suffix('.csv');mode='a' if o.resume and hist.exists() else 'w';best=1.1 if o.resume else -1.;best_path=o.output.with_name(o.output.stem+'_best_progress'+o.output.suffix);hard_path=o.output.with_name(o.output.stem+'_best_hard'+o.output.suffix)
 # Candidate zero is original V19; a regressed epoch can never replace it.
 if o.resume:
  baseline_metrics=validate(reference,reference,vl,device)
  save(best_path,int(reference_ck.get('epoch',start-1)),reference,optimizer,scheduler,baseline_metrics,o,dynamic)
  print(f'safe reference={reference_path} candidate-zero saved to {best_path}',flush=True)
 with hist.open(mode,newline='') as f:
  w=csv.writer(f)
  if mode=='w':w.writerow(('epoch',*names,'progress','hard',*METRICS,*(x+'_r' for x in METRICS),*(g['name'] for g in pg)))
  for epoch in range(start,o.epochs+1):
   model.train();sums={k:0. for k in names}
   for bi,(ir,vi,labels,_) in enumerate(tl,1):
    ir=ir.to(device,non_blocking=True);vi=vi.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=scaler.is_enabled()):
     with torch.no_grad():ref=reference(ir,vi);bt=visual_terms(ref,ir,vi,ref)
     out=model(ir,vi,True);terms=visual_terms(out['fused'],ir,vi,ref);visual=.35*terms['intensity']+1.0*terms['gradient']+.45*terms['laplacian']+.30*terms['contrast']+.35*terms['ssim']+.8*terms['wavelet'];pareto,_=normalized_pareto_guard(terms,bt,0.)
     ml=metric_target_losses(out['fused'],ir,vi,ref,o.sd_gain,o.detail_gain,o.correlation_gain,o.information_gain,entropy_gain=o.entropy_gain);metric,weights=dynamic(ml);reg=semantic_region_losses(out,ir,vi,labels,class_weights);region=reg['foreground_intensity']+.25*reg['boundary_gradient']+.20*reg['background_texture'];route=route_loss(out);loss=o.lambda_visual*visual+o.lambda_metric*metric+o.lambda_pareto*pareto+o.lambda_semantic*reg['semantic']+o.lambda_region*region+o.lambda_route*route
    if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite epoch={epoch} batch={bi}')
    scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),2.);old=scaler.get_scale();scaler.step(optimizer);scaler.update();
    if scaler.get_scale()>=old:scheduler.step()
    vals={'total':loss,'visual':visual,'metric':metric,'pareto':pareto,'semantic':reg['semantic'],'region':region,'route':route,**ml}
    for k in names:sums[k]+=float(vals[k].detach())
    if o.log_interval and bi%o.log_interval==0:print(f'epoch {epoch}/{o.epochs} batch {bi}/{len(tl)} total={float(loss):.5f} metric={float(metric):.5f} weights='+('/'.join(f'{float(x):.2f}' for x in weights)),flush=True)
   means={k:v/len(tl) for k,v in sums.items()};m=validate(model,reference,vl,device);rat=np.array([m['ratio_'+x] for x in METRICS]);goal_progress=rat if o.resume else rat/GOAL_RATIOS;progress=float(goal_progress.min()+.1*np.exp(np.log(goal_progress.clip(1e-8)).mean()));hard=bool(np.all(goal_progress>=1.));row=[epoch,*(means[k] for k in names),progress,int(hard),*(m[x] for x in METRICS),*rat,*(g['lr'] for g in optimizer.param_groups)];w.writerow(row);f.flush();print(f'epoch {epoch}/{o.epochs} total={means["total"]:.5f} progress={progress:.5f} hard={hard} ratios='+('/'.join(f'{x:.4f}' for x in rat))+' gate='+('/'.join(f'{x:.4f}' for x in goal_progress)),flush=True);save(o.output,epoch,model,optimizer,scheduler,m,o,dynamic)
   pareto_safe=bool(np.all(rat>=1.0-1e-8)) if o.resume else True
   if pareto_safe and progress>best:best=progress;save(best_path,epoch,model,optimizer,scheduler,m,o,dynamic)
   if hard:save(hard_path,epoch,model,optimizer,scheduler,m,o,dynamic);print(f'HARD TARGET checkpoint: {hard_path}',flush=True)
   if o.save_every and epoch%o.save_every==0:save(o.output.parent/(o.output.stem+'_checkpoints')/f'epoch_{epoch:03d}.ckpt',epoch,model,optimizer,scheduler,m,o,dynamic)
 print(f'Finished. Best progress: {best_path}',flush=True)

if __name__=='__main__':main()
