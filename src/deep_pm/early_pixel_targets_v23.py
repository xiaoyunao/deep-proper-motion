"""Frozen outer-ring targets: late detections once and physical-run pixel cuts."""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import pandas as pd
from .astrometry import project,unproject,years,sdss_tai_to_tt
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree,angular_distance,native_covariance,projected_covariance,forward_jacobian,fit_vector_motion
from .continuous_reference import PRECISION,SPATIAL
from .__main__ import tomllib
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .allband_qso import physical_candidates
from .full_coordinate_sources import reference_context,component_map,write_rows
from .early_pixel_data_v23 import Field
from .early_pixel_model_v23 import Cut
from .io import digest,save_json
from .public_summary import check_aggregate


def prepare(base,root):
    out=root/'pixels';out.mkdir(exist_ok=True);old=base/'v2.2';forty=read(old/'ngc-early/private_frozen_forty.json')['targets'];plan={e['core']:e for e in read(old/'registration/private_full_coordinate_plan.json')['cores']};mapping,_=component_map(str(old));contexts={};config=tomllib.loads(PRECISION.read_text());prod=tomllib.loads(SPATIAL.read_text())['production'];records=[]
    for core in sorted({r['core'] for r in forty}):
        e=plan[core];data,_,t,sigma,_=quality_detections(Path(e['source']),e['center'],config,prod);groups=calibration_groups(data);physical=physical_candidates(data)&data.survey.eq('NSC').to_numpy();which=np.flatnonzero(physical)
        tree=AngularTree(project(data.ra_deg.iloc[which],data.dec_deg.iloc[which],e['center']),e['center'])
        for target in [r for r in forty if r['core']==core]:
            goal=project([target['ra']],[target['dec']],e['center'])[0];candidates=which[np.asarray(tree.query_ball_point(goal,1000),int)];indices=[];ambiguous=0
            for exposure in sorted(set(data.exposure_id.iloc[candidates])):
                ix=candidates[data.exposure_id.iloc[candidates].to_numpy()==exposure]
                if len(ix)>1:
                    values=angular_distance(project(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],e['center']),goal,e['center']);order=np.argsort(values)
                    if values[order[1]]-values[order[0]]<300:ambiguous+=1;continue
                    indices.append(int(ix[order[0]]))
                elif len(ix):indices.append(int(ix[0]))
            ix=np.array(indices,int);parts=mapping.reindex(groups[ix]).dropna().astype(int).unique();record=dict(target,qualified_nearby_NSC=len(candidates),competing_exposures=ambiguous,selected_detections_before_reference=len(ix))
            if len(parts)!=1:record.update(status='no_unique_reference_component');records.append(record);continue
            component=int(parts[0])
            if component not in contexts:contexts[component]=reference_context(str(old/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'))
            center,solution,_,_,graphsha=contexts[component];supported=[];local=(target['ra'],target['dec'])
            for i in ix:
                if groups[i] not in solution:continue
                b=np.linalg.inv(forward_jacobian([data.ra_deg.iloc[i]],[data.dec_deg.iloc[i]],center))[0];cov=b@np.cov(solution[groups[i]][1],rowvar=False)@b.T
                if np.sqrt(np.diag(cov)).max()<=12:supported.append(i)
            ix=np.array(supported,int);record['supported_NSC_detections']=len(ix)
            if len(ix)<2 or np.ptp(t[ix])<.6:record.update(status='insufficient_late_time_support');records.append(record);continue
            graphxy=project(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center);corrected=graphxy-np.array([solution[g][0] for g in groups[ix]]);sky=unproject(corrected,center);xy=project(*sky.T,local)
            a=forward_jacobian(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],local)@np.linalg.inv(forward_jacobian(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center));cp=projected_covariance(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center,native_covariance(sigma[ix]));cov=a@cp@a.transpose(0,2,1)
            model=fit_vector_motion(xy,t[ix],cov,2015.5);ref=np.stack([solution[g][1] for g in groups[ix]],axis=1);ref=np.einsum('nij,dnj->dni',a,ref)
            shape=data.morphology.iloc[ix].to_numpy(float);shape=shape[np.isfinite(shape)&(shape>=0)&(shape<=1)]
            record.update(status='conditional_late_association',reference_component=component,reference_model_sha256=graphsha,late_xy=xy,late_time=t[ix],late_covariance=cov/model.weight[:,None,None],late_robust_weight=model.weight,late_reference_draws=ref,late_only_state=model.coeff.T.ravel(),late_only_conditional_covariance=model.cov,
                late_detection_ids=data.detection_id.iloc[ix].tolist(),late_exposures=data.exposure_id.iloc[ix].tolist(),late_bands=data.band.iloc[ix].tolist(),late_instruments=data.instrument.iloc[ix].fillna('unknown').tolist(),late_morphology_median=float(np.median(shape)) if len(shape) else None,
                caveat='Position-gated external optical identity, one detection/exposure with300mas competitor margin. No NSC object ID truth, no early target centroid/PM or Gaia used. Fixed late-only robust weights; unmodelled night and colour terms remain.')
            records.append(record)
        print('V23_LATE_MEMBERS_CORE',core,flush=True)
    claims=Counter(i for r in records if r['status']=='conditional_late_association' for i in r['late_detection_ids'])
    for r in records:
        if r['status']=='conditional_late_association' and any(claims[i]>1 for i in r['late_detection_ids']):r['status']='shared_late_detection_identity_ambiguous'
    write_rows(out/'private_frozen_forty_late.json.gz',records)
    summary=dict(frozen_targets=len(records),status=dict(Counter(r['status'] for r in records)),categories={c:dict(Counter(r['status'] for r in records if r['category']==c)) for c in sorted({r['category'] for r in records})},late_input_sha256=digest(out/'private_frozen_forty_late.json.gz'),frozen_forty_sha256=digest(old/'ngc-early/private_frozen_forty.json'),slurm_job=os.environ['SLURM_JOB_ID'],scope='All40 optical identities preserved. Late actual all-band positions only once, physical flags/SNR/errors and native reference12mas; no prior true motion, no SDSS centroid likelihood. Shared late detection claims remain ambiguous, not assigned to two objects. This is conditional identity support, not stellar truth.')
    check_aggregate(summary);save_json(out/'v23_late_member_summary.json',summary)


class TargetFactory:
    def __init__(self,base,root):
        self.base=base;self.root=root;self.assets=read(root/'assets/private_field_assets.json');self.ties={p.parent.name:read(p) for p in (root/'pixelprep').glob('*/private_frame_tie.json')};self.graphs={}
        self.getfield=lru_cache(maxsize=16)(lambda key:Field(base,next(a for a in self.assets if f"{a['run']:06d}-{a['camcol']}-{a['field']:04d}"==key)))

    def make(self,row,position=None,require_tie=False):
        local=tuple([row['ra'],row['dec']] if position is None else position);component=int(row['reference_component'])
        if component not in self.graphs:self.graphs[component]=reference_context(str(self.base/'v2.2/reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'))
        center,solution,_,_,_=self.graphs[component];options={};unsupported=[]
        for key,tie in self.ties.items():
            field=self.getfield(key);pixel=field.wcs.all_world2pix([local],0)[0];edge=min(pixel[0],pixel[1],field.pixel.shape[1]-1-pixel[0],field.pixel.shape[0]-1-pixel[1]);a=tie['asset']
            if edge<12:continue
            run=a['run']
            if run not in options or edge>options[run][0]:options[run]=(edge,key)
        cuts=[];ref=[];reg=[];meta=[]
        for run,(_,key) in sorted(options.items()):
            tie=self.ties[key];field=self.getfield(key);a=tie['asset'];group=f"SDSS:{run}:{a['camcol']}"
            if require_tie and not tie['tie_pass']:unsupported.append('independent_frame_tie_failed');continue
            if group not in solution:unsupported.append('unsolved_early_reference');continue
            inverse=np.linalg.inv(forward_jacobian([local[0]],[local[1]],center))[0];r=solution[group][1]@inverse.T
            if np.sqrt(np.diag(np.cov(r,rowvar=False))).max()>12:unsupported.append('early_reference_support');continue
            zero=project(*unproject(project([local[0]],[local[1]],center)+solution[group][0],center).T,local)[0]
            conversion=np.linalg.inv(forward_jacobian([local[0]],[local[1]],tie['center_deg']))[0];delta=conversion@np.asarray(tie['constant_offset_mas']);draws=np.asarray(tie['calibration_draws'])@conversion.T
            pixel,plate=field.affine(local);stamp=field.stamp(*pixel)
            if stamp is None:unsupported.append('masked_or_edge_early_pixels');continue
            data,sigma,gx,gy=stamp;year=float(years(sdss_tai_to_tt(a['tai_r_at_row_zero_seconds']+a['tai_r_per_row_seconds']*(pixel[1]+tie['rowc_minus_image_y']))))
            cuts.append(Cut(data,sigma,gx,gy,pixel,plate,zero-delta,year,field.psf,group,int(run),r,draws));ref.append(r);reg.append(-draws);meta.append(dict(field_tag=key,run=int(run),time=year,independent_frame_tie_pass=bool(tie['tie_pass']),usable_pixels=len(data)))
        return cuts,np.stack(ref,axis=1) if ref else np.zeros((48,0,2)),np.stack(reg,axis=1) if reg else np.zeros((48,0,2)),meta,unsupported

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    prepare(a.base,a.root)
