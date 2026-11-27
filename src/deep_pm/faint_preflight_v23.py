"""Read-only baseline binding, HACKS coverage and real SDSS frame metadata.

No source proper-motion residual is evaluated at this discovery stage.
"""
import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path,PurePosixPath
import os,re,tarfile,time
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from .blind_catalogue_quality import read
from .exposure_catalogues import inside
from .io import digest,save_json
from .public_summary import check_aggregate

class Text(HTMLParser):
    def __init__(self):super().__init__();self.words=[]
    def handle_data(self,text):self.words.append(text)

def fetch(url,root):
    dest=root/Path(url).name;root.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        for attempt in range(3):
            try:
                with requests.get(url,timeout=(15,120),stream=True) as response:
                    response.raise_for_status()
                    pending=dest.with_suffix(dest.suffix+'.pending');count=0
                    with pending.open('wb') as out:
                        for block in response.iter_content(1024*1024):out.write(block);count+=len(block)
                    if response.headers.get('Content-Length') and count!=int(response.headers['Content-Length']):raise ValueError('Incomplete archive length')
                    pending.replace(dest)
                break
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(1+attempt)
    return dest

def run(base,work):
    work.mkdir(parents=True,exist_ok=True);old=base/'v2.2';plan=read(old/'registration/private_full_coordinate_plan.json')
    freeze=read(old/'v22_final_stage_binding_summary.json')
    assert freeze['new_retained_solutions']==15858226 and freeze['registered_cores_checked']==2875
    rows=[];cache=work/'cache';urls=['https://archive.stsci.edu/prepds/hugs/','https://archive.stsci.edu/hlsp/hacks']
    html=fetch(urls[0],cache/'hugs'/'index.html').read_text() if False else None
    response=requests.get(urls[0],timeout=(15,90));response.raise_for_status();(work/'private_hugs_page.html').write_text(response.text)
    parser=Text();parser.feed(response.text);text=re.sub(r'\s+',' ',' '.join(parser.words))
    for match in re.finditer(r'NGC\s+(\d{4})\s+([0-9]+\.[0-9]+)\s+([+\-]?[0-9]+\.[0-9]+)',text):
        name='ngc'+match[1];ra,dec=map(float,match.groups()[1:]);cores=[e['core'] for e in plan['cores'] if bool(inside(np.array([ra]),np.array([dec]),e['spec'])[0])]
        if cores:rows.append(dict(field=name,center_deg=[ra,dec],cores=cores))
    # Candidate choice uses coverage metadata only. Additional catalogues are
    # selected before any ground/HST motion residual or stellar output is read.
    alternatives=sorted([r for r in rows if r['field']!='ngc5466'],key=lambda r:r['field'])[:2]
    selected=[dict(field='ngc5466',center_deg=[211.363708,28.534444],cores=[e['core'] for e in plan['cores'] if bool(inside(np.array([211.363708]),np.array([28.534444]),e['spec'])[0])])]+alternatives
    save_json(work/'private_HST_geometric_registration.json',dict(candidates=rows,selected=selected,rule='NGC5466 plus at most two lexicographically first HACKS/HUGS centres inside frozen v22 actual core footprints; geometry only, no motion or residual selection.',plan_sha256=digest(old/'registration/private_full_coordinate_plan.json')))
    hst=[]
    for field in selected:
        name=field['field'];url=f'https://archive.stsci.edu/hlsps/hacks/bundles/hlsp_hacks_hst_multi_{name}_multi_v1.0_phot-pm-catalog-bundle.tar.gz'
        file=fetch(url,cache/'HACKS');folder=work/'HACKS'/name;folder.mkdir(parents=True,exist_ok=True);schema=[]
        with tarfile.open(file,'r:gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(('.fits','.txt')):continue
                safe=PurePosixPath(member.name)
                if safe.is_absolute() or '..' in safe.parts:raise ValueError('Unsafe archive member')
                dest=folder/safe.name
                if not dest.exists():dest.write_bytes(archive.extractfile(member).read())
                if dest.suffix=='.fits':
                    with fits.open(dest,memmap=False) as h:
                        for index,t in enumerate(h):
                            if isinstance(t,(fits.BinTableHDU,fits.TableHDU)):
                                schema.append(dict(file=dest.name,hdu=index,records=len(t.data),columns={c.name:dict(format=c.format,unit=c.unit) for c in t.columns},header={k:str(t.header[k]) for k in t.header if k not in ('HISTORY','COMMENT','') and not k.startswith(('TTYPE','TFORM','TUNIT','TDIM'))}))
        hst.append(dict(field=name,url=url,archive_sha256=digest(file),files=schema))
        print('HST_SCHEMA',name,[(s['file'],s['records']) for s in schema],flush=True)
    save_json(work/'private_HST_schema.json',hst)
    receipt=read(old/'ngc-early/private_file_receipts.json');forty=read(old/'ngc-early/private_frozen_forty.json');frames=[]
    for filename,sha in receipt['frames'].items():
        file=Path(filename)
        if digest(file)!=sha:raise ValueError('Changed frozen SDSS frame')
        with fits.open(file,memmap=False) as h:
            frames.append(dict(file=filename,sha256=sha,hdus=[dict(hdu=i,shape=list(t.data.shape) if t.data is not None else None,columns=t.columns.names if hasattr(t,'columns') else None) for i,t in enumerate(h)],header=dict(h[0].header)))
    save_json(work/'private_SDSS_frame_schema.json',frames)
    reader=old/'ngc-mask-reader/readAtlasImages-v5_4_11';executables=[str(p) for p in reader.glob('**/read_PSF') if p.is_file()]
    out=dict(baseline_science='0320f59b972755454252a383ba1007c665c928e8',plan_sha256=digest(old/'registration/private_full_coordinate_plan.json'),final_binding_sha256=digest(old/'v22_final_stage_binding_summary.json'),
        all_v22_solutions_frozen=15858226,HST_geometry_candidates=[r['field'] for r in rows],selected_fields=[r['field'] for r in selected],HST_catalogue_tables=[dict(field=r['field'],tables=[dict(file=t['file'],n=t['records'],schema=t['columns']) for t in r['files']]) for r in hst],
        frozen_forty=len(forty['targets']),frozen_frames=len(frames),official_read_PSF_available=bool(executables),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Schema/coverage only, before external PM residuals. Frozen outer-ring forty/415 are separate from HST central-field parent. Metadata candidates not assumed usable; actual HST quality/crowding and overlap gate precedes motion evaluation.')
    check_aggregate(out);save_json(work/'v23_preflight_summary.json',out)
    print('V23_PREFLIGHT_COMPLETE',len(selected),len(frames),bool(executables),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.work)
