#!/usr/bin/env python3
"""Transfer only named public v2.3 aggregates and four figures; never raw data."""
import argparse,hashlib,io,json,re,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from deep_pm.public_summary import check_aggregate

PUBLIC=[
 'preflight/v23_preflight_summary.json',
 'assets/v23_field_assets_summary.json',
 'hst/v23_HST_frozen_overlap_summary.json',
 'hst/v23_HST_validation_summary.json',
 'hst/v23_HST_opportunity_summary.json',
 'diagnostics-flag-audit/v23_HST_quality_bottleneck_summary.json',
 'diagnostics/v23_spatial_PSF_validation_summary.json',
 'budget/v23_frozen_error_budget_summary.json',
 'pixelprep/v23_frame_registration_summary.json',
 'pixels/v23_late_member_summary.json',
 'pixels/v23_pixel_validation_summary.json',
 'v23_key_science_summary.json',
 'v23_delivery_summary.json',
 'v23_resource_final_summary.json',
]+['figures/'+n for n in ['v23_HST_summary.png','v23_error_budget_summary.png','v23_pixel_validation_summary.png','v23_trajectories_summary.png']]


def run(host,remote_root,destination):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',host) or not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote_root):raise ValueError('Invalid host/root')
    command='tar -C '+shlex.quote(remote_root)+' -cf - -- '+' '.join(shlex.quote(p) for p in PUBLIC)
    archive=subprocess.run(['ssh','-T','-o','LogLevel=ERROR','-o','BatchMode=yes',host,command],check=True,capture_output=True).stdout
    if len(archive)>32*1024**2:raise ValueError('Unexpected public archive size')
    contents={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
        for member in tar:
            if member.name not in PUBLIC or member.name in contents or not member.isfile() or member.size>8*1024**2:raise ValueError('Not an allowed public artifact')
            value=tar.extractfile(member).read()
            if member.name.endswith('.json'):
                def reject(x):raise ValueError('Nonfinite JSON')
                parsed=json.loads(value,parse_constant=reject)
                if member.name=='v23_resource_final_summary.json':
                    for group in ['parents','steps']:
                        for entry in parsed[group].values():entry['state']=entry['state'].split(' by ')[0]
                    parsed['public_redaction']='Cancellation actor numeric UID removed; original sacct receipt remains on NERSC. Job states and all resource values unchanged.'
                    value=(json.dumps(parsed,indent=2,allow_nan=False)+'\n').encode()
                check_aggregate(parsed)
                if re.search(rb'/pscratch/|/global/homes/|/Users/|BEGIN [A-Z ]*PRIVATE KEY',value):raise ValueError('Private path/material in public artifact')
            elif not value.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Expected PNG')
            contents[member.name]=value
    if set(contents)!=set(PUBLIC):raise ValueError('Missing named public artifact')
    receipt={}
    for source,value in contents.items():
        target=destination/('figures' if source.endswith('.png') else 'results/summary')/Path(source).name;target.parent.mkdir(parents=True,exist_ok=True);draft=target.with_suffix(target.suffix+'.part');draft.write_bytes(value);draft.replace(target);receipt[str(target.relative_to(destination))]=hashlib.sha256(value).hexdigest()
    target=destination/'results/summary/v23_public_sync_summary.json';target.write_text(json.dumps(dict(files_sha256=receipt,source_scope='Explicit18 public files only; no directory synchronization, per-source/validation/pixel arrays or authentication material.'),indent=2)+'\n')
    print('Validated public artifacts:',len(contents))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();run(a.host,a.remote_root,a.destination)
