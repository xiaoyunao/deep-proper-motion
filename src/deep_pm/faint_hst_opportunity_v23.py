"""Position-only HST opportunity audit; do not relax the frozen primary sample."""
import argparse,os
from collections import Counter
from pathlib import Path
import numpy as np
from astropy.io import fits
import duckdb
from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate


def run(base,root):
    parents=read(root/'hst/private_HST_parent.json.gz');db=duckdb.connect(str(base/'v2.2/catalogue/catalogue.duckdb'),read_only=True);out=[]
    for name in ['ngc5466','ngc5272','ngc5904']:
        part=[r for r in parents if r['field']==name];file=next((root/'preflight/HACKS'/name).glob('*_pm-catalog.fits'))
        with fits.open(file,memmap=False) as h:
            p=h[1].data;quality=np.isfinite(np.column_stack([p[k] for k in ['EPMRAC','EPMDECC','CHI2RX','CHI2RY']])).all(axis=1)&(p['EPMRAC']>0)&(p['EPMRAC']<=.5)&(p['EPMDECC']>0)&(p['EPMDECC']<=.5)&(p['CHI2RX']>=0)&(p['CHI2RX']<=1.5)&(p['CHI2RY']>=0)&(p['CHI2RY']<=1.5)&(p['NU']>=5)&(p['NF']>0)&(p['NU']>=.8*p['NF'])&(p['TIME']>=5)
            ids={int(x):i for i,x in enumerate(p['ID'])};flag=p['CORRFLAG'].copy()
        bins=Counter();matched=Counter();native=Counter();no_flag_qualified=0;no_flag_isolated=0
        for r in part:
            if not quality[ids[r['HST_ID']]] or not r['photometric_quality']:continue
            no_flag_qualified+=1;no_flag_isolated+=int(r['isolated']);m=r['F606W']
            native['other_or_missing' if m is None or not 20<=m<23 else str(int(m))]+=1
            if not r['matched_v22']:continue
            result=db.execute('SELECT magnitude FROM accepted WHERE src=?',[r['ground_src']]).fetchone()
            if result is None:raise ValueError('Frozen external ground identity missing')
            mag=result[0];bins['other' if not 20<=mag<23 else str(int(mag))]+=1;matched[str(int(flag[ids[r['HST_ID']]]))]+=1
        out.append(dict(field=name,HST_quality_except_correction_flag=no_flag_qualified,isolated_quality_except_correction_flag=no_flag_isolated,HST_F606W_parent_bins=dict(native),position_matches_ground_r_bins=dict(bins),matched_correction_flags=dict(matched)))
    db.close();summary=dict(fields=out,original_primary_freeze_sha256=digest(root/'hst/private_quality_freeze.json'),scope='Additional position/quality opportunity counts only, no ground PM residuals read and no replacement of the CORRFLAG=1 primary selection. Flag0 is not relabelled as corrected. Paper appendixB defines1 as high-frequency correction applied; corrected-minus-raw can be nonzero at0 because low-frequency corrections are distinct. HST nativeF606W is not SDSS/NSC r. Empty primary comparison is reported as no independent stellar precision measurement, never zero error.')
    check_aggregate(summary);save_json(root/'hst/v23_HST_opportunity_summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.root)
