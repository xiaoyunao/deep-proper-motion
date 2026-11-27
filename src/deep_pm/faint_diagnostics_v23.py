"""Audit external quality bottlenecks and independently verify spatial PSFs."""
import argparse,os,subprocess,shutil
from pathlib import Path
import numpy as np
from astropy.io import fits
from .blind_catalogue_quality import read
from .early_pixel_model_v23 import SpatialPSF
from .io import digest,save_json
from .public_summary import check_aggregate


def run(base,root,quality_only=False):
    out=root/('diagnostics-flag-audit' if quality_only else 'diagnostics');out.mkdir(exist_ok=True);result=[]
    for field in read(root/'preflight/private_HST_geometric_registration.json')['selected']:
        name=field['field'];file=next((root/'preflight/HACKS'/name).glob('*_pm-catalog.fits'))
        with fits.open(file,memmap=False) as h:p={c:np.array(h[1].data[c]) for c in h[1].columns.names}
        conditions={'positive_error_lt0p5':(p['EPMRAC']>0)&(p['EPMRAC']<=.5)&(p['EPMDECC']>0)&(p['EPMDECC']<=.5),'chi2_le1p5':(p['CHI2RX']>=0)&(p['CHI2RY']>=0)&(p['CHI2RX']<=1.5)&(p['CHI2RY']<=1.5),'NU_ge5':p['NU']>=5,'fraction_ge0p8':(p['NF']>0)&(p['NU']>=.8*p['NF']),'span_ge5':p['TIME']>=5,'corrected_flag1':p['CORRFLAG']==1}
        cumulative=np.ones(len(p['ID']),bool);funnel={}
        for key,value in conditions.items():cumulative&=value;funnel[key]=int(cumulative.sum())
        result.append(dict(field=name,total=len(cumulative),individual_support={k:int(v.sum()) for k,v in conditions.items()},ordered_funnel=funnel,correction_flag_counts={str(k):int(np.sum(p['CORRFLAG']==k)) for k in np.unique(p['CORRFLAG'])},corrected_minus_raw_by_flag={str(k):dict(n=int(np.sum(p['CORRFLAG']==k)),exactly_unchanged=int(np.sum((p['CORRFLAG']==k)&(p['PMRAC']==p['PMRAR'])&(p['PMDECC']==p['PMDECR']))),absolute_position_correction_p16_p50_p84=np.quantile(np.hypot(p['PMRAC'][p['CORRFLAG']==k]-p['PMRAR'][p['CORRFLAG']==k],p['PMDECC'][p['CORRFLAG']==k]-p['PMDECR'][p['CORRFLAG']==k]),[.16,.5,.84])) for k in np.unique(p['CORRFLAG'])},
            PM_error_E_N_p16_p50_p84=np.quantile(np.stack([p['EPMRAC'],p['EPMDECC']],axis=1),[.16,.5,.84],axis=0),reduced_chi2_E_N_p16_p50_p84=np.quantile(np.stack([p['CHI2RX'],p['CHI2RY']],axis=1),[.16,.5,.84],axis=0),used_exposure_p16_p50_p84=np.quantile(p['NU'],[.16,.5,.84]),time_span_p16_p50_p84=np.quantile(p['TIME'],[.16,.5,.84])))
    summary=dict(fields=result,slurm_job=os.environ['SLURM_JOB_ID'],scope='Independent input-quality diagnostics, no ground proper-motion residual or outcome-driven gate change. Counts include corrected-flag and temporal-support bottlenecks; historical field selection and current main output remain fixed.')
    check_aggregate(summary);save_json(out/'v23_HST_quality_bottleneck_summary.json',summary)
    if quality_only:return
    source=base/'v2.2/ngc-mask-reader/readAtlasImages-v5_4_11';build=out/'official_psf_reader';build.mkdir(exist_ok=True)
    # Only change output datatype in a new copy; original source/build untouched.
    code=(source/'main_PSF.c').read_text();modified=code.replace('regType = TYPE_U16;','regType = TYPE_FL32;')
    if modified==code:raise ValueError('Official output datatype declaration changed')
    (build/'main_PSF_float.c').write_text(modified);binary=build/'read_PSF_float'
    subprocess.run(['cc','-O2','-fcommon','-DSDSS_LITTLE_ENDIAN','-DCHECK_LEAKS','-DSTAND_ALONE','-I'+str(source),str(build/'main_PSF_float.c'),str(source/'libatlas.a'),'-lm','-o',str(binary)],check=True,capture_output=True)
    checks=[]
    for a in read(root/'assets/private_field_assets.json'):
        file=Path(a['psField']);psf=SpatialPSF(file)
        for row,col in [(350.,700.),(1100.,1500.)]:
            dest=out/f"private_psf_{a['run']}_{a['camcol']}_{a['field']}_{int(row)}.fits"
            subprocess.run([str(binary),str(file),'3',str(row),str(col),str(dest)],check=True,capture_output=True)
            c=fits.getdata(dest).astype(float);p=psf.image(col,row);delta=c-c.mean();delta2=p-p.mean();scale=float(np.sum(delta*delta2)/np.sum(delta2**2));intercept=float(c.mean()-scale*p.mean());resid=c-(scale*p+intercept)
            relative=float(np.max(abs(resid))/max(np.max(abs(c-intercept)),1e-30))
            if relative>2e-5:raise ValueError('Python spatial PSF differs from independent official C kernel')
            checks.append(dict(run=a['run'],camcol=a['camcol'],field=a['field'],test_row=row,test_column=col,relative_maximum_shape_difference=relative,C_output_constant_intercept=intercept))
    summary=dict(physical_fields=13,independent_spatial_reconstructions=len(checks),checks=checks,official_source_sha256=digest(source/'main_PSF.c'),output_dtype_modified_source_sha256=digest(build/'main_PSF_float.c'),binary_sha256=digest(binary),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Independent official read_PSF implementation, changed only output datatype U16 to FL32 in separate new file; C++/C reconstruction compared against all-eigenimage Python FITS polynomial at two asymmetric actual locations per field. Constant soft bias and normalization profiled for shape verification; original v22 source/build unchanged.')
    check_aggregate(summary);save_json(out/'v23_spatial_PSF_validation_summary.json',summary);print('V23_INPUT_DIAGNOSTICS_AND_PSF_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);p.add_argument('--quality-only',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.root,a.quality_only)
