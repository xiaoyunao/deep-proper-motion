"""Register the sole selection candidate after loss attribution, before its PMs."""
import argparse
from collections import Counter
import gzip
import os
from pathlib import Path

from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate


def run(registration,selection,output):
    plan=read(registration);counts={str(low):Counter() for low in (21,22)}
    receipts={}
    for entry in plan['selected']:
        file=selection/entry['core']/'private_selection_trace.json.gz'
        data=read(file);receipts[entry['core']]=digest(file)
        for row in data['qso']:
            low=str(int(row['magnitude']))
            if low not in counts or not row['geometric_opportunity']:continue
            counts[low]['geometry']+=1
            if row['primary_exit']!='pointlike_morphology':continue
            counts[low]['morphology_exit']+=1
            morph=[m['valid_median'] for m in row['anchor_morphology'] if m['valid_median'] is not None]
            sdss=row['sdss_morphology']
            if any(.6<=m<.8 for m in morph):
                counts[low]['morphology_between06_08']+=1
                if len(sdss) and all(m==6 for m in sdss):counts[low]['potential_independent_sdss_star_pool']+=1
    parameters=dict(nsc_valid_median_min=.6,baseline_min=.8,
        independent_evidence='Unique SDSS source track with all defined run morphology=STAR(6); source tracks collapse independent SDSS runs before competitor testing.',
        independent_radius_mas=500,competitor_margin_mas=300,
        baseline_keep_fraction_min=.99,max_common_p68_degradation=.05,max_axis_zero_increase_masyr=.5,
        max_wrong_member_increment=.01,potential_pool_recovery_fraction_min=.10,
        minimum_added_qso_for_magnitude_zero=20,
        test_year='Metadata-selected final physical NSC year per core, excluded from every target-band association before training.',
        disposition='Only this one candidate; no second threshold search. Adopt only scoped layer with independent identity, extended-source and held-year evidence. Otherwise close.',
        old_unknown_policy='Legacy selection unchanged; unknown/invalid morphology never granted additional passage.')
    report=dict(registration_sha256=digest(registration),trace_sha256=receipts,
        opportunity_pool={k:dict(v) for k,v in counts.items()},parameters=parameters,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Potential pool derived from position/quality/morphology only, without reading candidate proper motions. It is not a promised recovery yield or a stellar purity estimate.')
    output.mkdir(parents=True,exist_ok=True);check_aggregate(report)
    path=output/'v20_selection_candidate_registration_summary.json'
    if path.exists() and read(path)!=report:raise ValueError('Sole candidate pre-registration changed')
    save_json(path,report);print('CANDIDATE_REGISTERED',report['opportunity_pool'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','selection','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute resources')
    run(**vars(a))
