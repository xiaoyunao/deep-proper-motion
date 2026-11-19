"""Register auxiliary filter/time/instrument tests using inventory metadata only."""
import argparse
import json
from pathlib import Path

from .io import digest,save_json
from .public_summary import check_aggregate


def select(inventory):
    data=inventory['regions']
    def first(predicate):
        return next((name for name in sorted(data) if any(predicate(label,stats) for label,stats in data[name]['instrument_band'].items())),None)
    fields={
        'Y_single_night':first(lambda label,s:label=='NSC:c4d:Y' and s['candidate_reference_supported_count']>0 and s['candidate_nights']==1),
        'Y_multiple_nights':first(lambda label,s:label=='NSC:c4d:Y' and s['candidate_reference_supported_count']>0 and s['candidate_nights']>=3),
        'VR_multiple_years':first(lambda label,s:label=='NSC:c4d:VR' and s['candidate_reference_supported_count']>0 and len(s['utc_year_counts'])>=2),
        '90Prime_instrument':first(lambda label,s:label.startswith('NSC:ksb:') and s['candidate_reference_supported_count']>0),
        'u_support_sensitivity':'a_09_00' if 'a_09_00' in data and 'u' in data['a_09_00']['actual_nsc_bands'] else None}
    result=dict(selected=fields,
        selection_basis='First lexical core satisfying physical inventory criteria, except the already registered a_09_00 u core. No proper-motion, QSO residual or model gain read.',
        purpose='Separate one-night centroid information from additional years, test VR and 90Prime, and test u only if a research support trial retains complete reference covariance.',
        interpretation='Auxiliary coverage challenges, not a representative prevalence or never-opened independent sky test; all cores were in v1.3. No new-band result used to choose them.')
    check_aggregate(result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inventory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    x=select(json.loads(a.inventory.read_text()));x['inventory_sha256']=digest(a.inventory);save_json(a.output,x)


if __name__=='__main__':main()
