"""Deidentify aggregate observation labels and name resampling scopes correctly."""
import argparse
from hashlib import sha256
from pathlib import Path
from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate


def export(source,output,kind):
    original=read(source);repeated=kind=='geometry'
    def transform(value):
        if isinstance(value,dict):
            mapping={'night_geometry':'utc_calendar_date_geometry','night':'utc_calendar_date' if repeated else 'observing_night_UTC18',
                'median_object_interval95':'measurement_resampling_sensitivity95' if repeated else 'pair_resampling_sensitivity95'}
            return {mapping.get(k,k):transform(v) for k,v in value.items()}
        if isinstance(value,list):return [transform(v) for v in value]
        if isinstance(value,str) and value.startswith(('NSC:','SDSS:')):return 'observation_'+sha256(('v22-public-observation:'+value).encode()).hexdigest()[:12]
        return value
    result=transform(original);result['raw_public_readout_sha256']=digest(source)
    result['export_scope']='Observational labels deidentified. Geometry date bins are UTC calendar dates; conditions use physical UTC18 observing nights. Resampling intervals are conditional measurement/pair sensitivity, not independent-source or independent-sky confidence. Private originals unchanged.'
    check_aggregate(result);save_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--kind',choices=['geometry','conditions','pairs'],required=True);a=p.parse_args();export(**vars(a))
