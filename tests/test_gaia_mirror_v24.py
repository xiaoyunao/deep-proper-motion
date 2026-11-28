import gzip
import numpy as np
import pandas as pd
import pytest
from deep_pm import footprint_gaia_mirror_v24 as mirror


def test_original_ids_select_disjoint_bulk_files_and_nulls(tmp_path):
    ids = np.array([12,14,16],dtype="int64") << 43
    source = {k:[0.,3.,np.nan] for k in mirror.COLUMNS}
    source["source_id"]=ids
    source["ra"]=[1.,2.,3.]
    source["dec"]=[-1.,-2.,-3.]
    table=pd.DataFrame(source)
    for bounds,frame in [("000012-000014",table.iloc[:2]),("000016-000017",table.iloc[2:])]:
        with gzip.open(tmp_path/f"GaiaSource_{bounds}.csv.gz","wt") as out:
            out.write("# catalogue\n")
            frame.to_csv(out,index=False)
    positions=table.iloc[[0,2]].set_index("source_id")[["ra","dec"]]
    files=mirror.original_files(tmp_path,positions.index)
    assert len(files)==2
    rows=pd.concat([mirror.original_rows(p,positions.index) for p in files])
    result=mirror.aligned(positions,rows)
    assert pd.isna(result.iloc[1].pmra) and result.iloc[0].pmra == 0
    assert mirror.compare(table.iloc[[0,2]],result)["passes"]
    assert not mirror.compare(table.iloc[[0,2]],result.fillna(0))["passes"]
    with pytest.raises(FileNotFoundError):
        mirror.original_files(tmp_path,pd.Index([15 << 43]))
    with pytest.raises(ValueError):
        mirror.aligned(positions,rows.iloc[:1])


def test_sky_index_wrap_matches_half_open_archive_selection(tmp_path):
    from astropy.io import fits
    import healpy as hp

    folder=tmp_path/"healpix"
    folder.mkdir()
    spec={"ra":[0.,1.],"dec":[0.,1.]}
    x=np.array([359.95,0.10,1.06,0.20])
    y=np.array([0.50,0.50,0.50,1.06])
    ids=np.array([40,41,42,43])
    pixel=hp.ang2pix(32,x,y,lonlat=True,nest=True)
    cover=hp.query_disc(32,hp.ang2vec(.5,.5,lonlat=True),np.deg2rad(np.hypot(.56,.56)+.002),inclusive=True,nest=True)
    for p in cover:
        inside=pixel==p
        hdu=fits.BinTableHDU.from_columns([fits.Column(name="SOURCE_ID",format="K",array=ids[inside]),
             fits.Column(name="RA",format="D",array=x[inside]),fits.Column(name="DEC",format="D",array=y[inside])])
        hdu.header.update(HPXNSIDE=32,HPXNEST=True)
        hdu.writeto(folder/f"healpix-{p:05d}.fits")
    ix,files=mirror.sky_index(tmp_path,spec,.06)
    assert list(ix.index)==[40,41] and len(files)==len(cover)
