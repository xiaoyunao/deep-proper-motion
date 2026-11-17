"""Count prospectively usable spectroscopic controls and physical scan overlaps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import project
from .exposure_catalogues import inside
from .exposure_preflight import sample, subboxes
from .footprints import sdss_visits


def inspect(geometry, spectra, window):
    box = geometry["box"]
    tiles = list(subboxes(box))
    center = tuple(np.mean(box[key]) for key in ("ra", "dec"))
    sky = sample(tiles, 400, 9137)
    xy = project(*sky.T, center)
    runs = list(sdss_visits(window, center, 7.3, rerun=301, vectorized=True))
    masks = np.array([visit.coverage(xy, sky).reshape(len(tiles), 400).any(axis=1)
                      for visit in runs], dtype=bool)
    output = []
    for i, (tile, stat) in enumerate(zip(tiles, geometry["tiles"])):
        qso = spectra.loc[inside(spectra.ra, spectra.dec, tile), "reference_r"]
        output.append(dict(ra_bounds_deg=tile["ra"], dec_bounds_deg=tile["dec"],
            timed_fraction=stat["joint_time"], repeat_fraction=stat["long_sdss_repeat"],
            sdss_visits=stat["sdss_visits"], spectral_total=len(qso),
            spectral_21_23=int(qso.between(21, 23, inclusive="left").sum()),
            spectral_22_23=int(qso.between(22, 23, inclusive="left").sum()),
            unique_scan_coverage=int(masks[:, i].sum()),
            nsc_estimated_measures=stat["approximate_nsc_measurements"]))
    sharing = (masks.astype(int).T @ masks.astype(int)).tolist()
    return dict(tiles=output, shared_scan_counts=sharing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("geometry", "spectra", "window"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    geometry = json.loads(args.geometry.read_text())
    spectra = pd.read_csv(args.spectra, usecols=["ra", "dec", "reference_r"])
    print(json.dumps(inspect(geometry, spectra, args.window), allow_nan=False))


if __name__ == "__main__":
    main()
