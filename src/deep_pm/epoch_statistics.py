"""Independent scans, nights and SDSS sky coverage for the v0.6 controls."""
from collections import Counter, defaultdict

from astropy.time import Time
import numpy as np

from .astrometry import MAS_PER_RADIAN, unproject
from .dark_precision import bin_name
from .footprints import sdss_visits
from .validation import ratio


SCAN_BINS = ("0", "1", "2", "3_to_5", "6_to_10", "gt_10")


def scan_bin(count):
    return SCAN_BINS[np.searchsorted([0, 1, 2, 5, 10], count)]


def observation_groups(data):
    # All current sites involved are in the Americas. Noon UTC gives a local
    # night boundary. SDSS run, not field or rerun, is the physical scan.
    utc = Time(data.observation_time.to_numpy(), format="mjd", scale="tt").utc.mjd
    night = np.floor(utc - .5).astype(int)
    return np.where(data.survey.eq("SDSS"), data.exposure_id, np.char.add("night:", night.astype(str))), night


def source_coverage(data, xy, tracks, nsc, r_keep, night, supported, spec, settings):
    shape, mag = data.morphology.to_numpy(), data.magnitude.to_numpy()
    time = data.observation_time.to_numpy()/365.25
    runs = data.exposure_id.to_numpy()
    limit = np.tan(np.deg2rad(spec["radius"]))*MAS_PER_RADIAN
    items = defaultdict(list)
    for index in tracks:
        late = index[nsc[index] & r_keep[index]]
        if not len(late) or np.median(shape[late]) < .8 or np.linalg.norm(np.median(xy[late], axis=0)) > limit:
            continue
        early = index[~nsc[index] & r_keep[index]]
        n_run = len(set(runs[early]))
        n_night = len(set(night[early]))
        n_cal = len(set(runs[early[supported[early]]]))
        span = float(np.ptp(time[early])) if len(early) else 0
        ns = float(np.ptp(time[late]))
        values = (n_run, n_night, len(early), span, n_cal)
        key = bin_name(np.median(mag[late]), settings["reference_r_edges"])
        items[key].append(values)
        items["all"].append(values)
        items["nsc_short" if ns < settings["short_nsc_baseline_years"] else "nsc_long"].append(values)
    result = {}
    for key, values in items.items():
        a = np.array(values)
        histogram = Counter(scan_bin(v) for v in a[:, 0])
        result[key] = dict(n=len(a), scans={k: ratio(histogram[k], len(a)) for k in SCAN_BINS},
                           repeated=ratio(np.count_nonzero(a[:, 0] >= 2), len(a)),
                           repeated_given_sdss=ratio(np.count_nonzero(a[:, 0] >= 2), np.count_nonzero(a[:, 0])),
                           repeated_calibrated=ratio(np.count_nonzero(a[:, 4] >= 2), len(a)),
                           nights_p16_50_84=np.quantile(a[:, 1], [.16,.5,.84]),
                           detections_p16_50_84=np.quantile(a[:, 2], [.16,.5,.84]),
                           repeated_span_years_p16_50_84=np.quantile(a[a[:, 0]>=2, 3], [.16,.5,.84])
                           if (a[:,0]>=2).any() else None)
    return result


def footprint(path, spec, samples, seed):
    center = spec["ra"], spec["dec"]
    rng = np.random.default_rng(seed)
    angle, uniform = rng.uniform(0,2*np.pi,samples), rng.random(samples)
    cap = 1-np.cos(np.deg2rad(spec["radius"]))
    radius = np.arccos(1-uniform*cap)
    xy = np.column_stack([np.cos(angle),np.sin(angle)])*np.tan(radius[:,None])*MAS_PER_RADIAN
    sky = unproject(xy,center)
    number = np.zeros(samples,dtype=int)
    scans = list(sdss_visits(path,center,spec["radius"],rerun=301,vectorized=True))
    for visit in scans:
        number += visit.coverage(xy,sky)
    counts = Counter(map(scan_bin, number))
    area = 2*np.pi*cap*(180/np.pi)**2
    return dict(core_area_sqdeg=area, sampled_positions=samples, intersecting_run_candidates=len(scans),
                covered=ratio(np.count_nonzero(number),samples),
                repeated=ratio(np.count_nonzero(number>=2),samples),
                repeated_given_covered=ratio(np.count_nonzero(number>=2),np.count_nonzero(number)),
                scan_area_sqdeg={k:area*counts[k]/samples for k in SCAN_BINS})
