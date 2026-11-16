"""HEALPix task boundaries and cross-block conflict reconciliation."""
import healpy as hp
import numpy as np

from .associate import choose_tracks


def halo_radius(mu_limit_masyr, span_yr, astrometric_padding_mas):
    return np.deg2rad((mu_limit_masyr * span_yr + astrometric_padding_mas) / 3600000)


def home_pixel(ra, dec, nside):
    return hp.ang2pix(nside, np.asarray(ra) % 360, dec, nest=True, lonlat=True)


def source_pixels(ra, dec, nside, halo_rad):
    """Return a conservative pixel load list using a circumscribed disc."""
    theta = np.deg2rad(90 - dec)
    phi = np.deg2rad(ra % 360)
    vec = hp.ang2vec(theta, phi)
    return hp.query_disc(nside, vec, halo_rad, nest=True, inclusive=True)


def reconcile(block_candidates, max_component):
    """Resolve shared detection IDs across tiles, then assign fitted owners."""
    unique = {}
    for candidate, owner in block_candidates:
        key = tuple(sorted(candidate.indices))
        if tuple(candidate.indices) != key:
            raise ValueError("Candidate detection IDs must be sorted to reconcile tiles")
        prior = unique.get(key)
        if prior is None or candidate.score > prior[0].score:
            unique[key] = candidate, owner
    tracks, difficult, _ = choose_tracks([candidate for candidate, _ in unique.values()], max_component)
    return [(track, unique[track.indices][1]) for track in tracks], difficult
