"""Independent calibration, event reconstruction, and sideband feature extraction.

No MOSKITA imports. Geometry/overscan/readout convention informed by local docs.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/analysis.json').read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calibrate(raw):
    """Weighted two-Gaussian fit to 0/1e peaks; ROI not used in fit."""
    b = np.asarray(raw, dtype=np.float64) - np.median(raw[:, CONFIG['overscan_start']:], axis=1)[:, None]
    a = b[0:512, 8:3080]
    edges = np.arange(-140, 342, 2.)
    hist, _ = np.histogram(a, edges)
    x = (edges[1:] + edges[:-1]) / 2

    def model(x, a0, mu, sigma, a1, gain, floor):
        return a0*np.exp(-0.5*((x-mu)/sigma)**2) + a1*np.exp(-0.5*((x-mu-gain)/sigma)**2) + floor

    p, cov = optimize.curve_fit(model, x, hist,
        p0=[hist.max(), 0, 36, max(hist[(x>170)&(x<270)].max(), 10), 215, 1],
        sigma=np.sqrt(np.maximum(hist, 1)), absolute_sigma=True,
        bounds=([0, -30, 15, 0, 170, 0], [np.inf, 30, 70, np.inf, 270, 1000]), maxfev=4000)
    a0, mu, sigma, a1, gain, floor = p
    noise = sigma/gain
    if not (0.08 < noise < 0.30 and a1 > 5 and np.isfinite(p).all()):
        raise ValueError(f'Calibration quality failure: {p}')
    q = ((a-mu)/gain).astype(np.float32)
    meta = dict(gain_adu_per_e=float(gain), gain_fit_se=float(np.sqrt(cov[4,4])),
                pedestal_adu=float(mu), noise_e=float(noise), one_peak_amplitude=float(a1),
                reduced_chi2=float(np.sum((hist-model(x,*p))**2/np.maximum(hist,1))/(len(x)-len(p))))
    return q, meta


def pixel_time(header):
    """Days per pixel: uniform serial clock, no assumed sensor mass."""
    seconds = (datetime.fromisoformat(header['DATEEND'].strip())-
               datetime.fromisoformat(header['DATESTART'].strip())).total_seconds()
    assert seconds > 0
    y, x = np.indices((512,3072))
    dt = seconds/(int(header['NROW'])*int(header['NCOL']))
    return (float(header['EXPOSURE']) + (y*int(header['NCOL'])+x+8)*dt)/86400, seconds


def reconstruct(q, threshold=None):
    """8-connected above-threshold pixels, summed charge, maximum-pixel anchor."""
    threshold = CONFIG['pixel_threshold_e'] if threshold is None else threshold
    labels, n = ndimage.label(q > threshold, structure=np.ones((3,3)))
    inds = np.flatnonzero(labels)
    ids = labels.ravel()[inds]
    charges = np.bincount(ids, weights=q.ravel()[inds], minlength=n+1)[1:]
    sizes = np.bincount(ids, minlength=n+1)[1:]
    # Deterministic max-charge anchor; minimum flat index resolves ties.
    maxq = np.full(n+1, -np.inf)
    np.maximum.at(maxq, ids, q.ravel()[inds])
    anchors = np.full(n+1, q.size, dtype=np.int64)
    hit = q.ravel()[inds] == maxq[ids]
    np.minimum.at(anchors, ids[hit], inds[hit])
    y,x = np.unravel_index(anchors[1:],q.shape)
    return labels, dict(charge=charges, size=sizes, y=y, x=x)


def tile_sum(a, size):
    h,w = a.shape
    return a.reshape(h//size,size,w//size,size).sum(axis=(1,3))


def sideband_maps(q, labels, events):
    """All charges 2.5..20e excluded from positive diagnostic channels.

    The method cannot see ROI components, including their individual pixels.
    No reclassification of subpixels from a 4..7e cluster as 1e events.
    """
    side = (events['charge'] >= CONFIG['sideband_e'][0]) & (events['charge'] < CONFIG['sideband_e'][1])
    low = np.zeros(q.shape, dtype=np.int32)
    np.add.at(low, (events['y'][side], events['x'][side]), 1)
    high_ids = np.r_[False, events['charge'] > CONFIG['high_charge_e']]
    high = high_ids[labels]
    negative = q < CONFIG['negative_tail_e']
    return np.stack([tile_sum(v, CONFIG['base_tile']) for v in (low,high,negative)],axis=-1)


def extract_file(args):
    path, cache = map(Path,args)
    import re
    module = re.search(r'mod(S\d+)', path.name).group(1)
    image_id = int(re.search(r'_img(\d+)',path.name).group(1))
    split = 'train' if image_id <= CONFIG['train_max_image'] else ('validation' if image_id <= CONFIG['validation_max_image'] else 'test')
    manifest = dict(filename=path.name, bytes=path.stat().st_size, sha256=sha256(path),
                    module=module, image_id=image_id, split=split, hdus=[])
    with fits.open(path, memmap=False) as hdul:
        assert len(hdul)==4
        for i,hdu in enumerate(hdul):
            assert hdu.data.shape == (515,3200) and np.isfinite(hdu.data).all()
            q, cal = calibrate(hdu.data)
            times, duration = pixel_time(hdu.header)
            labels, ev = reconstruct(q)
            features = sideband_maps(q, labels, ev)
            roi = (ev['charge']>=3.5)&(ev['charge']<7.5)
            meta = dict(module=module,image_id=image_id,hdu=i,split=split,
                shape=list(hdu.data.shape), date_start=hdu.header['DATESTART'], date_end=hdu.header['DATEEND'],
                readout_seconds=duration, nominal_exposure_seconds=float(hdu.header['EXPOSURE']),
                channel=hdu.header.get('CHID'), exposure_pixel_days=float(times.sum()),
                roi_events=int(roi.sum()), roi_mult_pixel=int(np.sum(roi & (ev['size']>1))), **cal)
            np.savez_compressed(cache/f'{module}_{image_id}_{i}.npz',
                features=features, exposure=tile_sum(times,CONFIG['base_tile']),
                charge=ev['charge'],size=ev['size'],y=ev['y'],x=ev['x'],
                pixel_hist=np.histogram(q,bins=np.linspace(-1,10,441))[0],
                metadata=json.dumps(meta))
            manifest['hdus'].append(meta)
    return manifest
