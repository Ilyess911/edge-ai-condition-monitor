import numpy as np

from src.data.paderborn import defect_orders
from src.features.bearing import FEATURE_NAMES, BearingFeatureExtractor


def _window(fs, n, speed_rpm, defect_hz=None, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.arange(n) / fs
    vib = 0.2 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * speed_rpm / 60 * t)
    if defect_hz:
        # Impacts at the defect frequency, each ringing a 6 kHz resonance.
        ring = np.exp(-np.arange(int(0.004 * fs)) / (0.0008 * fs)) * np.sin(
            2 * np.pi * 6000 * np.arange(int(0.004 * fs)) / fs)
        for t0 in np.arange(0, n / fs, 1 / defect_hz):
            i = int(t0 * fs)
            seg = vib[i:i + len(ring)]
            seg += 2.0 * ring[: len(seg)]
    cur = np.sin(2 * np.pi * 50 * t) + 0.01 * rng.standard_normal(n)
    const = np.ones(n)
    return np.column_stack([vib, cur, speed_rpm * const, 0.7 * const, 1000 * const])


def test_envelope_feature_finds_outer_race_defect_frequency():
    fs, n, rpm = 64_000, 32_000, 1500
    ext = BearingFeatureExtractor(fs, n)
    bpfo = defect_orders()["bpfo"] * rpm / 60
    healthy = dict(zip(FEATURE_NAMES, ext(_window(fs, n, rpm))))
    faulty = dict(zip(FEATURE_NAMES, ext(_window(fs, n, rpm, defect_hz=bpfo))))
    assert faulty["env_bpfo"] > 5 * healthy["env_bpfo"]
    assert faulty["env_bpfo"] > faulty["env_bpfi"]
    assert faulty["vib_kurtosis"] > healthy["vib_kurtosis"]


def test_features_stay_finite_at_low_sampling_rate():
    fs, n = 2_000, 1_000
    f = BearingFeatureExtractor(fs, n)(_window(fs, n, 900))
    assert len(f) == len(FEATURE_NAMES) and np.isfinite(f).all()
    bands = f[4:8]
    assert abs(bands.sum() - 1.0) < 1e-9 and bands[3] == 0.0  # nothing above Nyquist


def test_damage_profile_parser_handles_multiple_damages_and_missing_extent():
    from src.data.paderborn import parse_damage_profile

    text = ("Number of damages 2   \nMode   fatigue plastic deformation  \n"
            "Component  IR OR  \nPosition of damage raceway raceway  \n"
            "Damage combination M M  \nExtent of damage 1 n/a  \n"
            "Characteristic of \ndamage single point single point  \n"
            "Damage occurrence Damage method lifetime  test lifetime  test  \n")
    p = parse_damage_profile(text)
    assert p["n_damages"] == "2" and p["component"] == "IR OR" and p["combination"] == "M M"
    assert p["extent"] == "1 n/a"
