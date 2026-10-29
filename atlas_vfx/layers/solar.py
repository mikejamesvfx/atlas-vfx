"""Where the sun is: NOAA's compact solar-position algorithm, pure python.

Generic astronomy (Meeus via the NOAA solar calculator spreadsheet), the same algorithm family as Atlas Scene's
`environment.sun.derive_sun`, which it matches to well under 0.01 deg. It exists here because derive_sun refuses
elevations below -4.2 deg (the Prague sky's domain) and a relight still needs the direction of a sun under the
horizon. Angles are GEOMETRIC: no atmospheric refraction (Prague and Atlas Scene take the geometric sun too).
Azimuth is degrees clockwise from true north (0 N, 90 E, 180 S, 270 W), elevation degrees above the horizon.

Ported from the 2026-10-28 relight scripts (day24_sky.py `sun_at`); no ComfyUI imports.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


def parse_utc(text: str) -> datetime:
    """An ISO 8601 time that names its offset ('2025-12-25T07:39:25Z' or '...+11:00') -> aware UTC datetime.
    A time without an offset is refused: it could be local or UTC, and an hour out moves the sun 15 deg."""
    s = str(text or "").strip()
    if not s:
        raise ValueError("capture time is empty; give an ISO time with its offset, e.g. 2025-12-25T07:39:25Z")
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"capture time {s!r} is not ISO 8601 (e.g. 2025-12-25T07:39:25Z)") from e
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"capture time {s!r} has no offset; add Z for UTC or +hh:mm, a bare time is ambiguous")
    return t.astimezone(timezone.utc)


def shifted(t: datetime, hours: float) -> datetime:
    return t + timedelta(hours=float(hours))


def sun_at(latitude_deg: float, longitude_deg: float, when: datetime) -> tuple[float, float]:
    """(azimuth_deg, elevation_deg) of the sun's centre seen from latitude/longitude (deg, east positive) at `when`
    (an aware datetime). Valid for any elevation, night included."""
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("sun_at needs an aware datetime (UTC offset known)")
    lat = float(latitude_deg)
    lon = float(longitude_deg)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"latitude {lat} / longitude {lon} out of range")
    t = when.astimezone(timezone.utc)
    jd = t.timestamp() / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + math.sin(3 * Mr) * 0.000289)
    om = 125.04 - 1934.136 * T
    lam = L0 + C - 0.00569 - 0.00478 * math.sin(math.radians(om))
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(om))
    dec = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam)))
    y = math.tan(math.radians(eps) / 2.0) ** 2
    L0r = math.radians(L0)
    eqt = 4.0 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                             - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))
    minutes = t.hour * 60.0 + t.minute + (t.second + t.microsecond * 1e-6) / 60.0
    tst = (minutes + eqt + 4.0 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    latr = math.radians(lat)
    cz = math.sin(latr) * math.sin(dec) + math.cos(latr) * math.cos(dec) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cz)))
    el = 90.0 - math.degrees(zen)
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(latr) - math.tan(dec) * math.cos(latr)))
          + 180.0) % 360.0
    return az, el
