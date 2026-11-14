"""Tests for weather/gfs_live.py (today's GFS analysis as the model's input).

They run offline. A small server on 127.0.0.1 plays NOAA's bucket and a forecast server, and the GRIB
records are written by hand in this file. One more test fetches the real analysis (about 18 MB); it is
skipped unless WX_LIVE=1.

    .venv/bin/python -m unittest tests/test_gfs_live.py -v
    WX_LIVE=1 WX_DATA=.local/bundle .venv/bin/python -m unittest tests/test_gfs_live.py -v

Written at the hackathon (Nov 13-14, 2026).
"""
import base64
import contextlib
import datetime
import http.server
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WEATHER = os.path.join(REPO, "weather")
sys.path.insert(0, WEATHER)

import gfs_live as gfs      # noqa: E402
import wx_common as wx      # noqa: E402

# Made-up statistics, close to the real ones, so the tests need no data folder.
MEAN = [0.0, 0.2, 278.6, 96800.0, 100960.0, 274.6, 0.0, 0.2, 740.0, 1.4,
        0.1, 13750.0, 6.6, 0.0, 54120.0, 253.0, 199400.0, 0.40, 0.61, 18.3]
STD = [5.5, 4.7, 21.0, 9290.0, 1330.0, 15.5, 6.0, 5.2, 1070.0, 8.1,
       6.2, 1467.0, 11.9, 9.1, 3343.0, 13.0, 5835.0, 0.25, 0.26, 16.3]
Z = [0.02 * c - 0.21 for c in range(20)]    # the z-score each made-up constant map should come out as

# The time the tests pretend it is: the 06Z cycle is the newest name, but only 00Z is "posted".
NOW = datetime.datetime(2026, 11, 14, 6, 10, tzinfo=datetime.timezone.utc)

# GRIB2 codes, written down here on their own (not copied from gfs_live): variable -> (category, number).
PARAMETERS = {"UGRD": (2, 2), "VGRD": (2, 3), "TMP": (0, 0), "PRES": (3, 0), "PRMSL": (3, 1), "HGT": (3, 5),
              "RH": (1, 1), "PWAT": (1, 3)}
# Names that look like the model's records but are not. They must never be picked.
DECOYS = ["TMP:2 mb", "UGRD:10 mb", "HGT:surface", "TMP:surface", "PRES:mean sea level"]

FINE_LATS = 90.0 - 0.25 * np.arange(721)
FINE_LONS = 0.25 * np.arange(1440)


def t2m_made_up(lat, lon):
    """A smooth made-up 2-m temperature (K). North differs from south and east from west."""
    return 250.0 + 40.0 * np.cos(np.deg2rad(lat)) + 5.0 * np.sin(np.deg2rad(lon)) + 0.1 * lat


def surface_of(level):
    """(GRIB2 type of fixed surface, its value) for a level text of NOAA's index."""
    if level.endswith(" mb"):
        return 100, int(level.split()[0]) * 100         # a pressure level, in Pa
    if level.endswith(" m above ground"):
        return 103, int(level.split()[0])
    return {"surface": 1, "mean sea level": 101, "entire atmosphere (considered as a single layer)": 200}[level], 0


def signed(value, bits):
    """GRIB stores a negative number as a sign bit plus the size."""
    return value if value >= 0 else (1 << (bits - 1)) | -value


def make_grib(key, field, south_first=False):
    """A GRIB2 record like NOAA's for the index name `key` ("variable:level"), written by hand.

    field: one number (a constant map) or a (721, 1440) array, 90 N first (packed in 16 bits).
    south_first=True writes a grid that starts at 90 S, which gfs_live must refuse.
    """
    variable, level = key.split(":", 1)
    category, number = PARAMETERS[variable]
    surface, surface_value = surface_of(level)
    points = 721 * 1440
    if np.ndim(field) == 0:
        reference, exponent, bits, data = float(field), 0, 0, b""
    else:
        field = np.asarray(field, np.float64)
        reference = np.float32(field.min())
        if float(reference) > field.min():
            reference = np.nextafter(reference, np.float32(-np.inf))
        reference = float(reference)
        exponent = int(math.ceil(math.log2((field.max() - reference) / 65535.0)))
        data = np.round((field - reference) / 2.0 ** exponent).astype(">u2").tobytes()
        bits = 16
    first_lat, last_lat, scan = (-90000000, 90000000, 64) if south_first else (90000000, -90000000, 0)
    identification = struct.pack(">IBHHBBBHBBBBBBB", 21, 1, 7, 0, 2, 1, 1, 2026, 11, 14, 0, 0, 0, 0, 0)
    grid = (struct.pack(">IBBIBBH", 72, 3, 0, points, 0, 0, 0) + struct.pack(">BBIBIBI", 6, 0, 0, 0, 0, 0, 0)
            + struct.pack(">IIII", 1440, 721, 0, 0) + struct.pack(">IIB", signed(first_lat, 32), 0, 48)
            + struct.pack(">IIIIB", signed(last_lat, 32), 359750000, 250000, 250000, scan))
    product = (struct.pack(">IBHH", 34, 4, 0, 0) + struct.pack(">BBBBBHBBI", category, number, 0, 0, 81, 0, 0, 1, 0)
               + struct.pack(">BBIBBI", surface, 0, surface_value, 255, 255, 0xFFFFFFFF))
    packing = (struct.pack(">IBIH", 21, 5, points, 0)
               + struct.pack(">fHHBB", reference, signed(exponent, 16), 0, bits, 0))
    bitmap = struct.pack(">IBB", 6, 6, 255)
    values = struct.pack(">IB", 5 + len(data), 7) + data
    body = identification + grid + product + packing + bitmap + values + b"7777"
    return b"GRIB" + struct.pack(">HBBQ", 0, 0, 2, 16 + len(body)) + body


def grib_payloads(z=Z, t2m_field=None):
    """{index name: GRIB record} for the model's 20 records.

    Each map is a constant, chosen so that channel c comes out with the z-score z[c] under MEAN and STD.
    t2m_field replaces the 2-m temperature by a real map.
    """
    payloads = {}
    for c, (key, _, _) in enumerate(gfs.CHANNELS):
        to_model_units = 9.80665 if key.startswith("HGT") else 0.01 if key.startswith("RH") else 1.0
        payloads[key] = make_grib(key, (MEAN[c] + z[c] * STD[c]) / to_model_units)
    if t2m_field is not None:
        payloads["TMP:2 m above ground"] = make_grib("TMP:2 m above ground", t2m_field)
    return payloads


def junk_payloads():
    """{index name: some bytes}, different for each record, for the tests that only download."""
    return {key: ("%s #%d " % (key, c)).encode() * (c + 5) for c, (key, _, _) in enumerate(gfs.CHANNELS)}


def build_file(cycle, payloads):
    """A made-up GFS file. Returns (the index text as NOAA writes it, the file's bytes).

    The model's records are not in the model's order and look-alike names sit in between. The file
    starts with PRMSL at offset 0, as NOAA's does. Its last record is one the model needs (it has no
    "next offset" in the index).
    """
    keys = [key for key, _, _ in gfs.CHANNELS]
    middle = [key for key in reversed(keys) if key not in (keys[0], keys[4])]
    order = [keys[4]] + DECOYS[:1] + middle[:9] + DECOYS[1:3] + middle[9:] + DECOYS[3:] + [keys[0]]
    lines, blob = [], b""
    for number, key in enumerate(order, 1):
        lines.append("%d:%d:d=%s:%s:anl:" % (number, len(blob), cycle, key))
        blob += payloads.get(key, ("decoy %s " % key).encode() * 7)
    return "\n".join(lines) + "\n", blob


def write_stats(folder, mean=MEAN, std=STD):
    with open(os.path.join(folder, "stats.json"), "w") as f:
        json.dump({"channels": wx.CHANNEL_NAMES, "mean": list(mean), "std": list(std)}, f)


def run_main(*argv):
    """Run gfs_live.main(argv) in this process. Returns (exit code, what it printed, its error text)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gfs.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def slow_cell(fine, i, j):
    """One cell of the model grid (row i, column j) from a fine map, the slow and obvious way."""
    total = weight_sum = 0.0
    for dy in range(-3, 4):
        lat = wx.LATS[i] + 0.25 * dy
        if lat > 90.0:
            continue
        row = int(round((90.0 - lat) * 4))
        for dx in range(-3, 4):
            col = int(round((wx.LONS[j] + 0.25 * dx) * 4)) % 1440
            weight = (0.5 if abs(dy) == 3 else 1.0) * (0.5 if abs(dx) == 3 else 1.0)
            weight *= max(math.cos(math.radians(lat)), 0.0)
            total += weight * fine[row, col]
            weight_sum += weight
    return total / weight_sum


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, headers=()):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the client hung up; some tests make it do that

    def do_GET(self):
        fake = self.server.fake
        byte_range = self.headers.get("Range")
        with fake.lock:
            fake.gets.append((self.path, byte_range, self.headers.get("User-Agent")))
            failing = fake.fail_next.get(self.path, 0)
            if failing:
                fake.fail_next[self.path] = failing - 1
        body = fake.files.get(self.path)
        if failing:
            self.reply(500, b"made-up server error")
        elif body is None:
            self.reply(404, b"<Error><Code>NoSuchKey</Code></Error>")
        elif byte_range and not fake.ignore_range:
            first, last = byte_range[len("bytes="):].split("-")
            first, last = int(first), int(last) if last else len(body) - 1
            self.reply(206, body[first:last + 1], [("Content-Range", "bytes %d-%d/%d" % (first, last, len(body)))])
        else:
            self.reply(200, body)

    def do_POST(self):
        fake = self.server.fake
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with fake.lock:
            fake.posts.append((self.path, body))
        if fake.post_status != 200:
            self.reply(fake.post_status, b'{"error": "made-up failure"}')
        else:
            self.reply(200, json.dumps({"latest": json.loads(body.decode())["time"]}).encode())


class FakeServer:
    """A server on 127.0.0.1 (the OS picks the port) that plays NOAA's bucket and a forecast server."""

    def __init__(self):
        self.files = {}             # path -> bytes
        self.fail_next = {}         # path -> how many of the next GETs get a 500
        self.ignore_range = False   # True: answer byte-range requests with the whole file
        self.post_status = 200
        self.gets = []              # (path, Range header, User-Agent) of every GET
        self.posts = []             # (path, body) of every POST
        self.lock = threading.Lock()
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.httpd.fake = self
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.02,), daemon=True)
        self.thread.start()

    def add_cycle(self, cycle, index_text, blob):
        """Post a cycle. Returns the path of its data file; the index is at that path plus ".idx"."""
        path = "/gfs.%s/%s/atmos/gfs.t%sz.pgrb2.0p25.f000" % (cycle[:8], cycle[8:], cycle[8:])
        self.files[path + ".idx"] = index_text.encode()
        self.files[path] = blob
        return path

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


class FakeServerCase(unittest.TestCase):
    """Points gfs_live at a FakeServer, freezes the clock at NOW and makes the retry waits short."""

    def setUp(self):
        self.fake = FakeServer()
        self.addCleanup(self.fake.stop)
        self.tmp = tempfile.mkdtemp(prefix="gfs_live_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for patch in (mock.patch.object(gfs, "BASE_URL", self.fake.url),
                      mock.patch.object(gfs, "BACKOFF_S", (0.01, 0.01)),
                      mock.patch.object(gfs, "utc_now", lambda: NOW)):
            patch.start()
            self.addCleanup(patch.stop)


class ChannelTableTests(unittest.TestCase):
    def test_table_matches_the_models_channels(self):
        variables = [("u_component_of_wind", "UGRD", 1.0), ("v_component_of_wind", "VGRD", 1.0),
                     ("geopotential", "HGT", 9.80665), ("relative_humidity", "RH", 0.01), ("temperature", "TMP", 1.0)]
        single = {"surface_pressure": "PRES:surface", "mean_sea_level_pressure": "PRMSL:mean sea level",
                  "total_column_water_vapour": "PWAT:entire atmosphere (considered as a single layer)"}
        self.assertEqual(len(gfs.CHANNELS), wx.N_CHANNELS)
        self.assertEqual(len(wx.CHANNEL_NAMES), wx.N_CHANNELS)
        for name, (key, codes, factor) in zip(wx.CHANNEL_NAMES, gfs.CHANNELS):
            if name in single:
                expected = (single[name], 1.0)
            else:
                (stem, variable, expected_factor), = [v for v in variables if v[0] in name]
                if name.startswith("10m_"):
                    level = "10 m above ground"
                elif name.startswith("2m_"):
                    level = "2 m above ground"
                else:
                    level = name[len(stem):] + " mb"        # "temperature850" -> "850 mb"
                expected = (variable + ":" + level, expected_factor)
            self.assertEqual((key, factor), expected, name)
            level = key.split(":", 1)[1]
            level_number = int(level.split()[0]) if level[0].isdigit() else 0
            self.assertEqual(codes, PARAMETERS[key.split(":")[0]] + (level_number,), name)


class IndexTests(unittest.TestCase):
    TEXT = ("1:0:d=2026111400:PRMSL:mean sea level:anl:\n"
            "2:865100:d=2026111400:CLMR:1 hybrid level:anl:\n"
            "3:951746:d=2026111400:TMP:2 m above ground:anl:\n"
            "4:1262924:d=2026111400:PWAT:entire atmosphere (considered as a single layer):anl:\n"
            "\n"
            "5:1478633:d=2026111400:VWSH:PV=-2e-06 (Km^2/kg/s) surface:anl:\n")

    def test_parse_index(self):
        records = gfs.parse_index(self.TEXT)
        self.assertEqual(records, [
            gfs.Record("PRMSL:mean sea level", 0, 865099, "2026111400"),
            gfs.Record("CLMR:1 hybrid level", 865100, 951745, "2026111400"),
            gfs.Record("TMP:2 m above ground", 951746, 1262923, "2026111400"),
            gfs.Record("PWAT:entire atmosphere (considered as a single layer)", 1262924, 1478632, "2026111400"),
            gfs.Record("VWSH:PV=-2e-06 (Km^2/kg/s) surface", 1478633, None, "2026111400"),
        ])
        self.assertEqual(gfs.parse_index(""), [])

    def test_parse_index_refuses_what_it_cannot_read(self):
        for bad in ("<html>not an index</html>\n",
                    "1:zero:d=2026111400:TMP:2 m above ground:anl:\n",
                    "1:0:2026111400:TMP:2 m above ground:anl:\n",
                    "1:500:d=2026111400:TMP:850 mb:anl:\n2:400:d=2026111400:TMP:500 mb:anl:\n",
                    "1:500:d=2026111400:TMP:850 mb:anl:\n2:500:d=2026111400:TMP:500 mb:anl:\n"):
            with self.assertRaises(gfs.GfsError, msg=bad):
                gfs.parse_index(bad)

    def test_pick_ranges_in_the_models_order(self):
        payloads = junk_payloads()
        index, blob = build_file("2026111400", payloads)
        ranges = gfs.pick_ranges(gfs.parse_index(index), "2026111400")
        self.assertEqual(len(ranges), 20)
        for (key, _, _), (first, last) in zip(gfs.CHANNELS, ranges):
            end = len(blob) if last is None else last + 1
            self.assertEqual(blob[first:end], payloads[key], key)
        self.assertEqual(ranges[4][0], 0)                       # PRMSL is the file's first record, at offset 0
        self.assertIsNone(ranges[0][1])                         # the file's last record: no end in the index
        self.assertEqual([r for r in ranges if r[1] is None], [ranges[0]])

    def test_pick_ranges_refuses_a_missing_record_or_another_date(self):
        index, _ = build_file("2026111400", junk_payloads())
        records = gfs.parse_index(index)
        without = [r for r in records if r.key != "RH:850 mb"]
        with self.assertRaises(gfs.GfsError) as caught:
            gfs.pick_ranges(without, "2026111400")
        self.assertIn("RH:850 mb", str(caught.exception))
        twice = records + [r for r in records if r.key == "TMP:850 mb"]
        with self.assertRaises(gfs.GfsError) as caught:
            gfs.pick_ranges(twice, "2026111400")
        self.assertIn("2 records named 'TMP:850 mb'", str(caught.exception))
        with self.assertRaises(gfs.GfsError) as caught:
            gfs.pick_ranges(records, "2026111406")
        self.assertIn("2026111400", str(caught.exception))


class CycleTests(unittest.TestCase):
    def test_newest_cycles(self):
        utc = datetime.timezone.utc
        self.assertEqual(gfs.newest_cycles(datetime.datetime(2026, 11, 14, 5, 54, 20, tzinfo=utc), 3),
                         ["2026111400", "2026111318", "2026111312"])
        self.assertEqual(gfs.newest_cycles(datetime.datetime(2026, 11, 14, 6, 0, 0, tzinfo=utc), 2),
                         ["2026111406", "2026111400"])
        self.assertEqual(gfs.newest_cycles(datetime.datetime(2026, 11, 14, 23, 59, 59, 999999, tzinfo=utc), 1),
                         ["2026111418"])
        self.assertEqual(gfs.newest_cycles(datetime.datetime(2027, 1, 1, 0, 30, tzinfo=utc), 2),
                         ["2027010100", "2026123118"])
        self.assertEqual(len(gfs.newest_cycles(gfs.utc_now(), gfs.WALK_BACK)), 8)

    def test_check_cycle(self):
        for good in ("2026111400", "2026111406", "2026111412", "2026111418", "2024022900"):
            gfs.check_cycle(good)
        for bad in ("20261114", "2026111403", "2026111424", "2026133100", "2026-11-14T00", "latest", "20261114000", ""):
            with self.assertRaises(gfs.GfsError, msg=bad):
                gfs.check_cycle(bad)

    def test_names(self):
        self.assertEqual(gfs.iso_time("2026111400"), "2026-11-14T00")
        self.assertEqual(gfs.iso_time("2027010118"), "2027-01-01T18")
        self.assertEqual(gfs.cycle_url("2026111406"),
                         "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261114/06/atmos/gfs.t06z.pgrb2.0p25.f000")


class RegridTests(unittest.TestCase):
    def test_a_constant_map_stays_constant(self):
        coarse = gfs.regrid(np.full((721, 1440), 273.15))
        self.assertEqual(coarse.shape, (wx.N_LAT, wx.N_LON))
        np.testing.assert_allclose(coarse, 273.15, rtol=1e-12)

    def test_a_map_of_latitude_gives_the_latitude(self):
        coarse = gfs.regrid(np.repeat(FINE_LATS[:, None], 1440, axis=1))
        error = np.abs(coarse - wx.LATS[:, None]).max(axis=1)
        # The cos(latitude) weights pull each mean a little towards the equator; more so near the poles.
        self.assertLess(error[np.abs(wx.LATS) <= 60].max(), 0.01)
        self.assertLess(error[np.abs(wx.LATS) <= 87].max(), 0.1)
        self.assertTrue((np.diff(coarse[:, 0]) > 0).all())              # rows run from south to north
        self.assertLess(abs(coarse[0, 0] + 88.5), 0.2)                  # the first row is the one at 88.5 S
        self.assertTrue(89.25 < coarse[-1, 0] < 90.0)                   # the last row is the cap around 90 N

    def test_a_map_of_longitude_gives_the_longitude(self):
        # Away from the 0 E seam, where this map jumps from 359.75 back to 0.
        coarse = gfs.regrid(np.repeat(FINE_LONS[None, :], 721, axis=0))
        np.testing.assert_allclose(coarse[:, 1:239], np.repeat(wx.LONS[None, 1:239], wx.N_LAT, axis=0), atol=1e-9)

    def test_longitude_wraps_and_edge_points_count_half(self):
        fine = np.zeros((721, 1440))
        fine[:, 1439] = 6.0                     # 359.75 E lies inside the cell centred on 0 E
        coarse = gfs.regrid(fine)
        np.testing.assert_allclose(coarse[:, 0], 1.0, rtol=1e-12)
        np.testing.assert_allclose(coarse[:, 1:], 0.0, atol=1e-12)
        fine = np.zeros((721, 1440))
        fine[:, 1437] = 6.0                     # 359.25 E is the edge between the cells at 358.5 E and 0 E
        coarse = gfs.regrid(fine)
        np.testing.assert_allclose(coarse[:, 239], 0.5, rtol=1e-12)
        np.testing.assert_allclose(coarse[:, 0], 0.5, rtol=1e-12)
        np.testing.assert_allclose(coarse[:, 1:239], 0.0, atol=1e-12)

    def test_against_the_slow_obvious_way(self):
        fine = np.random.RandomState(7).normal(size=(721, 1440))
        coarse = gfs.regrid(fine)
        cells = [(0, 0), (0, 239), (119, 0), (119, 239), (118, 1), (59, 120), (60, 121), (86, 181), (87, 182), (1, 238)]
        for i, j in cells:
            self.assertAlmostEqual(coarse[i, j], slow_cell(fine, i, j), places=10, msg="cell %d, %d" % (i, j))

    def test_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            gfs.regrid(np.zeros((720, 1440)))


class PointTests(unittest.TestCase):
    def test_ohare_on_a_sloping_plane(self):
        # Bilinear interpolation is exact on a map that is a plane in latitude and longitude.
        field = 2.0 * wx.LATS[:, None] + 3.0 * wx.LONS[None, :]
        self.assertEqual(gfs.OHARE, (41.9602, -87.9316))
        self.assertAlmostEqual(gfs.point_value(field, 41.9602, -87.9316),
                               2.0 * 41.9602 + 3.0 * (360.0 - 87.9316), places=9)

    def test_ohare_weights_by_hand(self):
        # O'Hare lies between rows 86 (40.5 N) and 87 (42.0 N) and columns 181 (271.5 E) and 182 (273.0 E).
        self.assertEqual((wx.LATS[86], wx.LATS[87], wx.LONS[181], wx.LONS[182]), (40.5, 42.0, 271.5, 273.0))
        field = np.zeros((wx.N_LAT, wx.N_LON))
        field[86, 181], field[86, 182], field[87, 181], field[87, 182] = 1.0, 2.0, 3.0, 4.0
        fy = (41.9602 - 40.5) / 1.5
        fx = (360.0 - 87.9316 - 271.5) / 1.5
        expected = (1 - fy) * ((1 - fx) * 1.0 + fx * 2.0) + fy * ((1 - fx) * 3.0 + fx * 4.0)
        self.assertAlmostEqual(gfs.point_value(field, *gfs.OHARE), expected, places=9)

    def test_grid_points_wrap_and_edges(self):
        field = np.arange(wx.N_LAT * wx.N_LON, dtype=np.float64).reshape(wx.N_LAT, wx.N_LON)
        self.assertAlmostEqual(gfs.point_value(field, wx.LATS[10], wx.LONS[20]), field[10, 20])
        self.assertAlmostEqual(gfs.point_value(field, 30.0, -90.0), gfs.point_value(field, 30.0, 270.0))
        self.assertAlmostEqual(gfs.point_value(field, wx.LATS[10], 359.25), 0.5 * (field[10, 239] + field[10, 0]))
        self.assertAlmostEqual(gfs.point_value(field, wx.LATS[10], 360.0), field[10, 0])
        self.assertAlmostEqual(gfs.point_value(field, wx.LATS[10], -1.5), field[10, 239])
        self.assertAlmostEqual(gfs.point_value(field, 90.0, 0.0), field[119, 0])
        self.assertAlmostEqual(gfs.point_value(field, -90.0, 0.0), field[0, 0])      # south of the grid


class StateTests(unittest.TestCase):
    def test_to_state_gives_float16_z_scores(self):
        mean, std = np.array(MEAN), np.array(STD)
        z = np.linspace(-2.0, 2.0, 20)
        state = gfs.to_state(np.ones((20, 120, 240)) * (mean + z * std)[:, None, None], mean, std)
        self.assertEqual((state.dtype, state.shape), (np.dtype(np.float16), (20, 120, 240)))
        np.testing.assert_allclose(state.astype(np.float64).mean(axis=(1, 2)), z, atol=2e-3)

    def test_npy_bytes_load_back(self):
        state = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        back = np.load(io.BytesIO(gfs.npy_bytes(state)))
        self.assertEqual(back.dtype, np.float16)
        np.testing.assert_array_equal(back, state)


class FetchTests(FakeServerCase):
    def setUp(self):
        super().setUp()
        self.payloads = junk_payloads()
        self.index, self.blob = build_file("2026111400", self.payloads)
        self.path = self.fake.add_cycle("2026111400", self.index, self.blob)
        self.expected = [self.payloads[key] for key, _, _ in gfs.CHANNELS]

    def test_fetch_cycle_gets_each_record_by_its_byte_range(self):
        records, nbytes = gfs.fetch_cycle("2026111400")
        self.assertEqual(records, self.expected)
        self.assertEqual(nbytes, len(self.index) + sum(len(r) for r in self.expected))
        self.assertEqual(self.fake.gets[0][:2], (self.path + ".idx", None))
        ranges = [byte_range for path, byte_range, _ in self.fake.gets if path == self.path]
        self.assertEqual(len(ranges), 20)                                   # never the whole file
        self.assertTrue(all(re.match(r"^bytes=\d+-\d*$", r) for r in ranges), ranges)
        self.assertEqual(len([r for r in ranges if r.startswith("bytes=0-")]), 1)   # the record at offset 0
        self.assertEqual(len([r for r in ranges if r.endswith("-")]), 1)            # the file's last record
        agents = set(agent for _, _, agent in self.fake.gets)
        self.assertEqual(agents, {gfs.USER_AGENT})
        for word in ("@", "weather", "desk", "forecast", "hackathon"):      # nothing that names us
            self.assertNotIn(word, gfs.USER_AGENT.lower())

    def test_a_server_error_is_tried_again(self):
        self.fake.fail_next[self.path] = 2
        with contextlib.redirect_stderr(io.StringIO()) as notes:
            records, _ = gfs.fetch_cycle("2026111400")
        self.assertEqual(records, self.expected)
        self.assertEqual(len([g for g in self.fake.gets if g[0] == self.path]), 22)
        self.assertIn("HTTP 500", notes.getvalue())

    def test_it_gives_up_after_three_tries(self):
        self.fake.fail_next[self.path + ".idx"] = 99
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(gfs.GfsError) as caught:
                gfs.fetch_cycle("2026111400")
        self.assertNotIsInstance(caught.exception, gfs.NotPosted)
        self.assertIn("after 3 tries", str(caught.exception))
        self.assertEqual(len(self.fake.gets), 3)

    def test_a_missing_cycle_is_not_posted_and_not_tried_again(self):
        with self.assertRaises(gfs.NotPosted):
            gfs.fetch_cycle("2026111406")
        self.assertEqual(len(self.fake.gets), 1)

    def test_an_index_without_its_data_file_is_not_posted(self):
        del self.fake.files[self.path]
        with self.assertRaises(gfs.NotPosted):
            gfs.fetch_cycle("2026111400")

    def test_a_server_that_ignores_byte_ranges_is_refused(self):
        self.fake.ignore_range = True
        with self.assertRaises(gfs.GfsError) as caught:
            gfs.fetch_cycle("2026111400")
        self.assertIn("206", str(caught.exception))

    def test_a_dead_server_is_a_clear_error(self):
        self.fake.stop()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(gfs.GfsError) as caught:
                gfs.http_get(self.fake.url + self.path + ".idx")
        self.assertIn("after 3 tries", str(caught.exception))


class EndToEndTests(FakeServerCase):
    def setUp(self):
        super().setUp()
        self.data = os.path.join(self.tmp, "data")
        os.makedirs(self.data)
        write_stats(self.data)
        self.out = os.path.join(self.tmp, "states")         # not made here: the tool makes it
        self.saved = os.path.join(self.out, "2026111400.npy")

    def post(self, payloads):
        self.fake.add_cycle("2026111400", *build_file("2026111400", payloads))

    def test_newest_cycle_is_saved_and_pushed(self):
        self.post(grib_payloads(t2m_field=t2m_made_up(FINE_LATS[:, None], FINE_LONS[None, :])))
        code, out, err = run_main("--out", self.out, "--data", self.data,
                                  "--push", self.fake.url, "--push", self.fake.url + "/")
        self.assertEqual(code, 0, out + err)

        # It walked back from the 06Z cycle, which is not posted, to 00Z.
        self.assertIn("cycle 2026111406 is not posted yet", out)
        self.assertIn("cycle 2026111400 (GFS analysis of 2026-11-14T00 UTC)", out)
        self.assertEqual(os.listdir(self.out), ["2026111400.npy"])

        # The file: float16 z-scores, the constants where they belong, the temperature map the right way round.
        state = np.load(self.saved)
        self.assertEqual((state.dtype, state.shape), (np.dtype(np.float16), (20, 120, 240)))
        for c in range(20):
            if c != wx.CH_T2M:
                np.testing.assert_allclose(state[c].astype(np.float64), Z[c], atol=2e-3, err_msg=wx.CHANNEL_NAMES[c])
        expected = (t2m_made_up(wx.LATS[:, None], wx.LONS[None, :]) - MEAN[2]) / STD[2]
        inner = np.abs(wx.LATS) <= 80
        np.testing.assert_allclose(state[wx.CH_T2M].astype(np.float64)[inner], expected[inner], atol=5e-3)

        # What it printed: every channel's mean z-score, O'Hare, megabytes, seconds, the file.
        for c, name in enumerate(wx.CHANNEL_NAMES):
            if c != wx.CH_T2M:
                self.assertRegex(out, r"(?m)^  %s +%s$" % (re.escape(name), re.escape("%+.2f" % Z[c])))
        self.assertRegex(out, r"(?m)^  2m_temperature +[-+]\d\.\d\d$")
        self.assertIn("largest: -0.21 (10m_u_component_of_wind)", out)
        ohare = float(re.search(r"O'Hare 2-m temperature: (-?[0-9.]+) C", out).group(1))
        self.assertAlmostEqual(ohare, t2m_made_up(41.9602, 360.0 - 87.9316) - 273.15, delta=0.05)
        megabytes, seconds = re.search(r"fetched ([0-9.]+) MB, took ([0-9.]+) s", out).groups()
        self.assertAlmostEqual(float(megabytes), 2.1, delta=0.1)       # one packed map of 2 MB, 19 tiny ones
        self.assertLess(float(seconds), 60.0)
        self.assertIn("wrote %s" % os.path.abspath(self.saved), out)

        # The pushes: POST /v1/state with the time and the file's bytes; the answers are printed.
        self.assertEqual([path for path, _ in self.fake.posts], ["/v1/state", "/v1/state"])
        message = json.loads(self.fake.posts[0][1].decode())
        self.assertEqual(sorted(message), ["npy_b64", "time"])
        self.assertEqual(message["time"], "2026-11-14T00")
        with open(self.saved, "rb") as f:
            self.assertEqual(base64.b64decode(message["npy_b64"]), f.read())
        self.assertEqual(out.count('/v1/state: 200 {"latest": "2026-11-14T00"}'), 2)

    def test_a_far_off_mean_z_score_is_refused_unless_forced(self):
        z = list(Z)
        z[16] = 1.5                                         # geopotential50 far from the statistics
        self.post(grib_payloads(z=z))
        args = ("--out", self.out, "--data", self.data, "--cycle", "2026111400", "--push", self.fake.url)
        code, out, err = run_main(*args)
        self.assertEqual(code, 1, out + err)
        self.assertIn("nothing saved", err)
        self.assertIn("geopotential50 is +1.50", err)
        self.assertIn("--force", err)
        self.assertEqual(os.listdir(self.out), [])
        self.assertEqual(self.fake.posts, [])
        code, out, err = run_main(*(args + ("--force",)))
        self.assertEqual(code, 0, out + err)
        self.assertIn("warning", out)
        self.assertEqual(os.listdir(self.out), ["2026111400.npy"])
        self.assertEqual(len(self.fake.posts), 1)

    def test_a_mean_z_score_inside_the_limit_is_saved(self):
        z = list(Z)
        z[16] = -0.9
        self.post(grib_payloads(z=z))
        code, out, err = run_main("--out", self.out, "--data", self.data, "--cycle", "2026111400")
        self.assertEqual(code, 0, out + err)
        self.assertIn("largest: -0.90 (geopotential50)", out)
        self.assertTrue(os.path.exists(self.saved))

    def test_a_cycle_that_is_not_posted(self):
        code, out, err = run_main("--out", self.out, "--data", self.data, "--cycle", "2026111406")
        self.assertEqual(code, 1)
        self.assertIn("cycle 2026111406 is not on NOAA's server", err)
        self.assertEqual(os.listdir(self.out), [])

    def test_no_cycle_posted_at_all(self):
        code, out, err = run_main("--out", self.out, "--data", self.data)
        self.assertEqual(code, 1)
        self.assertIn("none of the last 8 cycles (2026111406 back to 2026111212)", err)
        self.assertEqual(len(self.fake.gets), 8)

    def test_a_failed_push_keeps_the_file(self):
        self.post(grib_payloads())
        self.fake.post_status = 500
        code, out, err = run_main("--out", self.out, "--data", self.data, "--push", self.fake.url)
        self.assertEqual(code, 1)
        self.assertIn("push to %s/v1/state failed: HTTP 500" % self.fake.url, err)
        self.assertTrue(os.path.exists(self.saved))

    def test_a_record_that_is_not_what_the_index_says_is_refused(self):
        payloads = grib_payloads()
        payloads["TMP:850 mb"] = make_grib("TMP:500 mb", 250.0)
        self.post(payloads)
        code, out, err = run_main("--out", self.out, "--data", self.data)
        self.assertEqual(code, 1)
        self.assertIn("'TMP:850 mb' says (category, number, level) = (0, 0, 500)", err)
        self.assertEqual(os.listdir(self.out), [])

    def test_a_record_on_another_grid_is_refused(self):
        payloads = grib_payloads()
        payloads["HGT:500 mb"] = make_grib("HGT:500 mb", 5500.0, south_first=True)
        self.post(payloads)
        code, out, err = run_main("--out", self.out, "--data", self.data)
        self.assertEqual(code, 1)
        self.assertIn("'HGT:500 mb' is not on the GFS 0.25-degree grid", err)
        self.assertEqual(os.listdir(self.out), [])

    def test_a_cut_off_record_is_refused(self):
        payloads = grib_payloads()
        payloads["RH:500 mb"] = payloads["RH:500 mb"][:-9]
        self.post(payloads)
        code, out, err = run_main("--out", self.out, "--data", self.data)
        self.assertEqual(code, 1)
        self.assertIn("'RH:500 mb' are not one whole GRIB2 record", err)

    def test_mistakes_are_caught_before_any_download(self):
        self.post(grib_payloads())
        nowhere = os.path.join(self.tmp, "nowhere")
        code, out, err = run_main("--out", self.out, "--data", nowhere, "--cycle", "2026111400")
        self.assertEqual(code, 1)
        self.assertIn("stats.json", err)
        self.assertIn("--data", err)
        code, out, err = run_main("--out", self.out, "--data", self.data, "--push", "127.0.0.1:8100")
        self.assertEqual(code, 1)
        self.assertIn("--push wants a URL", err)
        self.assertEqual(self.fake.gets, [])
        self.assertFalse(os.path.exists(self.out))

    def test_the_command_line_refuses_a_bad_cycle(self):
        # A real run of the script; it stops before it would touch the network.
        command = [sys.executable, "-B", os.path.join(WEATHER, "gfs_live.py"),
                   "--out", self.out, "--data", self.data, "--cycle", "20261114"]
        done = subprocess.run(command, capture_output=True, text=True, timeout=50)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("YYYYMMDDHH", done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertFalse(os.path.exists(self.out))


@unittest.skipUnless(os.environ.get("WX_LIVE") == "1", "set WX_LIVE=1 to fetch about 18 MB from NOAA")
class LiveTest(unittest.TestCase):
    def test_the_newest_real_analysis(self):
        data = os.environ.get("WX_DATA") or os.path.join(REPO, ".local", "bundle")
        if not os.path.exists(os.path.join(data, "stats.json")):
            self.skipTest("no stats.json in %s; set WX_DATA" % data)
        out_dir = tempfile.mkdtemp(prefix="gfs_live_test_")
        self.addCleanup(shutil.rmtree, out_dir, True)
        code, out, err = run_main("--out", out_dir, "--data", data)
        self.assertEqual(code, 0, out + err)
        (name,) = os.listdir(out_dir)
        self.assertRegex(name, r"^20\d{6}(00|06|12|18)\.npy$")
        state = np.load(os.path.join(out_dir, name))
        self.assertEqual((state.dtype, state.shape), (np.dtype(np.float16), (20, 120, 240)))
        self.assertTrue(np.isfinite(state).all())
        self.assertLess(np.abs(state.astype(np.float64).mean(axis=(1, 2))).max(), 1.0)
        megabytes = float(re.search(r"fetched ([0-9.]+) MB", out).group(1))
        self.assertTrue(10.0 < megabytes < 30.0, megabytes)
        ohare = float(re.search(r"O'Hare 2-m temperature: (-?[0-9.]+) C", out).group(1))
        self.assertTrue(-40.0 < ohare < 45.0, ohare)

        # Geography says whether the maps are the right way up and start at 0 E. Surface pressure is low
        # on high ground: Antarctica's ice sheet and Tibet. It is near 1000 hPa over the Arctic Ocean and
        # over the Pacific at Tibet's latitude.
        mean, std = wx.read_stats(data)
        pressure = state[wx.CH_SP].astype(np.float64) * std[wx.CH_SP] + mean[wx.CH_SP]
        self.assertLess(pressure[wx.LATS <= -80].mean(), 80000.0)
        self.assertGreater(pressure[wx.LATS >= 80].mean(), 95000.0)
        band = pressure[(wx.LATS >= 30) & (wx.LATS <= 35)]
        self.assertLess(band[:, (wx.LONS >= 80) & (wx.LONS <= 95)].mean(), 70000.0)
        self.assertGreater(band[:, (wx.LONS >= 170) & (wx.LONS <= 190)].mean(), 99000.0)
        kelvin = state[wx.CH_T2M].astype(np.float64) * std[wx.CH_T2M] + mean[wx.CH_T2M]
        self.assertGreater(kelvin[np.abs(wx.LATS) <= 15].mean(), 293.0)         # the tropics are warm


if __name__ == "__main__":
    unittest.main()
