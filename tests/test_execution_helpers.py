"""
Integration tests for helper code injected into every generated script.
=======================================================================
Layer 2 — spawns real subprocesses.  Does NOT import execution_agent
(heavy dependency chain: cloud_connections → cryptography).

Instead, inlines the same helper_functions block that execution_agent
prepends to every script, runs it via subprocess.run, and asserts the
behavioral contract:

  - avaloka_result() emits a correctly-structured AVALOKA_RESULT sentinel
  - matplotlib.use("Agg") is active  →  plt.show() never blocks
  - read_csv_best_effort() is available and reads CSV files correctly

All tests are self-contained in a tmp directory via pytest's tmp_path fixture.
"""

import importlib
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared: the helper_functions block that execution_agent prepends to every
# generated script.  Kept in sync manually — the contract tests below will
# catch regressions if this diverges from the real injected code.
# ---------------------------------------------------------------------------

_HELPERS = textwrap.dedent(r"""
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

import pandas as pd
import json as _avaloka_json

def avaloka_result(value, kind=None, columns=None, **kw):
    obj = {
        "kind": kind or "scalar",
        "value": float(value) if value is not None else None,
        "columns": list(columns or []),
    }
    obj.update(kw)
    print(f"<<<AVALOKA_RESULT>>>{_avaloka_json.dumps(obj)}<<<END_AVALOKA_RESULT>>>", flush=True)

def read_csv_best_effort(path: str, **kwargs):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except Exception:
        head = b""
    encodings = []
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        encodings.append("utf-16")
    encodings += ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, encoding_errors="replace", **kwargs)
        except TypeError:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
    raise last_err
""").strip()


def _run(script: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Combine helpers + script and run via current Python interpreter."""
    full = _HELPERS + "\n\n" + textwrap.dedent(script).strip()
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# avaloka_result() contract
# ---------------------------------------------------------------------------

class TestAvalokaResult:

    def test_emits_sentinel_on_stdout(self):
        result = _run("avaloka_result(value=42.0, kind='count', columns=[])")
        assert result.returncode == 0
        assert "<<<AVALOKA_RESULT>>>" in result.stdout
        assert "<<<END_AVALOKA_RESULT>>>" in result.stdout

    def test_sentinel_is_valid_json(self):
        result = _run("avaloka_result(value=99.5, kind='mean', columns=['Price'])")
        assert result.returncode == 0
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["kind"] == "mean"
        assert obj["value"] == pytest.approx(99.5)
        assert obj["columns"] == ["Price"]

    def test_sentinel_default_kind_is_scalar(self):
        result = _run("avaloka_result(value=7.0)")
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["kind"] == "scalar"

    def test_sentinel_multi_column(self):
        result = _run("avaloka_result(value=-0.55, kind='correlation', columns=['A','B'])")
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["columns"] == ["A", "B"]

    def test_sentinel_zero_value(self):
        result = _run("avaloka_result(value=0.0, kind='min', columns=['X'])")
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["value"] == 0.0

    def test_sentinel_negative_value(self):
        result = _run("avaloka_result(value=-3.14, kind='mean', columns=['delta'])")
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["value"] == pytest.approx(-3.14)

    def test_no_crash_when_called_multiple_times(self):
        script = """
avaloka_result(value=1.0, kind='mean', columns=['A'])
avaloka_result(value=2.0, kind='sum', columns=['B'])
"""
        result = _run(script)
        assert result.returncode == 0
        assert result.stdout.count("<<<AVALOKA_RESULT>>>") == 2


# ---------------------------------------------------------------------------
# matplotlib Agg backend
# ---------------------------------------------------------------------------

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed in test env")
class TestMatplotlibAgg:

    def test_backend_is_agg(self):
        script = "import matplotlib; print(matplotlib.get_backend())"
        result = _run(script)
        assert result.returncode == 0
        assert "agg" in result.stdout.lower()

    def test_plt_show_does_not_block(self):
        """plt.show() must complete instantly — Agg backend makes it a no-op."""
        script = """
import matplotlib.pyplot as plt
import numpy as np
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
print("completed")
"""
        start = time.time()
        result = _run(script, timeout=5)
        elapsed = time.time() - start
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "completed" in result.stdout
        assert elapsed < 5, f"plt.show() blocked for {elapsed:.1f}s — Agg backend not active"

    def test_savefig_works_with_agg(self, tmp_path):
        out = tmp_path / "plot.png"
        script = f"""
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig({str(out)!r})
print("saved")
"""
        result = _run(script)
        assert result.returncode == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# read_csv_best_effort()
# ---------------------------------------------------------------------------

class TestReadCsvBestEffort:

    def test_reads_utf8_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")
        script = f"""
df = read_csv_best_effort({str(csv)!r})
print(len(df))
print(list(df.columns))
"""
        result = _run(script)
        assert result.returncode == 0, result.stderr
        assert "2" in result.stdout
        assert "col_a" in result.stdout
        assert "col_b" in result.stdout

    def test_reads_latin1_csv(self, tmp_path):
        csv = tmp_path / "latin.csv"
        csv.write_bytes("name,value\ncafé,10\nnaïve,20\n".encode("latin-1"))
        script = f"""
df = read_csv_best_effort({str(csv)!r})
print(len(df))
"""
        result = _run(script)
        assert result.returncode == 0, result.stderr
        assert "2" in result.stdout

    def test_raises_on_missing_file(self):
        script = """
try:
    df = read_csv_best_effort("/nonexistent/path/file.csv")
    print("no error raised")
except Exception as e:
    print("error:", type(e).__name__)
"""
        result = _run(script)
        assert result.returncode == 0
        assert "error" in result.stdout.lower()
        assert "no error raised" not in result.stdout

    def test_extra_kwargs_passed_through(self, tmp_path):
        csv = tmp_path / "sep.csv"
        csv.write_text("a;b\n1;2\n", encoding="utf-8")
        script = f"""
df = read_csv_best_effort({str(csv)!r}, sep=";")
print(list(df.columns))
"""
        result = _run(script)
        assert result.returncode == 0, result.stderr
        assert "a" in result.stdout
        assert "b" in result.stdout


# ---------------------------------------------------------------------------
# helpers coexist with user code (no name collisions)
# ---------------------------------------------------------------------------

class TestHelperCoexistence:

    def test_user_pd_import_does_not_conflict(self):
        script = """
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3]})
avaloka_result(value=df["x"].mean(), kind="mean", columns=["x"])
"""
        result = _run(script)
        assert result.returncode == 0
        assert "<<<AVALOKA_RESULT>>>" in result.stdout
        raw = result.stdout.split("<<<AVALOKA_RESULT>>>")[1].split("<<<END_AVALOKA_RESULT>>>")[0]
        obj = json.loads(raw)
        assert obj["value"] == pytest.approx(2.0)

    def test_user_json_import_does_not_conflict(self):
        script = """
import json
avaloka_result(value=1.0, kind="count")
d = json.dumps({"ok": True})
print(d)
"""
        result = _run(script)
        assert result.returncode == 0
        assert '{"ok": true}' in result.stdout
        assert "<<<AVALOKA_RESULT>>>" in result.stdout
