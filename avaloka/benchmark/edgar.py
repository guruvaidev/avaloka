"""Fetch one real SEC filing, when explicitly asked to.

The 10-K benchmark task is synthetic by default -- a workbook built to the
shape SEC's ``Financial_Report.xlsx`` has, with planted ground truth, so it
runs with no network and can be graded exactly. This module is the opt-in
other half: ``--edgar`` fetches an actual filing so the shape can be checked
against the real thing rather than against our model of it.

It is opt-in for the same reason the Kaggle adapter is: it reaches the network,
and ``python -m avaloka.benchmark run`` with every key unset must still work
and must call nothing. It returns ``None`` rather than raising whenever the
filing cannot be had, so the runner skips the task cleanly.

**A 10-K as filed is PDF or HTML, and Avaloka reads neither.** What SEC also
publishes per filing is ``Financial_Report.xlsx`` and the XBRL instance
document -- both in the declared upload set. Those are what this fetches.
Nothing here should be read as implying PDF support.

**Licence.** EDGAR filings are US government works in the public domain, so
there is no licence problem with reading one. SEC's access policy does require
a declared, contactable ``User-Agent`` and rate limiting, which is why
``AVALOKA_EDGAR_USER_AGENT`` is mandatory: a default would either be a fake
contact, which the policy forbids, or somebody's real address baked into a
public repository. No filing is committed to the tree.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from avaloka.benchmark.spec import GroundTruth

LOGGER = logging.getLogger(__name__)

#: SEC asks for no more than ten requests a second; one every two seconds is
#: far inside that and makes the whole thing a non-event for their servers.
_MIN_INTERVAL_S = 2.0
_last_request_at = 0.0

#: SEC's fair-access policy requires a real contact in the User-Agent. There is
#: deliberately no default: see the module docstring.
USER_AGENT_ENV = "AVALOKA_EDGAR_USER_AGENT"

_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/Financial_Report.xlsx"


def _user_agent() -> Optional[str]:
    value = (os.environ.get(USER_AGENT_ENV) or "").strip()
    if not value:
        LOGGER.warning(
            "%s is not set, so the EDGAR task is skipped. SEC's fair-access "
            "policy requires a declared contact, e.g. "
            '%s="Avaloka benchmark (you@example.com)"', USER_AGENT_ENV, USER_AGENT_ENV)
        return None
    # A User-Agent with no contact in it does not satisfy the policy, and a
    # 403 three layers down is a worse way to find that out.
    if "@" not in value:
        LOGGER.warning(
            "%s=%r carries no contact address; SEC's policy asks for one and may "
            "return 403. Skipping rather than sending it.", USER_AGENT_ENV, value)
        return None
    return value


def _throttle() -> None:
    global _last_request_at
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def ensure_edgar_filing(cik: Optional[str], accession: Optional[str],
                        workdir: Path) -> Optional[Path]:
    """Download one filing's ``Financial_Report.xlsx``, or return None.

    Returns None -- never raises -- on a missing contact, a missing argument, a
    network error, an HTTP error or an implausibly small response, so an
    unavailable filing is a skipped task rather than a failed run.
    """
    if not cik or not accession:
        LOGGER.warning("EDGAR task has no cik/accession; skipping.")
        return None
    agent = _user_agent()
    if agent is None:
        return None

    url = _ARCHIVE.format(cik=str(cik).lstrip("0"), accession=accession.replace("-", ""))
    dest = Path(workdir) / f"edgar_{accession.replace('-', '')}_Financial_Report.xlsx"
    if dest.exists() and dest.stat().st_size > 4096:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={
        "User-Agent": agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    })
    try:
        _throttle()
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        LOGGER.warning("Could not fetch %s: %s", url, exc)
        return None

    # An xlsx is a zip. An HTML error page that arrived with a 200 is not, and
    # writing it under an .xlsx name would hand the reader a file whose failure
    # looks like a reader defect rather than a failed download.
    if len(payload) < 4096 or not payload.startswith(b"PK"):
        LOGGER.warning(
            "%s returned %d bytes that are not a zip/xlsx (starts %r); skipping.",
            url, len(payload), payload[:16])
        return None

    dest.write_bytes(payload)
    LOGGER.info("Fetched %s (%d bytes) -> %s", url, len(payload), dest)
    return dest


#: What can honestly be asserted about *someone else's* filing.
#:
#: Ground truth cannot be planted in a document we did not write, and guessing
#: Apple's line-item labels here would make the task grade our guess rather
#: than the reader. So this declares no columns and no row count: the EDGAR
#: task grades only the shape questions that hold for any well-formed filing --
#: that no column comes back as a placeholder, and that the generated narration
#: names only columns that exist. The synthetic SEC-shaped workbook in
#: ``datasets_spreadsheet`` is where the planted figure is graded.
EDGAR_SHAPE_TRUTH = GroundTruth(
    columns=[],
    n_rows=None,
    header_row=None,
    shape=("a real SEC Financial_Report.xlsx: multi-sheet, title rows, period "
           "columns as headers, thousands scaling, parenthesised negatives"),
)
