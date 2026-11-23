"""What a model call cost, when the provider did not say.

OpenRouter reports the cost of each call in its usage block, and the recorder
uses that figure as given. Other providers report tokens only. For those, a
price is used only if the operator configured one:

    AVALOKA_MODEL_PRICES='{"openai/gpt-oss-120b": [0.15, 0.60]}'

The pair is USD per million input tokens and per million output tokens.

There is deliberately no built-in table. Prices change without notice, and a
stale constant would put a confident wrong number in front of a user. Unknown
cost is reported as unknown (``None``) and the trace says so.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_cache: Tuple[Optional[str], Dict[str, Tuple[float, float]]] = (None, {})


def _table() -> Dict[str, Tuple[float, float]]:
    global _cache
    raw = os.getenv("AVALOKA_MODEL_PRICES") or ""
    if raw == _cache[0]:
        return _cache[1]
    table: Dict[str, Tuple[float, float]] = {}
    if raw:
        try:
            for model, pair in (json.loads(raw) or {}).items():
                p_in, p_out = float(pair[0]), float(pair[1])
                if p_in >= 0 and p_out >= 0:
                    table[str(model)] = (p_in, p_out)
        except Exception:  # noqa: BLE001
            logger.warning("[trace] AVALOKA_MODEL_PRICES is not valid JSON of "
                           "{model: [in_per_mtok, out_per_mtok]}; ignoring it")
            table = {}
    _cache = (raw, table)
    return table


def cost_for(model: Optional[str], tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    """USD for one call, or ``None`` when no configured price covers it."""
    if not model or tokens_in is None or tokens_out is None:
        return None
    price = _table().get(model)
    if price is None:
        return None
    return round((tokens_in * price[0] + tokens_out * price[1]) / 1_000_000, 8)
