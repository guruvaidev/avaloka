"""How generated code's ``main`` receives its frames.

The coder is told the primary dataset arrives as ``main(df)`` and that any
other dataset is read inside ``main``. Models write a join the other natural
way just as often -- ``def main(df_customers, df_orders)`` with both frames
loaded in the ``__main__`` block -- and the driver, which strips that block
and calls ``main(df)``, crashed it with "main() missing 1 required positional
argument: 'df_orders'" (web e2e, 2026-10-18). Both the validator's sample run
and the execution driver now bind ``main``'s parameters here instead:

* a ``main`` with at most one required parameter gets the primary frame, as
  it always did;
* otherwise each required parameter is matched, by name, to one dataset of
  the turn: ``df_orders``, ``orders_df``, ``orders`` and ``df_order`` all name
  the dataset whose alias or file stem is ``orders``; ``df`` is the primary;
* a parameter that names no dataset, or names one ambiguously, is an error
  that says which parameter and which datasets there were -- never a guess.

THIS FILE IS ALSO SHIPPED AS TEXT
    app/agents/execution_agent.py splices this module's source into the
    script it runs, where the repository may not exist: standard library only,
    no relative imports, no ``from __future__`` line.
"""
import inspect
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional

#: The parameter names that mean "the primary frame".
_PRIMARY_NAMES = frozenset({"df", "data", "frame", "dataframe", "df_main", "main_df", "primary"})


def _key(name: Any) -> str:
    """``orders.csv``, ``Orders``, ``df_orders`` and ``orders_df`` -> ``orders``."""
    stem = os.path.splitext(os.path.basename(str(name)))[0].lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    stem = re.sub(r"^(?:df|data|frame)_", "", stem)
    stem = re.sub(r"_(?:df|data|frame)$", "", stem)
    return stem


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def required_parameters(main: Callable[..., Any]) -> List[str]:
    """The positional parameters ``main`` cannot be called without."""
    try:
        params = inspect.signature(main).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p.name for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]


def bind_main_frames(main: Callable[..., Any], primary: Any,
                     sources: Optional[Mapping[str, Any]],
                     load: Callable[[Any], Any],
                     primary_source: Optional[Any] = None) -> List[Any]:
    """The positional arguments to call ``main`` with.

    ``sources`` maps a dataset's alias or file name to what ``load`` reads
    (a path, usually); several keys may share one source. ``primary`` is the
    frame already loaded from ``primary_source``, so a parameter naming the
    primary dataset reuses it rather than reading it again.
    """
    names = required_parameters(main)
    if len(names) <= 1:
        return [primary]
    by_key: Dict[str, List[Any]] = {}
    for label, source in (sources or {}).items():
        for key in {_key(label), _singular(_key(label))}:
            if key and source not in by_key.setdefault(key, []):
                by_key[key].append(source)
    cache: Dict[str, Any] = {}
    args: List[Any] = []
    for name in names:
        if name.lower() in _PRIMARY_NAMES:
            args.append(primary)
            continue
        key = _key(name)
        found = by_key.get(key) or by_key.get(_singular(key)) or []
        if len(found) != 1:
            available = sorted({_key(label) for label in (sources or {})}) or ["(none)"]
            why = "names no dataset" if not found else "names more than one dataset"
            raise TypeError(
                f"main() parameter {name!r} {why} of this turn; the datasets are: "
                f"{', '.join(available)}. Name each parameter after its dataset "
                f"(df_<alias>) or read the extra datasets inside main(df).")
        source = found[0]
        if primary_source is not None and str(source) == str(primary_source):
            args.append(primary)
            continue
        if str(source) not in cache:
            cache[str(source)] = load(source)
        args.append(cache[str(source)])
    return args
