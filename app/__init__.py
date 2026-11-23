# Before anything under app/ can import daft: its import-time usage beacon is
# off unless a deployment opts back in (app/core/thirdparty_optout.py).
from app.core.thirdparty_optout import disable_thirdparty_telemetry as _disable_thirdparty_telemetry

_disable_thirdparty_telemetry()
