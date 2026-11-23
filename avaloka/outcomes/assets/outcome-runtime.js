/*
 * Avaloka Outcome Report runtime: the figure formatter and the what-if
 * scenario model, for the HTML export and the UI.
 *
 * This is the JavaScript twin of avaloka/outcomes/format.py and
 * avaloka/outcomes/scenario.py. The rules and the operation order are the
 * same, so both give the same strings and the same floats for the same input;
 * avaloka/outcomes/golden/*.json holds them equal, run by both
 * tests/avaloka/outcomes/test_spec.py and ui/tests/unit/outcome-runtime.test.ts.
 * Change one side and the other in the same commit, then the golden vectors.
 *
 * Plain ES2019 script, no dependencies, no module syntax: it is inlined into
 * the export under a CSP that allows only hashed scripts, so it never uses
 * eval, Function or string timers. It defines one global, AvalokaOutcome
 * (or module.exports under CommonJS).
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  } else {
    root.AvalokaOutcome = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : typeof self !== "undefined" ? self : this,
function () {
  "use strict";

  var SPEC_VERSION = "1.0";
  var UNITS = ["count", "currency", "percent", "pp", "ratio", "days", "unitless"];
  var MAX_DECIMALS = 4;
  var CURRENCY_SYMBOLS = { USD: "$", EUR: "€", GBP: "£", JPY: "¥", INR: "₹" };
  var SCALED_UNITS = ["count", "currency", "unitless"];
  var SUFFIXES = ["", "K", "M", "B", "T"];
  var SUFFIX_FROM_DIGITS = 5;
  var COMPACT_FROM_DIGITS = 4;
  var CURRENCY_CODE = /^[A-Z]{3}$/;
  var NUMBER_TEXT = /^(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/;
  var PLACEHOLDER = /\{(f[0-9]{1,3})\}/g;

  var Z_DEFAULT = 1.96;
  var SUPPORTED_FORMS = ["linear", "step"];
  var SURE_LEVELS = ["fairly_sure", "direction_only", "either_way"];
  var SURE_TEXT = {
    fairly_sure: "Fairly sure of the direction and size.",
    direction_only: "Sure of the direction, not the size.",
    either_way: "Could go either way at this setting."
  };
  var FAIRLY_SURE_R = 0.5;

  class OutcomeError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "OutcomeError";
      this.code = code;
    }
  }

  function has(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  // ── exact decimal digits: value = ±digits · 10^exponent ──────────────────

  function toDecimal(value) {
    if (typeof value !== "number") {
      throw new OutcomeError("value", String(value) + " is not a number");
    }
    if (!isFinite(value)) {
      throw new OutcomeError("value", String(value) + " is not finite");
    }
    var match = NUMBER_TEXT.exec(String(value));
    if (match === null) {
      throw new OutcomeError("value", "cannot read " + String(value));
    }
    var frac = match[3] || "";
    var digits = (match[2] + frac).replace(/^0+/, "") || "0";
    return { negative: match[1] === "-", digits: digits,
             exponent: parseInt(match[4] || "0", 10) - frac.length };
  }

  function increment(digits) {
    var out = digits.split("");
    for (var i = out.length - 1; i >= 0; i--) {
      if (out[i] === "9") {
        out[i] = "0";
      } else {
        out[i] = String(Number(out[i]) + 1);
        return out.join("");
      }
    }
    return "1" + out.join("");
  }

  function zeros(n) {
    return n > 0 ? new Array(n + 1).join("0") : "";
  }

  function roundDigits(digits, exponent, decimals) {
    if (exponent >= -decimals) {
      return (digits + zeros(exponent + decimals)).replace(/^0+/, "") || "0";
    }
    var cut = -decimals - exponent;
    var padded = digits.padStart(cut + 1, "0");
    var keep = padded.slice(0, padded.length - cut);
    var drop = padded.slice(padded.length - cut);
    var half = "5" + zeros(cut - 1);
    if (drop > half || (drop === half && Number(keep[keep.length - 1]) % 2 === 1)) {
      keep = increment(keep);
    }
    return keep.replace(/^0+/, "") || "0";
  }

  function placePoint(scaled, decimals) {
    var padded = scaled.padStart(decimals + 1, "0");
    if (decimals === 0) {
      return [padded, ""];
    }
    var whole = padded.slice(0, padded.length - decimals).replace(/^0+/, "") || "0";
    return [whole, padded.slice(padded.length - decimals)];
  }

  function roundHalfEven(value, decimals, shift) {
    var d = toDecimal(value);
    var scaled = roundDigits(d.digits, d.exponent + (shift || 0), decimals);
    var parts = placePoint(scaled, decimals);
    var text = decimals ? parts[0] + "." + parts[1] : parts[0];
    return d.negative && scaled !== "0" ? "-" + text : text;
  }

  // ── the formatter ────────────────────────────────────────────────────────

  function group(whole) {
    var head = whole.length % 3 || 3;
    var out = [whole.slice(0, head)];
    for (var i = head; i < whole.length; i += 3) {
      out.push(whole.slice(i, i + 3));
    }
    return out.join(",");
  }

  function magnitude(digits, exponent, decimals, compact) {
    var parts = placePoint(roundDigits(digits, exponent, decimals), decimals);
    var tier = 0;
    var fromDigits = compact === undefined || compact === null ? SUFFIX_FROM_DIGITS
                                                               : COMPACT_FROM_DIGITS;
    if (compact !== false && parts[0].length >= fromDigits) {
      tier = Math.min(Math.floor((parts[0].length - 1) / 3), SUFFIXES.length - 1);
    }
    if (tier === 0) {
      return group(parts[0]) + (parts[1] ? "." + parts[1] : "");
    }
    for (;;) {
      var m = placePoint(roundDigits(digits, exponent - 3 * tier, 1), 1);
      if (m[0].length > 3 && tier < SUFFIXES.length - 1) {
        tier += 1; // 999,960 is 1,000.0K, which is 1M
        continue;
      }
      return group(m[0]) + (m[1] === "0" ? "" : "." + m[1]) + SUFFIXES[tier];
    }
  }

  function formatNumber(value, unit, decimals, currency, signed, compact) {
    if (UNITS.indexOf(unit) < 0) {
      throw new OutcomeError("unit", "unknown unit " + String(unit));
    }
    if (typeof decimals !== "number" || !Number.isInteger(decimals) ||
        decimals < 0 || decimals > MAX_DECIMALS) {
      throw new OutcomeError("decimals", "decimals must be an integer 0-" + MAX_DECIMALS);
    }
    if (currency !== undefined && currency !== null &&
        (typeof currency !== "string" || !CURRENCY_CODE.test(currency))) {
      throw new OutcomeError("currency", String(currency) + " is not an ISO 4217 code");
    }
    if (compact !== undefined && compact !== null && typeof compact !== "boolean") {
      throw new OutcomeError("compact", "compact must be true, false or absent, not " +
        String(compact));
    }
    var d = toDecimal(value);
    var isZero = roundDigits(d.digits, d.exponent, decimals) === "0";
    var body;
    if (SCALED_UNITS.indexOf(unit) >= 0) {
      body = magnitude(d.digits, d.exponent, decimals, compact);
    } else {
      var parts = placePoint(roundDigits(d.digits, d.exponent, decimals), decimals);
      body = group(parts[0]) + (parts[1] ? "." + parts[1] : "");
    }

    if (unit === "currency" && currency !== undefined && currency !== null) {
      body = has(CURRENCY_SYMBOLS, currency) ? CURRENCY_SYMBOLS[currency] + body
                                             : currency + " " + body;
    } else if (unit === "percent") {
      body += "%";
    } else if (unit === "pp") {
      body += " pp";
    } else if (unit === "days") {
      body += body === "1" ? " day" : " days";
    }

    if (isZero) {
      return body;
    }
    if (d.negative) {
      return "-" + body;
    }
    return signed ? "+" + body : body;
  }

  function formatFigure(figure, opts) {
    return formatNumber(figure.value, figure.unit, figure.decimals, figure.currency,
                        Boolean(opts && opts.signed), figure.compact);
  }

  function renderText(text, figures) {
    var byId = {};
    if (Array.isArray(figures)) {
      for (var i = 0; i < figures.length; i++) {
        byId[figures[i].id] = figures[i];
      }
    } else {
      byId = figures;
    }
    return text.replace(PLACEHOLDER, function (_, id) {
      if (!has(byId, id)) {
        throw new OutcomeError("unresolved_figure", id + " is not in figures");
      }
      return formatFigure(byId[id]);
    });
  }

  // ── the scenario model (§2.2) ────────────────────────────────────────────

  function num(value, what) {
    if (!isFiniteNumber(value)) {
      throw new OutcomeError("value", what + " must be a finite number, got " + String(value));
    }
    return value;
  }

  function checkDriver(driver) {
    var id = driver.id;
    if (SUPPORTED_FORMS.indexOf(driver.form) < 0) {
      throw new OutcomeError("unsupported_form", "driver " + id + ": form " + driver.form +
        " is not computed in this version (supported: " + SUPPORTED_FORMS.join(", ") + ")");
    }
    var lo = num(driver.lo, "driver " + id + " lo");
    var hi = num(driver.hi, "driver " + id + " hi");
    var baseline = num(driver.baseline, "driver " + id + " baseline");
    num(driver.beta, "driver " + id + " beta");
    var se = num(driver.se, "driver " + id + " se");
    if (!(lo < hi)) {
      throw new OutcomeError("driver", "driver " + id + ": lo must be below hi");
    }
    if (!(lo <= baseline && baseline <= hi)) {
      throw new OutcomeError("driver", "driver " + id + ": baseline is outside [lo, hi]");
    }
    if (se < 0) {
      throw new OutcomeError("driver", "driver " + id + ": se must not be negative");
    }
    if (driver.form === "step" && (lo !== 0 || hi !== 1 || (baseline !== 0 && baseline !== 1))) {
      throw new OutcomeError("driver", "driver " + id + ": a step driver has lo 0, hi 1 and " +
        "a baseline of 0 (control) or 1 (treatment)");
    }
  }

  function howSure(y0, y, lo, hi, clamped) {
    var excludesNoChange = lo > y0 || hi < y0;
    var level;
    if (!excludesNoChange) {
      level = 2;
    } else if ((hi - lo) / (2 * Math.abs(y - y0)) < FAIRLY_SURE_R) {
      level = 0;
    } else {
      level = 1;
    }
    if (clamped) {
      level = Math.min(level + 1, 2);
    }
    return SURE_LEVELS[level];
  }

  function evaluate(whatif, values, opts) {
    var drivers = whatif.drivers || [];
    if (drivers.length === 0) {
      throw new OutcomeError("driver", "a what-if panel needs at least one driver");
    }
    var y0 = num(whatif.y0, "y0");
    var z = num(has(whatif, "z") ? whatif.z : Z_DEFAULT, "z");
    var q = opts && opts.conformalQ !== undefined && opts.conformalQ !== null
      ? opts.conformalQ : whatif.conformal_q;
    if (q !== undefined && q !== null) {
      q = num(q, "conformal_q");
    } else {
      q = null;
    }
    values = values || {};
    var known = {};
    var i;
    for (i = 0; i < drivers.length; i++) {
      known[drivers[i].id] = true;
    }
    var names = Object.keys(values);
    for (i = 0; i < names.length; i++) {
      if (!has(known, names[i])) {
        throw new OutcomeError("unknown_driver", names[i] + " is not a driver of this panel");
      }
    }

    var applied = {};
    var clamped = [];
    var y = y0;
    var variance = 0;
    for (i = 0; i < drivers.length; i++) {
      var driver = drivers[i];
      checkDriver(driver);
      var id = driver.id;
      var x0 = driver.baseline;
      var requested = num(has(values, id) ? values[id] : x0, "value for " + id);
      if (driver.form === "step" && requested !== 0 && requested !== 1) {
        throw new OutcomeError("step_level", id + " takes 0 (control) or 1 (treatment)");
      }
      var x = Math.min(Math.max(requested, driver.lo), driver.hi);
      if (x !== requested) {
        clamped.push({ driver: id, requested: requested, applied: x,
                       bound: x === driver.lo ? "lo" : "hi" });
      }
      applied[id] = x;
      y += driver.beta * (x - x0);
      var s = Math.abs(x - x0) * driver.se;
      variance += s * s;
    }

    var sigma = Math.sqrt(variance);
    var lo = y - z * sigma;
    var hi = y + z * sigma;
    if (q !== null && q > 0) {
      lo = Math.min(lo, y - q);
      hi = Math.max(hi, y + q);
    }
    return { applied: applied, clamped: clamped, sigma: sigma,
             result: { y: y, lo: lo, hi: hi, sure: howSure(y0, y, lo, hi, clamped.length > 0) } };
  }

  return {
    SPEC_VERSION: SPEC_VERSION,
    UNITS: UNITS,
    CURRENCY_SYMBOLS: CURRENCY_SYMBOLS,
    SURE_LEVELS: SURE_LEVELS,
    SURE_TEXT: SURE_TEXT,
    OutcomeError: OutcomeError,
    roundHalfEven: roundHalfEven,
    formatNumber: formatNumber,
    formatFigure: formatFigure,
    renderText: renderText,
    evaluate: evaluate,
    howSure: howSure
  };
});
