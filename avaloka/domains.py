"""Domain vocabulary — a closed set of dataset domains and what each one implies.

This is the understanding agent as 2.0 ships it: no model, no second agent.
``classify`` reads column *names* against per-domain cue patterns and returns
one domain from :data:`VOCABULARY`, with the hint set that domain carries
(typical units, targets, join keys, sanity checks the analysis should answer
or flag, and friendly first questions Avaloka can ask the user). The app
profiler consults a model only when this says ``generic`` or is unsure, and
maps the model's free text back onto the same vocabulary with :func:`coerce`.

Privacy: classification uses names only. ``sample_rows`` may confirm that a
matched column is numeric or text; no value is ever copied into the result.
Hint text is published code and must never quote customer content.

Standard library only: this module lives in the CLI package (S1.08).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers
import re
from typing import Any, Iterable, Mapping, Sequence

GENERIC = "generic"

# Below this the names name no domain (``generic``, with the leading candidate
# kept as ``tentative``) and the app asks the model. One distinct cue gives
# 0.57, two give 0.69 (see ``_confidence``).
CONFIDENT = 0.6


@dataclass(frozen=True)
class Domain:
    id: str
    label: str
    # Regexes matched against a normalised column name ("PatientId" -> "patient id").
    cues: tuple[str, ...]
    # Free-text fragments (lower-case) a model's answer may use for this domain.
    aliases: tuple[str, ...]
    units: tuple[str, ...]
    targets: tuple[str, ...]
    join_keys: tuple[str, ...]
    sanity_checks: tuple[str, ...]
    questions: tuple[str, ...]
    _compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)
    _alias_compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", tuple(re.compile(c) for c in self.cues))
        object.__setattr__(self, "_alias_compiled",
                           tuple(re.compile(r"(?<![a-z0-9])" + re.escape(a)) for a in self.aliases))

    def hints(self) -> dict[str, list[str]]:
        return {
            "units": list(self.units),
            "targets": list(self.targets),
            "join_keys": list(self.join_keys),
            "sanity_checks": list(self.sanity_checks),
        }


def _d(**kw: Any) -> Domain:
    for key in ("cues", "aliases", "units", "targets", "join_keys", "sanity_checks", "questions"):
        kw[key] = tuple(kw.get(key, ()))
    return Domain(**kw)


VOCABULARY: dict[str, Domain] = {d.id: d for d in (
    _d(
        id="healthcare", label="Healthcare",
        cues=(r"\bpatient", r"\bdiagnos", r"\bicd\b", r"\badmi(ssion|t)", r"\bdischarge",
              r"\bhospital", r"\bnpi\b", r"\b(procedure|cpt)\b", r"\b(medication|drug|dose|dosage)\b",
              r"\bbmi\b", r"\b(blood pressure|systolic|diastolic)\b", r"\b(glucose|cholesterol|hba1c)\b",
              r"\bencounter", r"\bclaims?\b", r"\b(los|length of stay)\b", r"\breadmi(ssion|t)",
              r"\b(mortality|survival|deceased)\b", r"\b(symptom|comorbid)"),
        aliases=("health", "medical", "clinical", "hospital", "patient", "claims", "pharma", "epidemiolog"),
        units=("mg/dL or mmol/L for labs (state which)", "mmHg for blood pressure", "days for length of stay",
               "years for age", "one currency for charges"),
        targets=("readmission", "mortality", "length_of_stay", "total_charges", "diagnosis"),
        join_keys=("patient_id", "encounter_id / admission_id", "provider_id / npi"),
        sanity_checks=(
            "Billing per condition should be compared on the same admission type (emergency vs elective).",
            "Counts are per encounter unless a patient key is deduplicated; one patient can have many visits.",
            "Lab values need their unit stated before comparison (glucose in mg/dL is not mmol/L).",
            "Protected health identifiers (names, record numbers, dates of birth) are not features.",
        ),
        questions=(
            "Is each row a patient or a visit?",
            "Should admission type (emergency vs elective) split every comparison?",
            "Which outcome matters most: readmission, length of stay, or cost?",
        ),
    ),
    _d(
        id="finance", label="Finance / banking",
        cues=(r"\baccount (id|no|number)\b", r"\b(transaction|txn|trans) ?(id|type|amount|date)?\b",
              r"\bbalance\b", r"\b(iban|swift|bic|routing)\b", r"\b(currency|ccy)\b",
              r"\b(loan|mortgage)\b", r"\bcredit (score|limit|card)\b", r"\b(interest rate|apr)\b",
              r"\bmerchant", r"\bfraud", r"\boverdraft", r"\bbranch\b", r"\b(debit|chargeback)\b",
              r"\b(kyc|aml)\b", r"\b(defaulted|loan status|delinquen)"),
        aliases=("finance", "fintech", "banking", "bank", "payment", "lending", "credit", "insurance"),
        units=("one currency per amount (or a currency column)", "basis points for rates",
               "percent for APR", "days past due"),
        targets=("is_fraud", "default / delinquency", "loan_status", "credit_risk"),
        join_keys=("account_id", "customer_id", "transaction_id"),
        sanity_checks=(
            "Amounts in mixed currencies are not summable; convert or split by currency first.",
            "Debits and credits carry a sign or a type column; a sum that ignores it is meaningless.",
            "Fraud and default are rare classes; accuracy is the wrong metric, use precision/recall or AUC-PR.",
            "Balances are snapshots and transactions are flows; do not average one as if it were the other.",
        ),
        questions=(
            "Is every amount in one currency, or is there a currency column?",
            "Are these transactions (flows) or balances (snapshots)?",
            "Which event are you after: fraud, default, or attrition?",
        ),
    ),
    _d(
        id="personal_finance", label="Personal finance",
        cues=(r"\bbudget", r"\b(expense|expenses|spending|spent)\b", r"\bsavings?\b",
              r"\b(income|salary|paycheck|payday)\b", r"\bpayee\b", r"\bsubscription",
              r"\bnet worth\b", r"\b(bills?|rent|utilities)\b", r"\bwallet", r"\bgrocer",
              r"\b(emi|installment|instalment)\b", r"\b(allowance|pocket money)\b"),
        aliases=("personal finance", "household finance", "budget", "expense", "spending"),
        units=("one currency per month", "percent of income"),
        targets=("monthly_spend", "savings_rate", "over_budget"),
        join_keys=("month", "account", "category"),
        sanity_checks=(
            "Monthly totals need a complete month; a partial current month understates spending.",
            "Transfers between your own accounts are not spending; exclude them before summing.",
            "Income and expenses must be signed consistently before a net figure is computed.",
        ),
        questions=(
            "Is this one person's ledger or several accounts combined?",
            "Should transfers between accounts be excluded from spending?",
            "Do you want month-by-month trends or a category breakdown?",
        ),
    ),
    _d(
        id="stock_market", label="Stock market",
        cues=(r"\b(ticker|symbol)\b", r"\bopen\b", r"\bhigh\b", r"\blow\b", r"\b(adj(usted)? )?close\b",
              r"\bvolume\b", r"\bdividend", r"\bmarket cap", r"\bp ?e ratio\b", r"\beps\b", r"\bvwap\b",
              r"\b(log )?returns?\b", r"\b(bid|ask)\b", r"\bexchange\b", r"\b(split|volatility|beta)\b"),
        aliases=("stock", "equit", "trading", "market data", "securities", "forex", "crypto", "ticker"),
        units=("price in the quote currency", "volume in shares", "returns in percent or log units"),
        targets=("next_close", "return", "direction", "volatility"),
        join_keys=("ticker", "date", "exchange"),
        sanity_checks=(
            "Prices must be split- and dividend-adjusted before returns are computed across an adjustment date.",
            "Trading days are not calendar days; weekend and holiday gaps are not missing data.",
            "Predicting tomorrow's close from today's is leakage-prone; evaluate on returns with a time-ordered split.",
        ),
        questions=(
            "Daily, intraday, or monthly bars?",
            "Is the price already adjusted for splits and dividends?",
            "Are you after a direction call, a volatility estimate, or a portfolio view?",
        ),
    ),
    _d(
        id="retail", label="Retail / e-commerce",
        cues=(r"\bsku\b", r"\bproduct ?(id|code|name|category)?\b", r"\border ?(id|date|number|no)\b",
              r"\b(quantity|qty)\b", r"\bunit price\b", r"\b(discount|coupon|promo)", r"\b(cart|basket|checkout)\b",
              r"\b(store|shop|outlet) ?(id|name)?\b", r"\b(returned|refund)", r"\b(rating|review)",
              r"\b(sales|revenue)\b", r"\b(brand|catalog)"),
        aliases=("retail", "e-commerce", "ecommerce", "commerce", "shopping", "orders", "sales", "merchandis"),
        units=("one currency per price", "units per order line", "days to return"),
        targets=("sales", "revenue", "demand", "return_flag", "basket_size"),
        join_keys=("order_id", "sku / product_id", "customer_id", "store_id"),
        sanity_checks=(
            "Revenue per order is quantity x unit price minus discount; summing unit price alone overstates it.",
            "Returns and refunds are negative sales; decide whether gross or net sales is the question.",
            "Seasonality (weekends, holidays, promotions) dominates daily sales; compare like periods.",
        ),
        questions=(
            "Is each row an order, an order line, or a product?",
            "Should sales be gross or net of returns?",
            "Is there a promotion calendar you want compared against?",
        ),
    ),
    _d(
        id="telecom", label="Telecom / churn",
        cues=(r"\bchurn", r"\btenure\b", r"\bcontract\b", r"\bmonthly( charges?)?\b", r"\btotal charges?\b",
              r"\b(phone service|multiple lines)\b", r"\b(internet service|fiber|dsl)\b", r"\bstreaming",
              r"\b(tech support|online (security|backup))\b", r"\b(paperless|payment method)\b",
              r"\b(data (usage|plan)|minutes|sms|call (count|duration))\b", r"\bdevice protection\b",
              r"\b(senior citizen|dependents)\b", r"\b(tickets?|complaints?)\b", r"\b(plan|tariff)\b"),
        aliases=("telecom", "telco", "churn", "subscription", "mobile operator", "carrier"),
        units=("one currency per month for charges", "months for tenure", "GB for data, minutes for calls"),
        targets=("churn", "tenure", "monthly_charges", "upsell"),
        join_keys=("customer_id", "account_id", "phone_number"),
        sanity_checks=(
            "Churn is an event in time: the tenure of customers still active is censored, not final.",
            "Total charges is roughly tenure x monthly charges; keeping both leaks the target through arithmetic.",
            "Churn rates must be compared within contract type; month-to-month customers churn more by construction.",
        ),
        questions=(
            "Is each row a customer or a billing line?",
            "Over what window is churn defined (30, 60, 90 days)?",
            "Do you want to predict churn or explain it?",
        ),
    ),
    _d(
        id="genomics", label="Genomics",
        cues=(r"\bgene( id| name| symbol)?\b", r"\b(chrom(osome)?|chr)\b", r"\b(variant|snp|rsid|allele|genotype|mutation)\b",
              r"\b(ref|alt)\b", r"\b(sequence|read (count|depth)|coverage)\b", r"\b(expression|tpm|fpkm|rpkm)\b",
              r"\b(fdr|log2? ?fold|fold change)\b", r"\b(phenotype|trait)\b", r"\bstrand\b",
              r"\b(transcript|exon|intron|protein|peptide)\b", r"\bsample id\b", r"\b(tumou?r|normal) ?(sample)?\b"),
        aliases=("genomic", "genetic", "bioinformatic", "sequencing", "biolog", "proteomic", "omics"),
        units=("TPM / FPKM for expression", "base pairs for positions", "log2 fold change"),
        targets=("phenotype", "expression", "variant_effect", "case_control"),
        join_keys=("gene_id", "sample_id", "variant (chrom + pos + ref + alt)"),
        sanity_checks=(
            "Genome build (GRCh37 vs GRCh38) must match before positions are joined.",
            "Expression counts need normalisation (TPM or size factors) before samples are compared.",
            "Thousands of genes against few samples: p-values need multiple-testing correction (FDR).",
        ),
        questions=(
            "Is each row a gene, a variant, or a sample?",
            "Which genome build are the coordinates on?",
            "Are these raw counts or already normalised?",
        ),
    ),
    _d(
        id="robotics", label="Robotics",
        cues=(r"\bepisode", r"\bframe( id| index| idx)?\b", r"\b(time ?step|step)\b", r"\bjoint",
              r"\b(actions?|observations?|obs)\b", r"\breward", r"\bgripper", r"\b(end effector|eef|pose|quaternion|yaw|pitch|roll)\b",
              r"\b(lidar|imu|odom(etry)?|accel|gyro)", r"\b(velocity|angular)", r"\b(camera|rgb|depth)\b",
              r"\b(success|done|terminal)\b", r"\bpolicy\b", r"\b(torque|servo|motor)"),
        aliases=("robot", "control", "autonomous", "reinforcement", "trajector", "manipulat", "lerobot", "rlds"),
        units=("radians or degrees for angles (state which)", "m/s for velocity", "N·m for torque",
               "Hz for the control rate"),
        targets=("next action", "reward", "success"),
        join_keys=("episode_id", "frame_index", "timestamp"),
        sanity_checks=(
            "Episode/frame ordering must be monotonic; a shuffle before windowing destroys the trajectory.",
            "Angles wrap at +/-pi; averaging raw angles near the boundary is wrong.",
            "Sensor streams at different rates need resampling to one clock before they are joined.",
            "Success rate is per episode, not per frame; frame-level accuracy overstates it.",
        ),
        questions=(
            "Is each row a frame, a step, or an episode?",
            "What is the control frequency, and are all sensors on the same clock?",
            "Which signal is the outcome: reward, success, or a tracking error?",
        ),
    ),
    _d(
        id="astronomy", label="Astronomy",
        cues=(r"\b(ra|right ascension)\b", r"\b(dec|declination)\b", r"\bredshift", r"\b(mag|magnitude)\b",
              r"\bflux", r"\bparallax", r"\b(proper motion|pmra|pmdec)\b", r"\b(spectral|spec type)",
              r"\b(galaxy|star|quasar|qso)\b", r"\b(ugriz|[ugriz] band|band)\b", r"\b(luminosity|teff)\b",
              r"\b(transit|light ?curve|period)\b", r"\b(mjd|jd|epoch)\b", r"\bobj ?id\b", r"\b(telescope|survey|plate|fiber ?id)\b"),
        aliases=("astro", "sky survey", "cosmolog", "telescope", "celestial", "exoplanet"),
        units=("degrees for RA/Dec", "magnitudes (logarithmic, smaller is brighter)", "Jansky for flux",
               "parsec / milliarcsecond for parallax", "MJD for epochs"),
        targets=("class (star / galaxy / quasar)", "redshift", "period"),
        join_keys=("obj_id / source_id", "(ra, dec) within a sky tolerance", "epoch"),
        sanity_checks=(
            "Magnitudes are logarithmic and inverted: a smaller number is brighter; do not average them as if linear.",
            "Positional joins need a sky-tolerance match on (RA, Dec), not an exact equality.",
            "Flux-limited samples are biased toward bright objects; a trend with distance may be selection.",
        ),
        questions=(
            "Is each row a source, an observation, or an epoch?",
            "Which photometric bands are these magnitudes in?",
            "Are you classifying sources or fitting a relation?",
        ),
    ),
    _d(
        id="manufacturing", label="Manufacturing / IoT",
        cues=(r"\b(machine|equipment|asset) ?(id|type)?\b", r"\bsensor", r"\b(vibration|rpm|rotational speed)",
              r"\b(temperature|temp|pressure|humidity)\b", r"\b(failure|fault|breakdown|downtime)",
              r"\b(maintenance|rul|remaining useful)", r"\b(cycles?|batch|lot) ?(id|no)?\b",
              r"\b(defects?|scrap|yield)\b", r"\b(line|shift|plant) ?(id)?\b", r"\btool wear\b",
              r"\b(setting \d|air temp|process temp)", r"\boperator", r"\b(reading|measurement)"),
        aliases=("manufactur", "iot", "industrial", "predictive maintenance", "sensor", "factory", "telemetry"),
        units=("°C or K for temperature (state which)", "kPa or bar for pressure", "RPM", "mm/s for vibration",
               "cycles for life"),
        targets=("failure", "remaining_useful_life", "defect", "downtime"),
        join_keys=("machine_id / unit_id", "timestamp", "batch / lot"),
        sanity_checks=(
            "Failures are rare and time-ordered; a random split leaks the future and inflates recall.",
            "Sensor readings need their unit (°C vs K, bar vs kPa) confirmed before thresholds are applied.",
            "Remaining useful life is per unit from its own history; do not pool cycles across machines without a unit key.",
        ),
        questions=(
            "Is each row a sensor reading, a cycle, or a machine?",
            "What counts as a failure event, and how far ahead do you need to see it?",
            "Are readings on a regular interval?",
        ),
    ),
    _d(
        id="hr", label="HR / people",
        cues=(r"\b(employee|emp|staff) ?(id|no|number|name)?\b", r"\b(salary|wage|compensation|pay(rate)?)\b",
              r"\b(department|dept)\b", r"\bjob ?(title|role|level|satisfaction)\b", r"\b(attrition|resign|terminat)",
              r"\b(hire|hiring|joining) date\b", r"\byears (at|in|since|with)\b", r"\b(performance|appraisal) ?(rating|score)?\b",
              r"\bpromot", r"\bover ?time\b", r"\b(engagement|work life balance)\b", r"\b(manager|supervisor)\b",
              r"\btraining (hours|times)", r"\b(absence|leave|pto)\b", r"\b(business travel|distance from home)\b"),
        aliases=("hr", "human resources", "people analytics", "employee", "workforce", "attrition", "talent"),
        units=("one currency per year for pay", "years for tenure", "hours for overtime and training"),
        targets=("attrition", "performance_rating", "promotion", "salary"),
        join_keys=("employee_id", "department", "manager_id"),
        sanity_checks=(
            "Attrition compares leavers against current staff; the current staff's tenure is censored.",
            "Pay comparisons must control for level and role before a gap is claimed.",
            "Protected attributes (gender, age, ethnicity) must not drive a hiring or pay model.",
        ),
        questions=(
            "Is each row an employee or an employment event?",
            "Which outcome matters: attrition, performance, or pay equity?",
            "Should comparisons be within department or across the company?",
        ),
    ),
    _d(
        id="education", label="Education",
        cues=(r"\b(student|pupil|learner) ?(id|name)?\b", r"\b(gpa|grades?|marks?|exam|test score)\b",
              r"\b(course|subject|module) ?(id|name)?\b", r"\b(teacher|instructor|school) ?(id|name)?\b",
              r"\b(attendance|absences)\b", r"\b(semester|term)\b", r"\benrol", r"\b(passed|failed|dropout)\b",
              r"\b(study ?(time|hours)|homework)\b", r"\bparent(al)? ?(education|job|status)",
              r"\b(scholarship|tuition)\b", r"\b(quiz|assignment)\b", r"\b(math|reading|writing) score\b"),
        aliases=("education", "student", "school", "academic", "university", "edtech", "exam"),
        units=("points or percent for scores (state the scale)", "hours per week for study time", "credits"),
        targets=("final_grade", "pass_fail", "dropout", "score"),
        join_keys=("student_id", "course_id", "school_id", "term"),
        sanity_checks=(
            "Scores on different scales (letter, percent, points) must be normalised before averaging.",
            "Students nest in classes and schools; a pooled regression overstates significance.",
            "Prior-term grades predict final grades by construction; say whether that is the question.",
        ),
        questions=(
            "Is each row a student, a student-course, or an assessment?",
            "Which outcome: final grade, pass/fail, or dropout?",
            "Are all scores on one scale?",
        ),
    ),
    _d(
        id="logistics", label="Logistics",
        cues=(r"\b(shipment|consignment|parcel|package) ?(id|no)?\b", r"\b(origin|destination|route|lane)\b",
              r"\b(carrier|vehicle|truck|fleet|driver)", r"\b(delivery|delivered|eta|transit time|lead time)",
              r"\b(warehouse|depot|hub|dock)", r"\b(pallets?|container)\b", r"\b(distance|km|miles|mileage)\b",
              r"\b(on time|late|delay)", r"\btracking", r"\b(freight|shipping) ?(cost|mode)?\b",
              r"\b(inventory|stock level|reorder)", r"\b(supplier|vendor|purchase order|po number)\b",
              r"\b(pickup|dispatch)"),
        aliases=("logistic", "supply chain", "shipping", "transport", "delivery", "fleet", "freight", "warehouse"),
        units=("km or miles (state which)", "kg or lb for weight", "hours or days for lead time",
               "one currency per shipment"),
        targets=("delay", "transit_time", "on_time", "cost_per_shipment"),
        join_keys=("shipment_id", "order_id", "route / lane", "carrier_id"),
        sanity_checks=(
            "On-time rate depends on the promised date; recompute it from promise vs delivery, not a status flag alone.",
            "Distance and weight need one unit each (km vs miles, kg vs lb) before a cost per unit is computed.",
            "Transit time across time zones needs timestamps in UTC or a zone column.",
        ),
        questions=(
            "Is each row a shipment, a leg, or a scan event?",
            "What counts as late: past the promised date or past the planned ETA?",
            "Do you want to explain delays or predict them?",
        ),
    ),
    _d(
        id="marketing", label="Marketing",
        cues=(r"\bcampaign", r"\bimpressions?\b", r"\b(clicks?|ctr)\b", r"\b(conversions?|cvr|converted)\b",
              r"\b(ad spend|cost per|cpc|cpm|cpa)\b", r"\b(channel|medium|utm)", r"\b(leads?|mql|sql)\b",
              r"\b(open rate|unsubscribe|bounce)", r"\b(audience|segment)\b", r"\b(roas|roi)\b",
              r"\b(ad group|creative|keyword)", r"\b(sessions?|page ?views|visits)\b", r"\b(attribution|touch ?point)",
              r"\b(email|newsletter) ?(sent|opened)?\b"),
        aliases=("marketing", "advertis", "ads", "campaign", "attribution", "crm", "growth", "media analytics"),
        units=("one currency for spend and revenue", "counts for impressions and clicks",
               "percent for CTR and CVR", "ROAS as a ratio"),
        targets=("conversion", "ctr", "roas", "lead_score"),
        join_keys=("campaign_id", "channel", "date", "user_id / cookie_id"),
        sanity_checks=(
            "Rates (CTR, CVR) must be recomputed from summed clicks and impressions, not averaged across rows.",
            "The attribution window and model (last-touch vs multi-touch) change the answer; name the one used.",
            "Spend and revenue must be in the same currency and period before ROAS is computed.",
        ),
        questions=(
            "Is each row a campaign-day, an impression, or a user?",
            "Which attribution model do you report on?",
            "Is the question about acquisition cost or return on spend?",
        ),
    ),
    _d(
        id="public_sector", label="Public sector",
        cues=(r"\b(census|population)\b", r"\b(county|district|municipality|ward|constituency|precinct|borough)\b",
              r"\b(fips|geoid|nuts|lsoa|postcode|zip ?code)\b", r"\b(appropriation|expenditure|grant)",
              r"\b(permit|licen[cs]e|inspection|violation|citation)", r"\b(crime|offen[cs]e|arrest|incident) ?(type|date|count)?\b",
              r"\b(voter|turnout|election|ballot|party)\b", r"\b(unemployment|poverty|median income|households?)\b",
              r"\b(agency|ministry|program) ?(name|id)?\b", r"\b(beneficiar|enrollees|recipients)",
              r"\b(complaint type|service request)\b", r"\bfiscal year\b"),
        aliases=("public sector", "government", "civic", "census", "municipal", "policy", "open data", "demograph"),
        units=("counts per 1,000 or 100,000 residents", "one currency per fiscal year", "percent"),
        targets=("rate per capita", "turnout", "response_time", "allocation"),
        join_keys=("geography code (FIPS / GEOID / postcode)", "fiscal year", "agency"),
        sanity_checks=(
            "Counts across areas need a per-capita or per-household denominator before comparison.",
            "Fiscal years and calendar years differ; say which one every total is on.",
            "Boundary codes change between censuses; join on the vintage, not just the name.",
            "Small-area counts can re-identify people; suppress or aggregate before publishing.",
        ),
        questions=(
            "Is each row a place, a person, or an event?",
            "What denominator should rates use: population, households, or area?",
            "Which fiscal or calendar year do the figures cover?",
        ),
    ),
    _d(
        id=GENERIC, label="Generic",
        cues=(), aliases=(),
        units=("state the unit of every numeric column before comparing them",),
        targets=("the column the question names",),
        join_keys=("any column with one value per row",),
        sanity_checks=(
            "Say what one row stands for before counting anything.",
            "Identifier columns are not features and not insights.",
            "A column that is arithmetic on another (total = rate x count) leaks the target.",
        ),
        questions=(
            "What does one row stand for?",
            "Which column is the outcome you care about?",
            "Is there a time column that orders the rows?",
        ),
    ),
)}

DOMAIN_IDS: tuple[str, ...] = tuple(VOCABULARY)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalise(name: Any) -> str:
    """``"PatientId"`` -> ``"patient id"``; ``"adj_close"`` -> ``"adj close"``."""
    text = _CAMEL.sub(" ", str(name))
    return _NON_WORD.sub(" ", text.lower()).strip()


def _confidence(best: int, runner_up: int) -> float:
    if best <= 0:
        return 0.0
    conf = min(0.95, 0.45 + 0.12 * best)
    # A tie with another domain is a coin toss; say so.
    return round(conf * 0.5, 2) if runner_up == best else round(conf, 2)


def _sample_type_check(columns: Sequence[str], sample_rows: Iterable[Mapping[str, Any]] | None) -> dict[str, list[str]]:
    """Names (only) of matched columns whose sample values are numeric or text.

    The values are inspected for their type and discarded; nothing from a row
    reaches the result.
    """
    numeric: list[str] = []
    text: list[str] = []
    rows = [r for r in (sample_rows or []) if isinstance(r, Mapping)]
    if not rows:
        return {"numeric": numeric, "text": text}
    for col in columns:
        seen = [r[col] for r in rows if col in r and r[col] is not None]
        if not seen:
            continue
        if all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in seen):
            numeric.append(col)
        elif all(isinstance(v, str) for v in seen):
            text.append(col)
    return {"numeric": numeric, "text": text}


def classify(columns: Sequence[Any], sample_rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Pick one domain from :data:`VOCABULARY` by column names, deterministically.

    Each domain scores the number of *distinct* cues its column names hit, so
    ``score_1 .. score_9`` counts once. Highest score wins; a tie halves the
    confidence. A read below :data:`CONFIDENT` (one cue, or a tie) names no
    domain: it is ``generic`` with confidence 0.0, and the leading candidate is
    kept under ``tentative`` for the app's model check. One ``Ticket`` column
    does not make the Titanic passenger list a telecom dataset. No hit at all
    is ``generic`` with no ``tentative``. ``sample_rows`` only feeds
    ``type_check`` (column names grouped by the type of their values) and never
    changes the domain.
    """
    names = [str(c) for c in (columns or []) if str(c).strip()]
    normalised = [(n, normalise(n)) for n in names]

    scores: dict[str, int] = {}
    matched: dict[str, list[dict[str, str]]] = {}
    for dom in VOCABULARY.values():
        hits: list[dict[str, str]] = []
        cues_hit: set[str] = set()
        for original, norm in normalised:
            for cue, pattern in zip(dom.cues, dom._compiled):
                if pattern.search(norm):
                    hits.append({"column": original, "cue": cue})
                    cues_hit.add(cue)
                    break
        scores[dom.id] = len(cues_hit)
        matched[dom.id] = hits

    ranked = sorted((d for d in DOMAIN_IDS if d != GENERIC), key=lambda d: (-scores[d], DOMAIN_IDS.index(d)))
    best_id, best = ranked[0], scores[ranked[0]]
    runner_up = scores[ranked[1]]
    confidence = _confidence(best, runner_up)
    cols = ", ".join(sorted({h['column'] for h in matched[best_id]})[:4])
    tentative = None
    if best == 0:
        reason = "no column name matched a domain cue; treating the data as generic"
    elif runner_up == best:
        reason = (f"{best} cue(s) matched for both {best_id} and {ranked[1]}; "
                  f"a tie names no domain, treating the data as generic")
    elif confidence < CONFIDENT:
        reason = (f"only {best} cue matched {best_id} ({cols}); "
                  f"not enough to name a domain, treating the data as generic")
    else:
        reason = f"{best} distinct cue(s) matched column names ({cols})"
    if best and confidence < CONFIDENT:
        tentative = {"domain": best_id, "confidence": confidence}
        best_id, confidence = GENERIC, 0.0
    elif not best:
        best_id = GENERIC
    dom = VOCABULARY[best_id]
    return {
        "domain": best_id,
        "label": dom.label,
        "confidence": confidence,
        "reason": reason,
        "matched_cues": matched.get(best_id, []),
        "candidates": [{"domain": d, "score": scores[d]} for d in ranked[:3] if scores[d] > 0],
        "tentative": tentative,
        "hints": dom.hints(),
        "questions": list(dom.questions),
        "type_check": _sample_type_check([h["column"] for h in matched.get(best_id, [])], sample_rows),
    }


def coerce(text: Any) -> str:
    """Map a free-text domain name (a model's answer) onto the vocabulary.

    Exact id or label first, then the earliest alias occurrence in the text;
    anything unrecognised is ``generic``. ``"FinTech/Payments"`` -> ``finance``,
    ``"Media/YouTube Analytics"`` -> ``generic``. An alias matches at the start
    of a word only, so ``hr`` is not found in "Throughput" nor ``ads`` in
    "Downloads".
    """
    if not isinstance(text, str) or not text.strip():
        return GENERIC
    low = text.strip().lower()
    for dom in VOCABULARY.values():
        if low in (dom.id, dom.label.lower()) or low.replace(" ", "_") == dom.id:
            return dom.id
    best: tuple[int, int, str] | None = None
    for order, dom in enumerate(VOCABULARY.values()):
        for pattern in dom._alias_compiled:
            hit = pattern.search(low)
            if hit and (best is None or (hit.start(), order) < best[:2]):
                best = (hit.start(), order, dom.id)
    return best[2] if best else GENERIC


def hints_for(domain_id: str) -> dict[str, Any]:
    """The hint set and questions of one domain (``generic`` for an unknown id)."""
    dom = VOCABULARY.get(domain_id, VOCABULARY[GENERIC])
    return {"domain": dom.id, "label": dom.label, "hints": dom.hints(), "questions": list(dom.questions)}
