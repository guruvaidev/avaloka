"""Derived categories: a breakdown whose categories the analysis made up.

On rc14 the recipes chat answered "what kinds of recipes dominate?" over German
recipes. The data has no category column, so the coder invented one: it
matched ENGLISH keywords (salad, soup, cake, sauce, ...) against the German
``Name`` and ``Ingredients`` and called every row that matched none "other".
The reply said "Other recipes dominate ... 67.6%", "Sauce" came second, the
leader was "roughly five times" the next and "Sushi" the smallest. Every figure
reconciled with the result, so the claim verifier passed all five; and every
one was an artefact of the keyword list, not a fact about the recipes.

A category breakdown is only as good as the rule that assigned the
categories. When the analysis assigned them by keyword matching, this module
finds that rule in the code that ran (its AST, never the reply), recomputes
how many rows the keywords match on the data the run read, and checks that the
keywords are in the language of the text they were matched against. Then:

* the reply carries a method line: "categories assigned by keyword matching on
  <column>; N% unmatched." N is computed here, deterministically;
* the breakdown is withheld, with the reason stated, when more than half the
  rows matched no keyword, or when the keywords' language is not the data's.

This rule (D) is the coordinator's default; the owner may overrule it.

What is NOT a derived category: a groupby on a column the data already has
(nothing is matched), a single ``str.contains`` filter (one keyword set, no
labels), a yes/no flag from one keyword set, and a value recode such as
``.map({'M': 'Male'})`` (equality, not substring matching).
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

logger = logging.getLogger(__name__)

#: Above this share of unmatched rows the breakdown mostly measures what the
#: keywords missed, and is withheld.
MAX_UNMATCHED = 0.5
#: Rows read per classified column to decide the data's language.
LANGUAGE_SAMPLE_ROWS = 5000

# --------------------------------------------------------------------------- #
# Language: a stopword ratio for the data, a word list for the keywords
# --------------------------------------------------------------------------- #
# Only words of three letters or more count, and only words that belong to one
# language of the set: "el" is Spanish and also German recipes' EL
# (Esslöffel), "de" is Dutch, French, Spanish and Portuguese.

_STOPWORDS = {
    "en": """the and of to with for are was this that these those its but not have has had you your
             our they their which what when where who will would can could should into over under about
             after before than then there here all some any each more most very just also only such how
             why from been were being""",
    "de": """der die das und mit von den dem des ein eine einer einem einen ist sind war nicht auch auf
             für bei aus nach oder aber wie wird werden sich ich sie wir ihr noch nur dann wenn vom zum
             zur etwas sehr alle kann hat haben dass über unter ohne mehr""",
    "fr": """les des une est sont pour avec dans sur par pas que qui aux cette ces elle ils nous vous
             mais plus très sans sous être avoir son ses leur comme aussi tout tous""",
    "es": """los las del una son para con por que como más pero sus este esta estos muy sin sobre
             también hay ser fue entre cuando todo todos""",
    "it": """gli della dei delle una sono per con che non alla nel nella sul come più anche questo
             questa molto senza tra fra suo sua""",
    "nl": """het een van met voor zijn niet ook als maar bij uit naar dat die deze dit wordt worden nog
             aan tot wij hij zij over onder zonder meer heel""",
    "pt": """uma são para com por que não mais mas como seu sua muito sem sobre também entre este esta
             dos das nos nas""",
}

#: Everyday words, for keywords: a keyword is a noun ("salad"), never a stopword.
_WORDS = {
    "en": """salad soup cake cookie bread sandwich stew pie dumpling smoothie chicken beef pork lamb fish
             shrimp noodle egg cheese potato vegetable fruit dessert drink juice breakfast lunch dinner
             snack sweet spicy vegetarian meat seafood chocolate cream milk roast grilled baked fried
             chowder broth lettuce spinach green pudding brownie cocktail milkshake apple banana bean
             mushroom onion garlic tomato corn pepper honey sugar flour casserole muffin pancake waffle
             cupcake biscuit wrap dip refund delivery shipping price payment billing complaint account
             order cancel late damaged broken quality service support login password error slow fast
             good bad great excellent poor happy angry love hate cheap expensive discount free new old
             small large big family kids children women men school work travel car house home health
             doctor hospital pain fever cough bike train bus flight hotel room cleaning food water coffee
             tea beer weather rain snow sun""",
    "de": """salat suppe kuchen brot brötchen eintopf auflauf torte kekse plätzchen nudeln nudel
             kartoffel kartoffeln fleisch hähnchen huhn rind schwein fisch gemüse obst nachtisch getränk
             frühstück mittagessen abendessen käse eier sahne milch zucker mehl apfel schokolade scharf
             süß vegetarisch pfanne gebraten gebacken braten soße sosse knödel klöße wurst schinken speck
             reis lieferung rückgabe preis zahlung rechnung beschwerde konto bestellung kaputt schnell
             langsam gut schlecht kinder familie arzt krankenhaus schmerzen wetter regen schnee auto haus
             zug flug wasser kaffee tee bier wein""",
    "fr": """salade soupe gâteau gateau pain ragoût tarte poulet boeuf bœuf porc agneau poisson crevette
             riz nouilles oeuf œuf fromage pomme légumes fruits boisson déjeuner dîner sucré épicé viande
             chocolat crème beurre lait livraison remboursement prix paiement facture plainte commande
             retour cassé rapide lent bon mauvais enfants famille médecin hôpital douleur voiture maison
             vol eau café thé bière vin pluie neige""",
    "es": """ensalada sopa pastel tarta pan guiso galleta pollo carne cerdo cordero pescado camarón
             gambas arroz fideos huevo huevos queso papa patata verduras fruta postre bebida desayuno
             almuerzo cena dulce picante crema mantequilla leche envío reembolso precio pago factura queja
             pedido devolución roto rápido lento bueno malo niños familia médico dolor coche casa tren
             vuelo agua cerveza vino lluvia nieve""",
    "it": """insalata zuppa torta pane stufato biscotto biscotti pollo manzo maiale agnello pesce gamberi
             riso uovo uova formaggio patate verdure frutta dolce bevanda colazione pranzo cena piccante
             carne panna burro latte consegna rimborso prezzo pagamento fattura reclamo ordine rotto
             veloce buono cattivo bambini famiglia medico ospedale dolore macchina treno volo acqua caffè
             birra vino pioggia neve""",
    "nl": """soep taart brood stoofpot koekje koekjes kip rundvlees varkensvlees vis garnalen rijst
             noedels eieren kaas aardappel aardappelen groente groenten toetje drankje ontbijt avondeten
             zoet pittig vlees chocola room boter melk levering bezorging terugbetaling prijs betaling
             factuur klacht bestelling kapot snel langzaam goed slecht kinderen dokter ziekenhuis pijn
             huis trein vlucht koffie thee wijn sneeuw""",
    "pt": """salada sopa bolo pão ensopado biscoito frango carne porco cordeiro peixe camarão arroz
             macarrão ovo ovos queijo batata fruta sobremesa bebida almoço jantar doce picante manteiga
             leite entrega reembolso preço pagamento fatura reclamação pedido devolução quebrado rápido
             lento bom ruim crianças família médico dor carro casa trem voo água cerveja vinho chuva""",
}

LANGUAGE_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish",
                  "it": "Italian", "nl": "Dutch", "pt": "Portuguese"}


def _exclusive(table: Mapping[str, str]) -> Dict[str, str]:
    """word -> language, for words of three letters or more in one language only."""
    seen: Dict[str, Set[str]] = {}
    for lang, words in table.items():
        for word in words.split():
            if len(word) >= 3:
                seen.setdefault(word, set()).add(lang)
    return {w: next(iter(langs)) for w, langs in seen.items() if len(langs) == 1}


_STOP_LANG = _exclusive(_STOPWORDS)
_WORD_LANG = _exclusive({lang: _STOPWORDS[lang] + " " + _WORDS[lang] for lang in _STOPWORDS})
_TOKEN = re.compile(r"[^\W\d_]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def text_language(texts: Iterable[Any], min_tokens: int = 50) -> Optional[str]:
    """The dominant language of *texts* by stopword ratio, or None.

    A language is dominant when its stopwords are at least 1% of the tokens
    and at least twice the runner-up's. Text with too few words (codes, IDs,
    short labels) has no language here.
    """
    hits: Dict[str, int] = {}
    total = 0
    for text in texts:
        for token in _tokens(str(text)):
            total += 1
            lang = _STOP_LANG.get(token)
            if lang:
                hits[lang] = hits.get(lang, 0) + 1
    if total < min_tokens or not hits:
        return None
    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    top, n = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if n / total < 0.01 or n < 2 * second:
        return None
    return top


def _word_language(word: str) -> Optional[str]:
    lang = _WORD_LANG.get(word)
    if lang is None and len(word) > 3 and word.endswith("s"):
        lang = _WORD_LANG.get(word[:-1])      # cookies, eggs
    return lang


def keyword_language(keywords: Sequence[str], attested: Optional[Set[str]] = None) -> Optional[str]:
    """The language of *keywords*, by majority over the ones a word list knows.

    A keyword the data itself uses as a word (pizza, curry in German recipes)
    is a loanword there and votes for nothing. The language needs at least two
    votes, a quarter of the keywords, and two thirds of the votes.
    """
    votes: Dict[str, int] = {}
    for keyword in keywords:
        words = _tokens(re.sub(r"\\[bBwsdW]", " ", keyword))
        langs = [l for l in (_word_language(w) for w in words if not (attested and w in attested)) if l]
        if langs and all(l == langs[0] for l in langs):
            votes[langs[0]] = votes.get(langs[0], 0) + 1
    if not votes:
        return None
    lang, n = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    if n < max(2, len(keywords) / 4) or n < 2 * sum(votes.values()) / 3:
        return None
    return lang


# --------------------------------------------------------------------------- #
# Detection: keyword classification in the code that ran
# --------------------------------------------------------------------------- #

@dataclass
class KeywordClassification:
    """A rule that assigns categories by matching keywords against text."""
    patterns: List[str]                         # regex alternatives, one per keyword
    columns: List[str]                          # the text it matched, as the code names it
    labels: List[str] = field(default_factory=list)
    label_column: Optional[str] = None
    default_label: Optional[str] = None
    case_insensitive: bool = False
    #: Column -> the columns it was derived from (``df['name_lc'] = df['Name'].str.lower()``).
    derived: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)   # as written, for the language check


_CONTAINS = frozenset({"contains", "match", "fullmatch", "findall", "count"})
_RE_CALLS = frozenset({"search", "match", "fullmatch", "findall"})
_APPLY = frozenset({"apply", "map", "transform", "applymap"})
_GROUP_CALLS = frozenset({"groupby", "value_counts", "pivot_table", "crosstab", "pivot", "agg", "count"})
_CASE_FOLD = frozenset({"lower", "casefold", "upper"})


class _Code:
    """What the classifier detection needs to know about one parsed script."""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.parents: Dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        #: name -> RHS node, for every ``name = <expr>``; the last one wins.
        self.assigned: Dict[str, ast.AST] = {}
        #: column -> RHS node, for every ``frame['col'] = <expr>`` (and ``.loc[..., 'col']``).
        self.column_exprs: Dict[str, List[ast.AST]] = {}
        #: column -> RHS node of a whole-column ``frame['col'] = <expr>`` only.
        self.column_sets: Dict[str, List[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    self.assigned[target.id] = node.value
                key = _column_key(target)
                if key is not None:
                    self.column_exprs.setdefault(key, []).append(node.value)
                if _subscript_key(target) is not None:
                    self.column_sets.setdefault(key, []).append(node.value)
        self.functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # -- literal values -------------------------------------------------- #
    def value(self, node: Optional[ast.AST], env: Mapping[str, list], depth: int = 0) -> list:
        """Every value *node* may take: a list of Python literals ([] = unknown)."""
        if node is None or depth > 8:
            return []
        if isinstance(node, ast.Name):
            if node.id in env:
                return list(env[node.id])
            if node.id in self.assigned:
                return self.value(self.assigned[node.id], env, depth + 1)
            return []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "join" and isinstance(node.func.value, ast.Constant) and node.args:
                out = []
                for parts in self.value(node.args[0], env, depth + 1):
                    if isinstance(parts, (list, tuple, set)) and all(isinstance(p, str) for p in parts):
                        out.append(node.func.value.value.join(parts))
                return out
            if attr in ("items", "keys", "values") and not node.args:
                out = []
                for d in self.value(node.func.value, env, depth + 1):
                    if isinstance(d, dict):
                        out.append(list(getattr(d, attr)()))
                return out
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # ``return LABELS[kw]``: any of the mapping's values.
            maps = [d for d in self.value(node.value, env, depth + 1) if isinstance(d, dict)]
            if maps:
                return [v for d in maps for v in d.values()]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "sorted", "tuple", "set") \
                and len(node.args) == 1:
            return [list(v) for v in self.value(node.args[0], env, depth + 1) if isinstance(v, (list, tuple, set, dict))]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elts = [self.value(e, env, depth + 1) for e in node.elts]
            if all(len(e) == 1 for e in elts):
                return [[e[0] for e in elts]]
            return []
        if isinstance(node, ast.Dict):
            keys = [self.value(k, env, depth + 1) if k is not None else [] for k in node.keys]
            vals = [self.value(v, env, depth + 1) for v in node.values]
            if all(len(k) == 1 for k in keys) and all(len(v) == 1 for v in vals):
                try:
                    return [{k[0]: v[0] for k, v in zip(keys, vals)}]
                except TypeError:
                    return []
            return []
        try:
            return [ast.literal_eval(node)]
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return []

    def elements(self, node: ast.AST, env: Mapping[str, list]) -> list:
        """What iterating *node* yields."""
        out = []
        for v in self.value(node, env):
            if isinstance(v, dict):
                out.extend(v.keys())
            elif isinstance(v, (list, tuple, set)):
                out.extend(v)
        return out

    # -- columns --------------------------------------------------------- #
    def columns(self, node: ast.AST, scope: Optional[ast.AST], params: Mapping[str, List[str]],
                depth: int = 0) -> List[str]:
        """The frame columns the text expression *node* is built from."""
        if depth > 6:
            return []
        out: List[str] = []
        for sub in ast.walk(node):
            key = _subscript_key(sub)
            if key is not None:
                out.append(key)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get" \
                    and sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
                out.append(sub.args[0].value)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id in params:
                    out.extend(params[sub.id])
                    continue
                rhs = _local_assign(scope, sub.id) if scope is not None else self.assigned.get(sub.id)
                if rhs is not None and rhs is not node:
                    out.extend(self.columns(rhs, scope, params, depth + 1))
            elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                    and sub.value.id not in _NOT_FRAMES and sub.attr not in _ATTR_METHODS:
                parent = self.parents.get(sub)
                if not (isinstance(parent, ast.Call) and parent.func is sub):
                    out.append(sub.attr)
        return list(dict.fromkeys(out))

    def folds_case(self, node: ast.AST, scope: Optional[ast.AST], depth: int = 0) -> bool:
        """Whether the text expression *node* is lower-cased on its way."""
        if depth > 6:
            return False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in _CASE_FOLD:
                return True
            if isinstance(sub, ast.Name):
                rhs = _local_assign(scope, sub.id) if scope is not None else self.assigned.get(sub.id)
                if rhs is not None and rhs is not node and self.folds_case(rhs, scope, depth + 1):
                    return True
            key = _subscript_key(sub)
            if key is not None:
                if any(self.folds_case(e, None, depth + 1) for e in self.column_exprs.get(key, [])):
                    return True
        return False


#: Names whose attributes are functions or modules, never columns.
_NOT_FRAMES = frozenset({"np", "pd", "numpy", "pandas", "math", "re", "os", "self", "str", "datetime", "dt",
                         "json", "ast", "string"})
#: Attributes that are methods or accessors, never a column.
_ATTR_METHODS = frozenset({"str", "lower", "upper", "casefold", "strip", "items", "keys", "values", "loc",
                           "iloc", "apply", "map", "fillna", "astype", "dtypes", "columns", "index", "shape",
                           "empty", "size", "T", "dt", "cat", "values"})


def _column_key(target: ast.AST) -> Optional[str]:
    """'col' of ``frame['col']`` or ``frame.loc[..., 'col']`` as an assignment target."""
    key = _subscript_key(target)
    if key is not None:
        return key
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute) \
            and target.value.attr in ("loc", "at") and isinstance(target.slice, ast.Tuple) \
            and len(target.slice.elts) == 2:
        col = target.slice.elts[1]
        if isinstance(col, ast.Constant) and isinstance(col.value, str):
            return col.value
    return None


def _subscript_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str) and not isinstance(node.value, ast.Attribute):
        return node.slice.value
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
            and node.value.attr not in ("loc", "at", "iloc") \
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _local_assign(scope: ast.AST, name: str) -> Optional[ast.AST]:
    """The last ``name = <expr>`` in *scope*."""
    found = None
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            found = node.value
    return found


@dataclass
class _Test:
    """One substring test: keywords looked for in a text expression."""
    patterns: List[str]
    keywords: List[str]
    text: ast.AST
    case_insensitive: bool


def _strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for v in values:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (list, tuple, set)):
            out.extend(_strings(v))
    return out


def _is_collection(code: _Code, node: ast.AST, env: Mapping[str, list]) -> bool:
    """Whether ``x in node`` is a membership test (a fixed set of values)."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return True
    # ``col in df.columns`` / ``key in d.keys()`` asks for a name, not a word in a text.
    if isinstance(node, ast.Attribute) and node.attr in ("columns", "index"):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("keys", "values"):
        return True
    return any(isinstance(v, (list, tuple, set, dict)) for v in code.value(node, env))


def _regex_alternatives(pattern: str) -> List[str]:
    """The top-level alternatives of a regex: ``'cake|pie'`` -> two keywords."""
    body = pattern
    m = re.fullmatch(r"(?:\(\?i\))?(?:\\b)?\((?:\?:)?(.*)\)(?:\\b)?", body)
    if m and "(" not in m.group(1):
        body = m.group(1)
    return [p for p in body.split("|") if p] if "(" not in body else [pattern]


def _tests(code: _Code, node: ast.AST, env: Mapping[str, list], scope: Optional[ast.AST]) -> List[_Test]:
    """The substring tests in *node*."""
    out: List[_Test] = []
    for sub in ast.walk(node):
        local_env = dict(env)
        if isinstance(sub, ast.Compare) and len(sub.ops) == 1 and isinstance(sub.ops[0], ast.In):
            right = sub.comparators[0]
            if _is_collection(code, right, local_env):
                continue
            words = [w for w in _strings(code.value(sub.left, _comprehension_env(code, sub, local_env))) if w]
            if words:
                out.append(_Test([re.escape(w) for w in words], words, right,
                                 code.folds_case(right, scope)))
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            func = sub.func
            text = None
            pattern_node = None
            regex, case_insensitive = True, False
            if func.attr in _CONTAINS and isinstance(func.value, ast.Attribute) and func.value.attr == "str" \
                    and sub.args:
                text, pattern_node = func.value.value, sub.args[0]
                for kw in sub.keywords:
                    if kw.arg == "regex" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        regex = False
                    if kw.arg == "case" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        case_insensitive = True
                    if kw.arg == "flags" and "I" in ast.unparse(kw.value):
                        case_insensitive = True
            elif func.attr in _RE_CALLS and isinstance(func.value, ast.Name) and func.value.id == "re" \
                    and len(sub.args) >= 2:
                text, pattern_node = sub.args[1], sub.args[0]
                flags = sub.args[2:] + [k.value for k in sub.keywords if k.arg == "flags"]
                case_insensitive = any("I" in ast.unparse(f) for f in flags)
            if text is None:
                continue
            raw = [w for w in _strings(code.value(pattern_node, _comprehension_env(code, sub, local_env))) if w]
            if not raw:
                continue
            if any(w.startswith("(?i)") for w in raw):
                case_insensitive = True
            patterns: List[str] = []
            keywords: List[str] = []
            for w in raw:
                if regex:
                    alts = _regex_alternatives(w.replace("(?i)", ""))
                    patterns.extend(alts)
                    keywords.extend(alts)
                else:
                    patterns.append(re.escape(w))
                    keywords.append(w)
            out.append(_Test(patterns, keywords, text, case_insensitive or code.folds_case(text, scope)))
    return out


def _comprehension_env(code: _Code, node: ast.AST, env: Mapping[str, list]) -> Dict[str, list]:
    """*env* plus the variables of the comprehensions enclosing *node*:
    ``any(k in s for k in KEYWORDS)``."""
    out = dict(env)
    parent = code.parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.Lambda, ast.Module)):
        if isinstance(parent, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            for gen in parent.generators:
                _bind(code, gen.target, gen.iter, out)
        parent = code.parents.get(parent)
    return out


def _bind(code: _Code, target: ast.AST, iterable: ast.AST, env: Dict[str, list]) -> None:
    """Bind the loop *target* to every value iterating *iterable* yields."""
    items = code.elements(iterable, env)
    if isinstance(target, ast.Name):
        env[target.id] = env.get(target.id, []) + items
    elif isinstance(target, ast.Tuple):
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Name):
                env[elt.id] = env.get(elt.id, []) + [
                    item[i] for item in items if isinstance(item, (list, tuple)) and len(item) > i]


def _loop_env(code: _Code, scope: ast.AST) -> Dict[str, list]:
    """The loop variables bound anywhere in *scope*, outer loops first."""
    env: Dict[str, list] = {}
    for node in ast.walk(scope):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            _bind(code, node.target, node.iter, env)
    return env


def _returned(code: _Code, func: ast.AST, env: Mapping[str, list]) -> tuple:
    """(labels a classifier function returns from a keyword branch, its default label)."""
    labels: List[str] = []
    default = None
    if isinstance(func, ast.Lambda):
        node = func.body
        while isinstance(node, ast.IfExp):
            labels.extend(_strings(code.value(node.body, env)))
            node = node.orelse
        values = _strings(code.value(node, env))
        default = values[0] if len(values) == 1 else None
        return labels, default
    body = list(func.body)
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            values = [v for v in code.value(node.value, env) if isinstance(v, str)]
            if node in body:
                default = values[0] if len(values) == 1 else default
            else:
                labels.extend(values)
    return labels, default


def _function_sites(code: _Code) -> List[KeywordClassification]:
    """Classifier functions: ``df['L'] = df.apply(classify, axis=1)`` and lambdas."""
    found: List[KeywordClassification] = []
    candidates: List[tuple] = []          # (function node, name or None)
    for name, fn in code.functions.items():
        candidates.append((fn, name))
    for node in ast.walk(code.tree):
        if isinstance(node, ast.Lambda) and isinstance(node.body, ast.IfExp):
            candidates.append((node, None))
    for fn, name in candidates:
        env = _loop_env(code, fn)
        tests = _tests(code, fn.body if isinstance(fn, ast.Lambda) else fn, env, fn)
        if not tests:
            continue
        labels, default = _returned(code, fn, env)
        keyword_labels = [l for l in dict.fromkeys(labels) if l != default]
        if len(keyword_labels) < 2:
            continue            # a filter or a yes/no flag, not a classification
        label_column, applied_to = _applied(code, fn, name)
        if label_column is None and applied_to is None and name is not None and not _called(code, name):
            continue            # defined, never run
        params = {}
        args = fn.args.args
        if args:
            first = args[0].arg
            params[first] = applied_to or []
        cols: List[str] = []
        for t in tests:
            cols.extend(c for c in code.columns(t.text, fn, params) if c)
        found.append(_classification(code, tests, cols, keyword_labels, label_column, default))
    return found


def _applied(code: _Code, fn: ast.AST, name: Optional[str]) -> tuple:
    """(label column, columns the function is applied to) for a classifier."""
    for node in ast.walk(code.tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _APPLY
                and node.args):
            continue
        arg = node.args[0]
        uses = arg is fn or (name is not None and (
            (isinstance(arg, ast.Name) and arg.id == name)
            or (isinstance(arg, ast.Lambda) and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                                   and c.func.id == name for c in ast.walk(arg)))))
        if not uses:
            continue
        receiver = _subscript_key(node.func.value)
        applied_to = [receiver] if receiver else None
        parent = code.parents.get(node)
        while parent is not None and not isinstance(parent, (ast.Assign, ast.stmt)):
            parent = code.parents.get(parent)
        label = _column_key(parent.targets[0]) if isinstance(parent, ast.Assign) and len(parent.targets) == 1 else None
        return label, applied_to
    return None, None


def _called(code: _Code, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for n in ast.walk(code.tree))


def _vector_sites(code: _Code) -> List[KeywordClassification]:
    """np.select / nested np.where over contains masks, and ``.loc[mask, 'L'] = 'label'`` chains."""
    found: List[KeywordClassification] = []
    module_env = _loop_env(code, code.tree)
    for node in ast.walk(code.tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "select" and len(node.args) >= 2:
            conds = node.args[0]
            if isinstance(conds, ast.Name):
                conds = code.assigned.get(conds.id, conds)
            choices = _strings(code.value(node.args[1], module_env))
            default = next((_strings(code.value(k.value, module_env)) for k in node.keywords if k.arg == "default"),
                           None)
            if len(node.args) > 2:
                default = _strings(code.value(node.args[2], module_env))
            tests = _tests(code, conds, module_env, None)
            default_label = default[0] if default else None
            keyword_labels = [c for c in dict.fromkeys(choices) if c != default_label]
            if tests and len(keyword_labels) >= 2:
                found.append(_vector_classification(code, node, tests, keyword_labels, default_label))
        elif node.func.attr == "where" and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy") \
                and len(node.args) == 3 and not _nested_where(code, node):
            labels: List[str] = []
            tests: List[_Test] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr == "where" \
                    and len(cur.args) == 3:
                tests.extend(_tests(code, cur.args[0], module_env, None))
                labels.extend(_strings(code.value(cur.args[1], module_env)))
                cur = cur.args[2]
            default = _strings(code.value(cur, module_env))
            default_label = default[0] if len(default) == 1 else None
            keyword_labels = [l for l in dict.fromkeys(labels) if l != default_label]
            if tests and len(keyword_labels) >= 2:
                found.append(_vector_classification(code, node, tests, keyword_labels, default_label))
    found.extend(_loc_sites(code, module_env))
    return found


def _nested_where(code: _Code, node: ast.AST) -> bool:
    parent = code.parents.get(node)
    return isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) and parent.func.attr == "where"


def _loc_sites(code: _Code, env: Mapping[str, list]) -> List[KeywordClassification]:
    """``df.loc[df['C'].str.contains('x'), 'L'] = 'label'``, twice or more for one L."""
    by_label: Dict[str, dict] = {}
    for node in ast.walk(code.tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and target.value.attr == "loc" and isinstance(target.slice, ast.Tuple) and len(target.slice.elts) == 2):
            continue
        label_column = _column_key(target)
        if label_column is None:
            continue
        mask = target.slice.elts[0]
        if isinstance(mask, ast.Name) and mask.id in code.assigned:
            mask = code.assigned[mask.id]
        tests = _tests(code, mask, env, None)
        values = _strings(code.value(node.value, env))
        if not tests or not values:
            continue
        site = by_label.setdefault(label_column, {"tests": [], "labels": [], "node": node})
        site["tests"].extend(tests)
        site["labels"].extend(values)
    found = []
    for label_column, site in by_label.items():
        default = None
        for expr in code.column_sets.get(label_column, []):
            values = _strings(code.value(expr, env))
            if len(values) == 1 and isinstance(expr, ast.Constant):
                default = values[0]
        keyword_labels = [l for l in dict.fromkeys(site["labels"]) if l != default]
        if len(keyword_labels) < 2:
            continue
        cols: List[str] = []
        for t in site["tests"]:
            cols.extend(code.columns(t.text, None, {}))
        found.append(_classification(code, site["tests"], cols, keyword_labels, label_column, default))
    return found


def _vector_classification(code: _Code, node: ast.AST, tests: List[_Test], labels: List[str],
                           default: Optional[str]) -> KeywordClassification:
    parent = code.parents.get(node)
    while parent is not None and not isinstance(parent, ast.stmt):
        parent = code.parents.get(parent)
    label = _column_key(parent.targets[0]) if isinstance(parent, ast.Assign) and len(parent.targets) == 1 else None
    cols: List[str] = []
    for t in tests:
        cols.extend(code.columns(t.text, None, {}))
    return _classification(code, tests, cols, labels, label, default)


def _classification(code: _Code, tests: List[_Test], cols: List[str], labels: List[str],
                    label_column: Optional[str], default: Optional[str]) -> KeywordClassification:
    patterns = list(dict.fromkeys(p for t in tests for p in t.patterns))
    keywords = list(dict.fromkeys(k for t in tests for k in t.keywords))
    columns = [c for c in dict.fromkeys(cols) if c != label_column]
    derived = {c: [d for d in code.columns(e, None, {}) if d != c]
               for c in columns for e in code.column_exprs.get(c, [])[:1]}
    return KeywordClassification(patterns=patterns, columns=columns, labels=labels, label_column=label_column,
                                 default_label=default, case_insensitive=any(t.case_insensitive for t in tests),
                                 derived={k: v for k, v in derived.items() if v}, keywords=keywords)


def keyword_classification(code: Any) -> Optional[KeywordClassification]:
    """The keyword classification in *code*, or None.

    Found: a function returning labels from keyword tests (``if 'x' in
    s.lower()`` chains, loops over a keyword dict or a label -> keywords dict)
    applied to rows or a column; ``np.select`` or nested ``np.where`` over
    ``str.contains`` masks; and two or more ``.loc[mask, 'L'] = label``
    assignments. It takes two labels that keywords assign: one keyword set is
    a filter or a flag, not a category breakdown.
    """
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    parsed = _Code(tree)
    try:
        sites = _function_sites(parsed) + _vector_sites(parsed)
    except RecursionError:
        return None
    sites = [s for s in sites if s.patterns and s.columns]
    if not sites:
        return None
    return max(sites, key=lambda s: (s.label_column is not None, len(s.patterns)))


def breakdown_reported(code: Any, cls: Any, records: Any) -> bool:
    """Whether the result reports the derived categories: the label column is
    in the result, or the code groups or counts by it."""
    label = cls.label_column
    if label is None:
        return bool(records)
    if isinstance(records, list) and records and isinstance(records[0], Mapping):
        if any(str(k).lower() == label.lower() for k in records[0]):
            return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _GROUP_CALLS:
            for sub in ast.walk(node):
                # A column by name, or a series held in a variable: the rweb16
                # recipes code counted ``name_tokens.value_counts()``, and with
                # only the column form recognised review() returned None (#584).
                if (isinstance(sub, ast.Constant) and sub.value == label) or \
                        (isinstance(sub, ast.Name) and sub.id == label):
                    return True
    return False


# --------------------------------------------------------------------------- #
# The data: unmatched share and language
# --------------------------------------------------------------------------- #

def _resolve(column: str, frame_columns: Sequence[str], derived: Mapping[str, List[str]],
             depth: int = 0) -> List[str]:
    if column in frame_columns:
        return [column]
    if depth > 4:
        return []
    out: List[str] = []
    for source in derived.get(column, []):
        out.extend(_resolve(source, frame_columns, derived, depth + 1))
    return list(dict.fromkeys(out))


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:          # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def measure(cls: KeywordClassification, frames: Sequence[tuple]) -> Optional[Dict[str, Any]]:
    """Unmatched share and languages, recomputed on the first frame that holds
    the classified columns."""
    for _label, frame in frames:
        try:
            names = [str(c) for c in frame.columns]
        except AttributeError:
            continue
        columns: List[str] = []
        for column in cls.columns:
            columns.extend(_resolve(column, names, cls.derived))
        columns = list(dict.fromkeys(columns))
        if not columns or len(frame) == 0:
            continue
        flags = re.IGNORECASE if cls.case_insensitive else 0
        try:
            pattern = re.compile("|".join(f"(?:{p})" for p in cls.patterns), flags)
        except re.error:
            pattern = re.compile("|".join(re.escape(k) for k in cls.keywords), flags)
        matched = None
        texts = []
        for column in columns:
            text = frame[column].map(_text)
            hit = text.map(lambda s: bool(pattern.search(s)))
            matched = hit if matched is None else (matched | hit)
            texts.extend(text.head(LANGUAGE_SAMPLE_ROWS).tolist())
        unmatched = 1.0 - float(matched.mean())
        data_lang = text_language(texts)
        attested = {t for s in texts for t in _tokens(s)} if data_lang else None
        return {
            "columns": columns,
            "rows": int(len(frame)),
            "unmatched_share": unmatched,
            "data_language": data_lang,
            "keyword_language": keyword_language(cls.keywords, attested),
        }
    return None


def _records_share(cls: KeywordClassification, records: Any) -> Optional[float]:
    """The default label's share of the result, when the data could not be read."""
    if cls.default_label is None or not isinstance(records, list) or not records:
        return None
    rows = [r for r in records if isinstance(r, Mapping)]
    key = next((k for k in rows[0] if cls.label_column and str(k).lower() == cls.label_column.lower()), None) \
        if rows else None
    if key is None:
        return None
    count_key = next((k for k in rows[0] if re.search(r"count|size|^n$|total|number", str(k), re.I)), None)
    try:
        counts = [float(r.get(count_key)) if count_key else 1.0 for r in rows]
    except (TypeError, ValueError):
        return None
    total = sum(counts)
    if total <= 0:
        return None
    other = sum(c for r, c in zip(rows, counts) if str(r.get(key)).lower() == cls.default_label.lower())
    return other / total


def _pct(share: float) -> str:
    return f"{share * 100:.1f}".rstrip("0").rstrip(".") + "%"


def _joined(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def review(code: Any, frames: Sequence[tuple], records: Any) -> Optional[Dict[str, Any]]:
    """The derived-category verdict for one turn, or None when its result is
    not a breakdown into categories the code derived."""
    cls = keyword_classification(code)
    if cls is not None and breakdown_reported(code, cls, records):
        return _keyword_review(cls, frames, records)
    derivation = label_derivation(code)
    if derivation is not None and breakdown_reported(code, derivation, records):
        try:
            return _derivation_review(derivation, frames, records)
        except Exception:  # noqa: BLE001 - a frame that will not read is no verdict, never a crash
            logger.debug("[derived-categories] could not check a derived label", exc_info=True)
    return None


def _keyword_review(cls: KeywordClassification, frames: Sequence[tuple], records: Any) -> Dict[str, Any]:
    measured = None
    try:
        measured = measure(cls, frames)
    except Exception:  # noqa: BLE001 - a frame that will not match is no verdict, never a crash
        logger.debug("[derived-categories] could not recompute the keyword match", exc_info=True)
    share = measured["unmatched_share"] if measured else _records_share(cls, records)
    columns = measured["columns"] if measured else cls.columns
    data_lang = measured.get("data_language") if measured else None
    kw_lang = measured.get("keyword_language") if measured else None
    on = _joined(columns)
    method = (f"categories assigned by keyword matching on {on}; {_pct(share)} unmatched." if share is not None
              else f"categories assigned by keyword matching on {on}; the unmatched share could not be computed.")
    reasons: List[str] = []
    if share is not None and share > MAX_UNMATCHED:
        reasons.append(f"{_pct(share)} of the rows matched no keyword, more than half, so the breakdown "
                       f"mostly measures what the keywords missed")
    if data_lang and kw_lang and data_lang != kw_lang:
        reasons.append(f"the keywords are {LANGUAGE_NAMES[kw_lang]} but the text in {on} is "
                       f"{LANGUAGE_NAMES[data_lang]}, so they cannot describe it")
    out = {
        "kind": "keyword_classification",
        "columns": list(columns),
        "label_column": cls.label_column,
        "default_label": cls.default_label,
        "labels": cls.labels[:50],
        "keywords": cls.keywords[:50],
        "n_keywords": len(cls.keywords),
        "unmatched_share": None if share is None else round(share, 6),
        "rows": measured["rows"] if measured else None,
        "data_language": data_lang,
        "keyword_language": kw_lang,
        "method": method,
        "withheld": bool(reasons),
        "reasons": reasons,
    }
    if reasons:
        out["reply"] = withheld_text(method, reasons)
    return out


# --------------------------------------------------------------------------- #
# Labels taken from free text: tokens, and value mappings (rc15)
# --------------------------------------------------------------------------- #
# The rc15 recipes chat answered "what kinds of recipes dominate?" with "the
# generic 'E' category, followed by 'Pfannkuchen', 'Se', 'Gef', and 'K'". The
# code split each German Name on [^0-9a-zA-Z]+, which cuts every umlaut out of
# a word ("Gefüllte" -> "gef", "llte"), and counted the pieces as recipe
# types. No keyword was matched, so the rule above never saw it. A label cut
# from free text is a derived category too: its method line says so, and a
# breakdown whose labels are mostly fragments is withheld.

#: Above this share of fragment labels the breakdown is withheld.
MAX_FRAGMENTS = 0.5
#: A column is free text when it has at least this many distinct values ...
FREE_TEXT_MIN_DISTINCT = 50
#: ... and this many words per distinct value on average. A code column has
#: one. (No ratio of distinct to rows: the recipes file repeats its 192 names
#: across 12,190 rows, and they are free text all the same.)
FREE_TEXT_MIN_WORDS = 1.5
#: Separators that split a free-text value into words, not a list into items.
_WORD_SEPARATORS = {None, " ", r"\s+", r"\s", r"\W+", r"[^\w]+"}
#: Result columns that may hold the derived labels, by name.
_LABEL_NAMES = re.compile(r"categor|type|kind|word|keyword|token|tag|label|group|class", re.IGNORECASE)

_METHOD = {
    "first_word": "categories taken from the first word of {on}.",
    "words": "categories taken from the words of {on}.",
    "pattern": "categories taken from a pattern extracted from {on}.",
    "mapping": "categories assigned by a value mapping on {on}; {unmatched} unmatched.",
}


@dataclass
class LabelDerivation:
    """Labels the code cut from, or mapped from, a free-text column."""
    kind: str                                   # first_word | words | pattern | mapping
    source: str                                 # the column, as the code names it
    label_column: Optional[str] = None
    mapping: Dict[str, Any] = field(default_factory=dict)
    case_insensitive: bool = False
    derived: Dict[str, List[str]] = field(default_factory=dict)
    #: Every column the code reads by name: a label may be cut from any of them.
    read_columns: List[str] = field(default_factory=list)


def _split_kind(call: ast.Call) -> Optional[str]:
    """'words' for a split into words, None for a split of a list ("a, b")."""
    sep = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg in ("pat", "sep")), None)
    if sep is None:
        return "words"
    if isinstance(sep, ast.Constant) and (sep.value is None or sep.value in _WORD_SEPARATORS):
        return "words"
    return None


def _first_of(code: _Code, node: ast.AST) -> bool:
    """Whether *node* (a list of tokens) is indexed at 0: ``.str[0]``, ``.str.get(0)``, ``[0]``."""
    parent = code.parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.attr == "str":
        grand = code.parents.get(parent)
        if isinstance(grand, ast.Subscript) and isinstance(grand.slice, ast.Constant) and grand.slice.value == 0:
            return True
        if isinstance(grand, ast.Attribute) and grand.attr == "get":
            call = code.parents.get(grand)
            return isinstance(call, ast.Call) and bool(call.args) and isinstance(call.args[0], ast.Constant) \
                and call.args[0].value == 0
    return isinstance(parent, ast.Subscript) and isinstance(parent.slice, ast.Constant) and parent.slice.value == 0


def _counted(code: _Code, node: ast.AST) -> bool:
    """``len(x.split())`` counts words; it derives no label."""
    parent = code.parents.get(node)
    return isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in ("len", "sum")


def _tokenizes(code: _Code, node: ast.AST, seen: Optional[Set[str]] = None) -> Optional[str]:
    """How *node* cuts a label from text: first_word, words, pattern, or None.

    Follows ``.apply``/``.map`` into the lambda or function it calls."""
    seen = set() if seen is None else seen
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        kind = None
        if isinstance(func, ast.Attribute):
            on_str = isinstance(func.value, ast.Attribute) and func.value.attr == "str"
            if func.attr == "split" and not (isinstance(func.value, ast.Name) and func.value.id == "re"):
                kind = _split_kind(sub)
            elif func.attr == "findall" and isinstance(func.value, ast.Name) and func.value.id == "re":
                kind = "words"
            elif func.attr == "split" and isinstance(func.value, ast.Name) and func.value.id == "re" and sub.args:
                sep = sub.args[0]
                kind = "words" if isinstance(sep, ast.Constant) and sep.value in _WORD_SEPARATORS else None
            elif on_str and func.attr in ("findall", "extractall"):
                kind = "words"
            elif on_str and func.attr == "extract":
                kind = "pattern"
            elif func.attr in _APPLY and sub.args:
                arg = sub.args[0]
                target = arg
                if isinstance(arg, ast.Name) and arg.id in code.functions and arg.id not in seen:
                    seen.add(arg.id)
                    target = code.functions[arg.id]
                if isinstance(target, (ast.Lambda, ast.FunctionDef)):
                    inner = _tokenizes(code, target, seen)
                    if inner:
                        return inner
                continue
        elif isinstance(func, ast.Name) and func.id in code.functions and func.id not in seen:
            seen.add(func.id)
            inner = _tokenizes(code, code.functions[func.id], seen)
            if inner:
                return inner
            continue
        if kind is None or _counted(code, sub):
            continue
        if kind == "words" and _first_of(code, sub):
            return "first_word"
        return kind
    return None


def _grouped(code: _Code, key: str) -> bool:
    """Whether the code groups or counts by *key* (a column or a variable)."""
    for node in ast.walk(code.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _GROUP_CALLS:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Constant) and sub.value == key) or (isinstance(sub, ast.Name) and sub.id == key):
                    return True
    return False


def label_derivation(code: Any) -> Optional[LabelDerivation]:
    """Labels the code takes from tokens of a text column (``str.split().str[0]``,
    ``str.extract``, ``re.findall`` in an applied function), or maps from a
    column's values (``.map(MAP)``, ``.replace({...})``), and then groups by.

    Whether the source is free text is a question for the data
    (:func:`free_text`); here it is only a candidate.
    """
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        parsed = _Code(ast.parse(code))
    except (SyntaxError, ValueError, RecursionError):
        return None
    tokenized: Optional[LabelDerivation] = None
    mapped: Optional[LabelDerivation] = None
    for node in ast.walk(parsed.tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        label = _column_key(target) or (target.id if isinstance(target, ast.Name) else None)
        if label is None:
            continue
        kind = _tokenizes(parsed, node.value)
        if kind and tokenized is None and _grouped(parsed, label):
            columns = [c for c in parsed.columns(node.value, None, {}) if c != label]
            if columns:
                tokenized = LabelDerivation(kind, columns[0], label_column=label,
                                            case_insensitive=parsed.folds_case(node.value, None))
                continue
        mapping = _value_mapping(parsed, node.value)
        if mapping is not None and mapped is None:
            receiver, table = mapping
            columns = parsed.columns(receiver, None, {})
            if columns and (_grouped(parsed, label) or tokenized is not None):
                mapped = LabelDerivation("mapping", columns[0], label_column=label, mapping=table,
                                         case_insensitive=parsed.folds_case(receiver, None))
    found = tokenized or mapped
    if found is None:
        return None
    if tokenized is not None and mapped is not None:
        tokenized.mapping = mapped.mapping     # tokens mapped to labels: the mapped labels are words
    found.derived = {c: [d for d in parsed.columns(e, None, {}) if d != c]
                     for c in [found.source] for e in parsed.column_exprs.get(c, [])[:1]}
    found.derived = {k: v for k, v in found.derived.items() if v}
    found.read_columns = list(dict.fromkeys(k for k in (_subscript_key(n) for n in ast.walk(parsed.tree)) if k))
    return found


def _value_mapping(code: _Code, node: ast.AST) -> Optional[tuple]:
    """(receiver, mapping) of a ``col.map({...})`` / ``col.replace({...})`` in *node*."""
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in ("map", "replace") and sub.args):
            continue
        receiver = sub.func.value
        if isinstance(receiver, ast.Attribute) and receiver.attr == "str":
            continue                      # str.replace edits text; it maps nothing
        tables = [v for v in code.value(sub.args[0], {}) if isinstance(v, dict)]
        if tables and len(tables[0]) >= 2 and all(isinstance(k, str) for k in tables[0]) \
                and not any(isinstance(v, dict) for v in tables[0].values()):
            return receiver, tables[0]
    return None


def free_text(series: Any) -> bool:
    """A column of free text: many distinct values, several words each. A code
    or a small category column (gender, a country code) is not."""
    values = series.dropna()
    if len(values) == 0 or values.map(lambda v: isinstance(v, str)).mean() < 0.9:
        return False
    distinct = values.drop_duplicates()
    if len(distinct) < FREE_TEXT_MIN_DISTINCT:
        return False
    return float(distinct.head(LANGUAGE_SAMPLE_ROWS).map(lambda v: len(_tokens(v))).mean()) >= FREE_TEXT_MIN_WORDS


def _source_frame(derivation: LabelDerivation, frames: Sequence[tuple]) -> Optional[tuple]:
    for _label, frame in frames:
        names = [str(c) for c in getattr(frame, "columns", [])]
        columns = _resolve(derivation.source, names, derivation.derived)
        if columns:
            return frame, columns[0]
    return None


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _long_format(rows: List[Mapping], key: Any, keys: List[Any], frame: Any) -> List[Mapping]:
    """The rows of a long-format result that hold derived labels.

    The rweb16 recipes code stacked five breakdowns into one Category/Value/
    Count table: Month, Weekday and Year beside its Name tokens and first
    ingredient tokens. Judged as one breakdown, "2005" and "Tuesday" were
    labels too. A tag column (a text column beside *key* whose values repeat)
    splits the result into blocks; a block every label of which is a whole
    value of one column of the data is a breakdown by that column, nothing
    derived, and is dropped. Without a tag column, or a frame, *rows* stand.
    """
    columns = [str(c) for c in getattr(frame, "columns", [])]
    if not columns:
        return rows
    tags = [k for k in keys if k != key and 1 < len({str(r.get(k)) for r in rows}) < len(rows)
            and not all(_number(r.get(k)) is not None for r in rows)]
    if not tags:
        return rows
    tag = tags[0]
    values: Dict[str, Set[str]] = {}

    def whole(column: str) -> Set[str]:
        if column not in values:
            series = frame[column]
            try:
                values[column] = {_text(v) for v in series.drop_duplicates().head(100000)}
            except TypeError:       # unhashable cells (lists): no whole value matches a label
                values[column] = set()
        return values[column]

    blocks: Dict[str, List[Mapping]] = {}
    for row in rows:
        blocks.setdefault(str(row.get(tag)), []).append(row)
    kept = []
    for block in blocks.values():
        labels = {_text(r.get(key)) for r in block} - {""}
        if labels and any(labels <= whole(c) for c in columns):
            continue
        kept.extend(block)
    return [r for r in rows if any(r is k for k in kept)]


def _shown_labels(derivation: LabelDerivation, records: Any, frame: Any = None) -> tuple:
    """(the distinct labels of the result, the leading one): the labels from
    the label column or the text column most like one; the leader by the
    result's count column, else its first row. In a long-format result only
    the blocks of derived labels count (:func:`_long_format`)."""
    rows = [r for r in records or [] if isinstance(r, Mapping)] if isinstance(records, list) else []
    if not rows:
        return [], None
    keys = list(rows[0])
    key = next((k for k in keys if derivation.label_column and str(k).lower() == derivation.label_column.lower()),
               None)
    if key is None:
        # Renamed on the way ('name_tokens' -> 'Word'): the text column with the
        # most distinct values, a label-like name first among equals. A constant
        # tag column ("type": "title_token") is not where the labels are.
        def numeric(v: Any) -> bool:
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False
        text = [k for k in keys if not all(numeric(r.get(k)) for r in rows)]
        if not text:
            return [], None
        key = max(text, key=lambda k: (len({str(r.get(k)) for r in rows}), bool(_LABEL_NAMES.search(str(k))),
                                       -keys.index(k)))
    rows = _long_format(rows, key, keys, frame)
    out = [str(r.get(key)) for r in rows if r.get(key) not in (None, "")]
    count_key = next((k for k in keys if k != key and re.search(r"count|size|^n$|total|number|freq", str(k), re.I)
                      and all(_number(r.get(k)) is not None for r in rows)), None)
    ranked = [r for r in rows if r.get(key) not in (None, "")]
    if count_key is not None:
        ranked = sorted(ranked, key=lambda r: -(_number(r.get(count_key)) or 0.0))
    leader = str(ranked[0].get(key)) if ranked else None
    return list(dict.fromkeys(out))[:50], leader


def fragments(labels: Sequence[str], vocabulary: Set[str], known: Iterable[str] = (),
              language: Optional[str] = None) -> List[str]:
    """The labels that are not words: under three letters, not a word the data
    uses, or a stopword of the data's *language* ("und" is no kind of recipe).
    Labels a mapping in the code produced are words by definition."""
    known_folded = {str(k).lower() for k in known}
    stopwords = set(_STOPWORDS[language].split()) if language in _STOPWORDS else set()
    out = []
    for label in labels:
        folded = label.strip().lower()
        if folded in known_folded:
            continue
        words = _tokens(folded)
        if len(folded) < 3 or not words or any(w not in vocabulary for w in words) \
                or all(w in stopwords for w in words):
            out.append(label)
    return out


def _derivation_review(derivation: LabelDerivation, frames: Sequence[tuple], records: Any) -> Optional[Dict[str, Any]]:
    found = _source_frame(derivation, frames)
    if found is None:
        return None                       # free text or a code column: only the data can say
    frame, column = found
    if not free_text(frame[column]):
        return None                       # a recode of a small category column is not derived
    texts = frame[column].map(_text)
    data_lang = text_language(texts.head(LANGUAGE_SAMPLE_ROWS).tolist())
    vocabulary = {t for s in texts for t in _tokens(s)}
    keys = [str(k) for k in derivation.mapping]
    # A label is a word if a column it may have been cut from uses it: the
    # rweb6 result put Ingredients' tokens beside Name's, and judged against
    # Name alone every ingredient read as a fragment.
    sources = {column}
    for name in derivation.read_columns:
        if name in frame.columns and name not in sources and frame[name].dtype == object:
            sources.add(name)
            vocabulary |= {t for s in frame[name].map(_text) for t in _tokens(s)}
    key_lang = keyword_language(keys, vocabulary) if keys else None
    unmatched = None
    if derivation.kind == "mapping":
        values = texts[texts != ""]
        values = values.str.lower() if derivation.case_insensitive else values
        lookup = {k.lower() for k in keys} if derivation.case_insensitive else set(keys)
        unmatched = 1.0 - float(values.isin(lookup).mean()) if len(values) else None
    method = _METHOD[derivation.kind].format(on=column, unmatched=_pct(unmatched) if unmatched is not None else "an unknown share")
    reasons: List[str] = []
    if unmatched is not None and unmatched > MAX_UNMATCHED:
        reasons.append(f"{_pct(unmatched)} of the rows have no entry in the mapping, more than half, so the "
                       f"breakdown mostly measures what the mapping missed")
    if data_lang and key_lang and data_lang != key_lang:
        reasons.append(f"the mapping's keys are {LANGUAGE_NAMES[key_lang]} but the text in {column} is "
                       f"{LANGUAGE_NAMES[data_lang]}, so they cannot describe it")
    labels, leader = _shown_labels(derivation, records, frame)
    if not labels and isinstance(records, list) and records:
        return None                       # every block is a breakdown by a column the data has
    pieces = fragments(labels, vocabulary, derivation.mapping.values(), data_lang) \
        if derivation.kind != "mapping" else []
    share = len(pieces) / len(labels) if labels else 0.0
    if share > MAX_FRAGMENTS:
        reasons.append(f"{_pct(share)} of its labels ({', '.join(pieces[:10])}) are fragments, shorter than three "
                       f"letters, a stopword, or not a word that occurs in {column}, so they name no category")
    elif leader is not None and leader in pieces:
        # The headline is what the reply leads with: "und" as the most common
        # recipe type is an artefact however few the other fragments are.
        reasons.append(f"its leading label, \u201c{leader}\u201d, is a fragment (shorter than three letters, a "
                       f"stopword, or not a word that occurs in {column}), so the breakdown leads with an artefact")
    out = {
        "kind": derivation.kind,
        "columns": [column],
        "label_column": derivation.label_column,
        "labels": labels,
        "fragments": pieces,
        "fragment_share": round(share, 6),
        "leading_label": leader,
        "mapping_keys": keys[:50],
        "unmatched_share": None if unmatched is None else round(unmatched, 6),
        "rows": int(len(frame)),
        "data_language": data_lang,
        "keyword_language": key_lang,
        "method": method,
        "withheld": bool(reasons),
        "reasons": reasons,
    }
    if reasons:
        out["reply"] = withheld_text(method, reasons)
    return out


def withheld_text(method: str, reasons: Sequence[str]) -> str:
    because = "; and ".join(reasons)
    return ("I've held back the category breakdown, so none of its figures is shown as a finding.\n"
            f"Method: {method}\nWhy: {because[0].upper()}{because[1:]}.")


# --------------------------------------------------------------------------- #
# Presentation (claim_verifier.presentable_reply)
# --------------------------------------------------------------------------- #

def _verdict(report: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    verdict = report.get("derived_categories") if isinstance(report, Mapping) else None
    return verdict if isinstance(verdict, Mapping) else None


def withheld_breakdown(report: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The reply in place of a breakdown the verdict withholds, or None."""
    verdict = _verdict(report)
    if verdict and verdict.get("withheld") and verdict.get("reply"):
        return str(verdict["reply"])
    return None


def with_method_line(reply: str, report: Optional[Mapping[str, Any]]) -> str:
    """*reply* with the method line of a breakdown that is shown."""
    verdict = _verdict(report)
    if not verdict or verdict.get("withheld") or not verdict.get("method"):
        return reply
    line = f"Method: {verdict['method']}"
    return reply if str(verdict["method"]) in reply else f"{reply}\n\n{line}"
