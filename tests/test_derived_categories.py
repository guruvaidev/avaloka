"""The keyword-classification detector and the language check
(app/agents/derived_categories), one shape at a time.

The end-to-end cases are in test_derived_category_method.py; these pin which
code counts as a derived category and which does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.derived_categories import (fragments, free_text, keyword_classification, keyword_language,
                                           label_derivation, measure, review, text_language)

#: rweb7 groq recipes_jsonl, verbatim: a label -> keywords dict over derived columns.
GROQ_RECIPES = """import pandas as pd
import ast

def main(df):
    df['name_lc'] = df['Name'].astype(str).str.lower()

    def ingredients_to_string(val):
        if isinstance(val, list):
            return ' '.join((str(item).lower() for item in val))
        return str(val).lower()
    df['ingredients_lc'] = df['Ingredients'].apply(ingredients_to_string)
    keyword_map = {'salad': ['salad', 'lettuce'], 'soup': ['soup', 'broth', 'stew'], 'dessert': ['cake', 'pie']}

    def assign_category(row):
        search_str = f"{row['name_lc']} {row['ingredients_lc']}"
        for cat, keywords in keyword_map.items():
            for kw in keywords:
                if kw in search_str:
                    return cat
        return 'other'
    df['category'] = df.apply(assign_category, axis=1)
    return df.groupby('category', as_index=False).size()
"""

IF_CHAIN = """import pandas as pd

def kind(s):
    s = s.lower()
    if 'cake' in s or 'pie' in s:
        return 'Dessert'
    elif 'soup' in s:
        return 'Soup'
    return 'Other'

def main(df):
    df['kind'] = df['title'].apply(kind)
    return df['kind'].value_counts().reset_index()
"""

LAMBDA_CHAIN = """import pandas as pd

def main(df):
    df['kind'] = df['title'].apply(lambda s: 'Dessert' if 'cake' in s else 'Soup' if 'soup' in s else 'Other')
    return df.groupby('kind').size().reset_index(name='n')
"""

NP_SELECT = """import numpy as np

def main(df):
    t = df['title']
    df['kind'] = np.select([t.str.contains('cake|pie', case=False), t.str.contains('soup', case=False)],
                           ['Dessert', 'Soup'], default='Other')
    return df.groupby('kind', as_index=False).size()
"""

NESTED_WHERE = """import numpy as np

def main(df):
    df['kind'] = np.where(df['title'].str.contains('cake'), 'Dessert',
                          np.where(df['title'].str.contains('soup'), 'Soup', 'Other'))
    return df.groupby('kind', as_index=False).size()
"""

JOINED_LIST = """import pandas as pd
GROUPS = {'Dessert': ['cake', 'pie'], 'Soup': ['soup']}

def main(df):
    df['kind'] = 'Other'
    for label, words in GROUPS.items():
        df.loc[df['title'].str.contains('|'.join(words), case=False), 'kind'] = label
    return df.groupby('kind', as_index=False).size()
"""


@pytest.mark.parametrize("code,label,columns", [
    (GROQ_RECIPES, "category", ["name_lc", "ingredients_lc"]),
    (IF_CHAIN, "kind", ["title"]),
    (LAMBDA_CHAIN, "kind", ["title"]),
    (NP_SELECT, "kind", ["title"]),
    (NESTED_WHERE, "kind", ["title"]),
    (JOINED_LIST, "kind", ["title"]),
], ids=["dict-label-to-keywords", "apply-if-chain", "lambda-chain", "np-select", "nested-np-where", "loc-joined-list"])
def test_keyword_classification_is_found(code, label, columns):
    cls = keyword_classification(code)
    assert cls is not None
    assert cls.label_column == label
    assert cls.columns == columns
    assert {"cake", "soup"} <= set(cls.keywords)


@pytest.mark.parametrize("code", [
    "def main(df):\n    return df.groupby('category', as_index=False).size()\n",
    "def main(df):\n    return df[df['title'].str.contains('cake')].shape[0]\n",
    "def main(df):\n    df['is_cake'] = df['title'].str.contains('cake|pie')\n    return df.groupby('is_cake').size()\n",
    "import numpy as np\ndef main(df):\n    df['f'] = np.where(df['title'].str.contains('cake'), 'Yes', 'No')\n"
    "    return df.groupby('f').size()\n",
    "def main(df):\n    df['g'] = df['gender'].map({'M': 'Male', 'F': 'Female'})\n    return df.groupby('g').size()\n",
    "def main(df):\n    df['g'] = df['gender'].apply(lambda s: 'Male' if s in ('M', 'm') else 'Female' if s in ('F',) else 'X')\n"
    "    return df.groupby('g').size()\n",
    "def main(df):\n    for col in ['a', 'b']:\n        if col in df.columns:\n            return 'yes'\n    return 'no'\n",
    "this is not python (",
], ids=["groupby", "one-filter", "boolean-flag", "yes-no-where", "value-recode", "membership-chain",
        "column-check", "unparseable"])
def test_not_a_derived_category(code):
    assert keyword_classification(code) is None


def test_unmatched_share_is_recomputed_on_the_derived_columns_sources():
    frame = pd.DataFrame({"Name": ["Chicken Soup", "Apple Pie", "Steak", "Rice"],
                          "Ingredients": [["broth"], ["apples"], ["beef"], ["lettuce"]]})
    measured = measure(keyword_classification(GROQ_RECIPES), [("r", frame)])
    assert measured["columns"] == ["Name", "Ingredients"]
    assert measured["unmatched_share"] == 0.25          # Steak only


def test_the_data_language_is_its_stopwords():
    german = ["Nudeln mit Tomatensauce und etwas Basilikum oder Petersilie"] * 10
    english = ["The delivery of my order was late and the box was damaged"] * 10
    assert text_language(german) == "de"
    assert text_language(english) == "en"
    assert text_language(["SKU-1029 A7", "SKU-2210 B3"] * 30) is None


def test_the_keywords_language_ignores_loanwords_the_data_uses():
    keywords = ["salad", "soup", "cake", "pizza", "curry", "sauce"]
    assert keyword_language(keywords) == "en"
    assert keyword_language(["salat", "suppe", "kuchen", "pizza"]) == "de"
    assert keyword_language(["pizza", "curry", "sushi"]) is None           # loanwords: no language
    assert keyword_language(["salad", "soup", "pizza"], attested={"salad", "soup"}) is None


def test_a_breakdown_the_result_does_not_report_is_no_verdict():
    code = IF_CHAIN.replace("return df['kind'].value_counts().reset_index()", "return df[['title']].head()")
    frame = pd.DataFrame({"title": ["cake", "soup", "x"]})
    assert review(code, [("t", frame)], [{"title": "cake"}]) is None


# --------------------------------------------------------------------------- #
# Labels cut from, or mapped from, free text (rc15)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code,kind,source", [
    ("def main(df):\n    df['t'] = df['Name'].str.split().str[0]\n    return df.groupby('t').size()\n",
     "first_word", "Name"),
    ("def main(df):\n    df['t'] = df['Name'].str.split(expand=True)[0]\n    return df['t'].value_counts()\n",
     "first_word", "Name"),
    ("def main(df):\n    df['t'] = df['Name'].str.extract(r'^(\\w+)')\n    return df.groupby('t').size()\n",
     "pattern", "Name"),
    ("import re\ndef main(df):\n    df['t'] = df['Name'].apply(lambda s: re.findall(r'\\w+', s)[0])\n"
     "    return df.groupby('t').size()\n", "first_word", "Name"),
    ("def main(df):\n    df['t'] = df['Name'].map({'a b': 'X', 'c d': 'Y'})\n    return df.groupby('t').size()\n",
     "mapping", "Name"),
], ids=["split-str0", "split-expand", "str-extract", "findall-first", "map-dict"])
def test_label_derivation_is_found(code, kind, source):
    found = label_derivation(code)
    assert found is not None
    assert (found.kind, found.source) == (kind, source)


@pytest.mark.parametrize("code", [
    "def main(df):\n    df['n'] = df['Name'].apply(lambda s: len(s.split()))\n    return df.groupby('n').size()\n",
    "def main(df):\n    df['i'] = df['Ingredients'].str.split(',')\n    return df.explode('i').groupby('i').size()\n",
    "def main(df):\n    df['d'] = df['email'].str.split('@').str[1]\n    return df.groupby('d').size()\n",
    "def main(df):\n    df['c'] = df['Name'].str.replace('-', ' ')\n    return df.groupby('c').size()\n",
    "def main(df):\n    df['t'] = df['Name'].str.split().str[0]\n    return df.head()\n",
], ids=["word-count", "comma-list", "email-domain", "str-replace", "not-grouped"])
def test_not_a_label_derivation(code):
    assert label_derivation(code) is None


def test_free_text_is_many_distinct_values_of_several_words():
    names = pd.Series([f"Gefüllte Paprika Nummer {i}" for i in range(60)] * 3)
    assert free_text(names)
    assert not free_text(pd.Series(["M", "F", "U"] * 40))
    assert not free_text(pd.Series([f"SKU{i}" for i in range(200)]))


def test_fragments_are_short_or_not_words_of_the_data():
    vocabulary = {"gefüllte", "pfannkuchen", "tomaten"}
    labels = ["e", "pfannkuchen", "gef", "llte", "Tomaten", "Soup"]
    assert fragments(labels, vocabulary, known=["Soup"]) == ["e", "gef", "llte"]


def test_a_stopword_of_the_data_language_is_a_fragment():
    vocabulary = {"und", "pfannkuchen", "the", "pancakes"}
    assert fragments(["und", "pfannkuchen"], vocabulary, language="de") == ["und"]
    assert fragments(["the", "pancakes"], vocabulary, language="en") == ["the"]
    assert fragments(["und", "pfannkuchen"], vocabulary) == []          # no language, no stopword rule
