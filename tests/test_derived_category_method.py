"""A category breakdown the code made up by keyword matching (rc14, defect D).

The rc14 recipes chat ("what kinds of recipes dominate?") matched ENGLISH
keywords against German recipe names and ingredients, and called every row
that matched none "other". Its reply, every figure of which reconciled with
the result and so passed the claim verifier: "Other recipes dominate ...
67.6%", the leader, "Sauce" second, "five times", "Sushi smallest". All of
them describe the keyword list, not the recipes.

Rule D (the coordinator's default; the owner may overrule it): a breakdown
into keyword-derived categories carries the method line "categories assigned
by keyword matching on <column>; N% unmatched.", and is withheld, with the
reason, when more than half the rows match no keyword or when the keywords'
language is not the data's.

The reply, code and result rows are the ones rc14 produced
(rweb14/results/rw7/recipes_jsonl/chat.json); the data is a German recipe file
in which 67.6% of rows contain none of the keywords. These tests run the
graph's path: ``claim_verifier_node`` then ``presentable_reply``, as
app/api/server.py does.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply

# --------------------------------------------------------------------------- #
# rc14 recipes_jsonl, verbatim
# --------------------------------------------------------------------------- #

RECIPES_QUESTION = "what kinds of recipes dominate?"
RECIPES_REPLY = ("Other recipes dominate, making up about 67.6 % of the collection.  \n\n"
                 "- **Leader**: “Other” with 8,235 items (≈67.6 %).  \n"
                 "- **Second**: “Sauce” at 1,589 items (≈13.0 %).  \n"
                 "- **Largest gap**: “Other” is roughly five times the next category, “sauce”.  \n"
                 "- **Surprise**: “Sushi” is the smallest group with only 2 items (≈0.02 %).")
RECIPES_CODE = """import pandas as pd
KEYWORD_MAP = {'salad': 'salad', 'soup': 'soup', 'pizza': 'pizza', 'pasta': 'pasta', 'cake': 'dessert', 'cookie': 'dessert', 'bread': 'bread', 'sandwich': 'sandwich', 'burger': 'burger', 'stew': 'stew', 'curry': 'curry', 'taco': 'taco', 'sushi': 'sushi', 'wrap': 'wrap', 'smoothie': 'smoothie', 'pie': 'dessert', 'biscuit': 'dessert', 'frittata': 'egg', 'omelette': 'egg', 'quiche': 'egg', 'risotto': 'rice', 'paella': 'rice', 'dumpling': 'dumpling', 'sauce': 'sauce'}

def infer_category(row):
    \"\"\"Infer a recipe category for a single row.\"\"\"
    name = str(row.get('Name', '')).lower()
    for kw, cat in KEYWORD_MAP.items():
        if kw in name:
            return cat
    ingredients = str(row.get('Ingredients', '')).lower()
    for kw, cat in KEYWORD_MAP.items():
        if kw in ingredients:
            return cat
    return 'other'

def main(df):
    print('Columns and dtypes:')
    print(df.dtypes)
    df['Category'] = df.apply(infer_category, axis=1)
    agg = df.groupby('Category', as_index=False).size().rename(columns={'size': 'Count'})
    total_recipes = agg['Count'].sum()
    agg['Percent'] = agg['Count'] / total_recipes * 100
    agg = agg.sort_values('Count', ascending=False).reset_index(drop=True)
    agg['Category'] = agg['Category'].fillna('other')
    agg['Count'] = agg['Count'].fillna(0)
    agg['Percent'] = agg['Percent'].fillna(0.0)
    return agg
if __name__ == '__main__':
    df = read_csv_best_effort('input.csv')
    result_df = main(df)
    result_df.to_csv('output.csv', index=False)"""
RECIPES_ROWS = [
    {"Category": "other", "Count": "8235", "Percent": "67.55537325676785"},
    {"Category": "sauce", "Count": "1589", "Percent": "13.035274815422476"},
    {"Category": "curry", "Count": "899", "Percent": "7.374897456931912"},
    {"Category": "dessert", "Count": "484", "Percent": "3.970467596390484"},
    {"Category": "rice", "Count": "242", "Percent": "1.985233798195242"},
    {"Category": "egg", "Count": "236", "Percent": "1.9360131255127155"},
    {"Category": "pizza", "Count": "148", "Percent": "1.214109926168991"},
    {"Category": "pasta", "Count": "128", "Percent": "1.0500410172272354"},
    {"Category": "burger", "Count": "92", "Percent": "0.7547169811320755"},
    {"Category": "salad", "Count": "59", "Percent": "0.4840032813781789"},
    {"Category": "sandwich", "Count": "59", "Percent": "0.4840032813781789"},
    {"Category": "taco", "Count": "9", "Percent": "0.07383100902378999"},
    {"Category": "wrap", "Count": "8", "Percent": "0.06562756357670221"},
    {"Category": "sushi", "Count": "2", "Percent": "0.016406890894175553"},
]
#: Every artefact figure of the rc14 reply.
RECIPES_ARTEFACTS = ("67.6", "Leader", "Sauce", "five times", "Sushi")

#: German recipes no keyword matches, and ones a keyword (sauce, curry, pizza) does.
UNMATCHED = [("Gebratener Hasenrücken", ["1 Hasenrücken", "50 g Butter", "Salz und Pfeffer", "etwas Thymian"]),
             ("Kartoffelsuppe mit Speck", ["1 kg Kartoffeln", "100 g Speck", "1 Zwiebel(n)", "Salz und Pfeffer"]),
             ("Herzhafte Pfannkuchen", ["250 g Mehl", "3 Ei(er)", "500 ml Milch", "etwas Öl"]),
             ("Rehrücken in Salzkruste mit Waldpilzgulasch", ["1 Rehrücken", "1 kg Salz", "300 g Pilze", "Lorbeer"]),
             ("Gefüllte Schinkenröllchen", ["8 Scheiben Schinken", "200 g Frischkäse", "Schnittlauch oder Petersilie"]),
             ("Fischsuppe nach Omas Art", ["500 g Fischfilet", "2 Möhren", "1 Liter Gemüsebrühe", "Salz und Pfeffer"])]
MATCHED = [("Nudeln mit Tomatensauce", ["500 g Nudeln", "1 Dose Tomaten", "Salz und Pfeffer"]),
           ("Hähnchen-Curry mit Reis", ["400 g Hähnchenbrust", "200 g Reis", "2 EL Currypulver"]),
           ("Pizza mit Salami und Pilzen", ["1 Pizzateig", "100 g Salami", "etwas Käse oder Mozzarella"]),
           ("Eier in Senfsauce", ["6 Ei(er)", "2 EL Senf", "200 ml Sahne und etwas Mehl"])]


@pytest.fixture
def recipes_path(tmp_path):
    """1,000 German recipes, 676 of which contain no keyword: 67.6% unmatched."""
    rows = [UNMATCHED[i % len(UNMATCHED)] for i in range(676)] + [MATCHED[i % len(MATCHED)] for i in range(324)]
    path = tmp_path / "recipes.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, (name, ingredients) in enumerate(rows):
            fh.write(json.dumps({"Url": f"https://example.org/rezepte/{i}/", "Ingredients": ingredients,
                                 "Name": name, "Year": 2020}, ensure_ascii=False) + "\n")
    return path


def _turn(question, reply, code, rows, path):
    """The state the graph hands the verifier, and the reply the user is shown."""
    state = {"messages": [HumanMessage(content=question), AIMessage(content=reply)],
             "coder_definition": {"code": code},
             "execution_result": {"status": "success", "reads": [{"path": str(path), "used": True}]},
             "output_json": rows}
    out = claim_verifier_node(state)
    return out, presentable_reply(reply, out["verification_report"])


def test_rc14_recipes_breakdown_is_withheld_for_both_reasons(recipes_path):
    out, shown = _turn(RECIPES_QUESTION, RECIPES_REPLY, RECIPES_CODE, RECIPES_ROWS, recipes_path)
    for artefact in RECIPES_ARTEFACTS:
        assert artefact not in shown.split("Why:")[0].replace("67.6% unmatched", ""), shown
    assert "I've held back the category breakdown" in shown
    assert "categories assigned by keyword matching on Name and Ingredients; 67.6% unmatched." in shown
    # Both triggers, stated.
    assert "67.6% of the rows matched no keyword, more than half" in shown
    assert "the keywords are English but the text in Name and Ingredients is German" in shown
    assert out["verification_safe_to_present"] is False
    assert out["verification_report"]["verified"] is False


def _english_path(tmp_path, labels):
    """Support tickets in English, one title per row, by the keyword they contain."""
    titles = {"billing": "I was charged twice on my billing statement and want the money back",
              "delivery": "The delivery of my order is late and the tracking has not moved",
              "login": "I cannot login to my account after the password reset",
              None: "The screen of the device went dark when I opened the box this morning"}
    path = tmp_path / "tickets.csv"
    pd.DataFrame({"ticket_id": range(len(labels)), "title": [titles[l] for l in labels]}).to_csv(path, index=False)
    return path


TICKETS_CODE = """import numpy as np
import pandas as pd

def main(df):
    title = df['title'].str.lower()
    conditions = [title.str.contains('billing|refund|charged'), title.str.contains('delivery|tracking|shipping'),
                  title.str.contains('login|password')]
    df['topic'] = np.select(conditions, ['Billing', 'Delivery', 'Login'], default='Other')
    out = df['topic'].value_counts().rename_axis('topic').reset_index(name='tickets')
    out['share'] = out['tickets'] / out['tickets'].sum() * 100
    return out
"""
TICKETS_QUESTION = "what are the tickets about?"
TICKETS_REPLY = "Billing is the largest topic with 400 tickets (40.0% of all tickets)."


def test_english_keywords_on_english_text_are_shown_with_the_method_line(tmp_path):
    labels = ["billing"] * 400 + ["delivery"] * 250 + ["login"] * 150 + [None] * 200     # 20% unmatched
    rows = [{"topic": "Billing", "tickets": "400", "share": "40.0"}, {"topic": "Delivery", "tickets": "250", "share": "25.0"},
            {"topic": "Other", "tickets": "200", "share": "20.0"}, {"topic": "Login", "tickets": "150", "share": "15.0"}]
    out, shown = _turn(TICKETS_QUESTION, TICKETS_REPLY, TICKETS_CODE, rows, _english_path(tmp_path, labels))
    assert shown.startswith(TICKETS_REPLY)
    assert shown.endswith("Method: categories assigned by keyword matching on title; 20% unmatched.")
    assert out["verification_report"]["derived_categories"]["withheld"] is False


LOC_CODE = """import pandas as pd

def main(df):
    df['topic'] = 'Other'
    df.loc[df['title'].str.contains('billing', case=False), 'topic'] = 'Billing'
    df.loc[df['title'].str.contains('delivery', case=False), 'topic'] = 'Delivery'
    return df.groupby('topic', as_index=False).size()
"""


def test_most_rows_unmatched_in_the_same_language_is_withheld(tmp_path):
    labels = ["billing"] * 250 + ["delivery"] * 150 + [None] * 600                       # 60% unmatched
    rows = [{"topic": "Other", "size": "600"}, {"topic": "Billing", "size": "250"}, {"topic": "Delivery", "size": "150"}]
    reply = "Most tickets (600) fall under Other, and Billing leads the named topics with 250."
    out, shown = _turn(TICKETS_QUESTION, reply, LOC_CODE, rows, _english_path(tmp_path, labels))
    assert "600" not in shown and "250" not in shown.replace("60% unmatched", "")
    assert "categories assigned by keyword matching on title; 60% unmatched." in shown
    assert "60% of the rows matched no keyword, more than half" in shown
    assert "keywords are" not in shown                     # same language: one reason only
    assert out["verification_safe_to_present"] is False


@pytest.mark.parametrize("code", [
    # A column the data already has: nothing was matched.
    """import pandas as pd

def main(df):
    return df.groupby('topic', as_index=False).size()
""",
    # One str.contains filter is a filter, not a classification.
    """import pandas as pd

def main(df):
    billing = df[df['title'].str.contains('billing', case=False)]
    return billing.groupby('topic', as_index=False).size()
""",
], ids=["plain-groupby", "single-contains-filter"])
def test_a_breakdown_by_an_existing_column_is_untouched(tmp_path, code):
    path = tmp_path / "tickets.csv"
    pd.DataFrame({"title": ["billing problem"] * 3 + ["other thing"] * 7,
                  "topic": ["Billing"] * 3 + ["Hardware"] * 7}).to_csv(path, index=False)
    rows = [{"topic": "Hardware", "size": "7"}, {"topic": "Billing", "size": "3"}]
    reply = "Hardware is the largest topic with 7 tickets."
    out, shown = _turn(TICKETS_QUESTION, reply, code, rows, path)
    # Untouched by the derived-category rule: the reply is shown, with no method
    # line and no derived-category verdict. Other checks may still add their own
    # caveat (e.g. #586 discloses the unrequested billing filter), so the test
    # pins only what this rule owns.
    assert shown.startswith(reply)
    assert "Method: categories assigned by keyword matching" not in shown
    assert "held back the category breakdown" not in shown
    assert "derived_categories" not in out["verification_report"]


# --------------------------------------------------------------------------- #
# rc15: labels cut from free text (rweb15 rw7c recipes_jsonl, verbatim)
# --------------------------------------------------------------------------- #
# The code split each German Name on [^0-9a-zA-Z]+, which cuts the umlauts out
# of words ("Gefüllte" -> "gef", "llte"), and counted the pieces as recipe
# types. No keyword was matched, so on 6fc75e99 review() returned None and the
# reply went out as written.

TOKENS_REPLY = 'The most common recipe types are the generic “E” category, followed by “Pfannkuchen”, “Se”, “Gef”, and “K”.  \n\n- **Top five**: E (602), Pfannkuchen (552), Se (539), Gef (499), K (457).  \n- **Next tier**: Tomaten (455) and Llte (438) are close behind.  \n- **Lower tier**: Auflauf (402), B (398), and H (388) trail the leaders.  \n- **Overall picture**: The data shows a strong skew toward simple or generic categories, with “E” dominating the list.'
TOKENS_CODE = 'import pandas as pd\nimport re\n\ndef _clean_text(text, stopwords):\n    """Lower‑case, replace non‑alphanumeric chars with space, split, remove stopwords."""\n    if pd.isna(text):\n        return []\n    cleaned = re.sub(\'[^0-9a-zA-Z]+\', \' \', str(text).lower())\n    tokens = [tok for tok in cleaned.split() if tok and tok not in stopwords]\n    return tokens\n\ndef main(df):\n    df = df.dropna(subset=[\'Name\'])\n    STOPWORDS = {\'mit\', \'und\', \'the\', \'a\', \'recipe\', \'ein\', \'eine\', \'der\', \'die\', \'das\', \'von\', \'zu\', \'für\', \'auf\', \'im\', \'in\', \'ist\', \'zu\', \'zu\', \'zu\', \'zu\'}\n    df[\'name_tokens\'] = df[\'Name\'].apply(lambda x: _clean_text(x, STOPWORDS))\n    df_exploded = df.explode(\'name_tokens\')\n    token_counts = df_exploded.groupby(\'name_tokens\').size().reset_index(name=\'Count\').rename(columns={\'name_tokens\': \'Word\'})\n    token_counts = token_counts.sort_values(\'Count\', ascending=False).head(10)\n    CATEGORY_MAP = {\'salat\': \'Salad\', \'pizza\': \'Pizza/Flatbread\', \'suppe\': \'Soup\', \'kuchen\': \'Cake\', \'brot\': \'Bread\', \'pasta\': \'Pasta\', \'risotto\': \'Rice Dish\', \'torte\': \'Cake\', \'omelett\': \'Egg Dish\', \'quiche\': \'Savory Tart\'}\n    token_counts[\'Category\'] = token_counts[\'Word\'].map(CATEGORY_MAP).fillna(token_counts[\'Word\'].str.title())\n    token_counts = token_counts[[\'Word\', \'Count\', \'Category\']]\n    return token_counts\nif __name__ == \'__main__\':\n    df = read_csv_best_effort(\'input.csv\')\n    df = main(df)\n    df.to_csv(\'output.csv\', index=False)'
TOKENS_ROWS = [{"Word": "e", "Count": "602", "Category": "E"}, {"Word": "pfannkuchen", "Count": "552", "Category": "Pfannkuchen"}, {"Word": "se", "Count": "539", "Category": "Se"}, {"Word": "gef", "Count": "499", "Category": "Gef"}, {"Word": "k", "Count": "457", "Category": "K"}, {"Word": "tomaten", "Count": "455", "Category": "Tomaten"}, {"Word": "llte", "Count": "438", "Category": "Llte"}, {"Word": "auflauf", "Count": "402", "Category": "Auflauf"}, {"Word": "b", "Count": "398", "Category": "B"}, {"Word": "h", "Count": "388", "Category": "H"}]
#: The fragments the reply presented as recipe types.
TOKEN_FRAGMENTS = ("“E”", "“Se”", "“Gef”", "“K”", "Llte")

ADJECTIVES = ["Herzhafte", "Gefüllte", "Überbackene", "Schnelle", "Bunte", "Würzige"]
DISHES = ["Pfannkuchen", "Tomaten", "Auflauf mit Käse", "Zucchini", "Paprika", "Kartoffeln", "Nudeln und Speck",
          "Champignons", "Hähnchenbrust", "Spätzle"]


@pytest.fixture
def names_path(tmp_path):
    """60 distinct German recipe names, each repeated: free text."""
    names = [f"{a} {d}" for a in ADJECTIVES for d in DISHES] * 5
    path = tmp_path / "recipes.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, name in enumerate(names):
            fh.write(json.dumps({"Name": name, "Year": 2020, "Day": i % 28 + 1}, ensure_ascii=False) + "\n")
    return path


def test_rc15_token_fragments_are_withheld(names_path):
    out, shown = _turn(RECIPES_QUESTION, TOKENS_REPLY, TOKENS_CODE, TOKENS_ROWS, names_path)
    for fragment in TOKEN_FRAGMENTS:
        assert fragment not in shown, shown
    assert "I've held back the category breakdown" in shown
    assert "Method: categories taken from the words of Name." in shown
    assert "70% of its labels (e, se, gef, k, llte, b, h) are fragments" in shown
    assert out["verification_safe_to_present"] is False


FIRST_WORD_CODE = """import pandas as pd

CATEGORY_MAP = {'chicken': 'Poultry', 'beef': 'Meat', 'tomato': 'Vegetable'}

def main(df):
    df['main_ingredient'] = df['title'].str.split().str[0].str.lower()
    out = df.groupby('main_ingredient', as_index=False).size()
    out['group'] = out['main_ingredient'].map(CATEGORY_MAP).fillna('Other')
    return out
"""
FIRST = ["Chicken", "Beef", "Tomato", "Lentil", "Mushroom"]
REST = ["soup with fresh herbs", "stew for the winter", "salad and bread", "pie with a crust", "curry with rice",
        "wrap for lunch", "bake with cheese", "risotto with peas", "skewers on the grill", "noodles and greens",
        "tart with onions", "burger with fries"]
#: 60 distinct English recipe titles.
TITLES = [f"{f} {r}" for f in FIRST for r in REST]


def test_first_word_of_english_titles_is_shown_with_the_method_line(tmp_path):
    path = tmp_path / "titles.csv"
    pd.DataFrame({"title": TITLES * 2}).to_csv(path, index=False)
    rows = [{"main_ingredient": w.lower(), "size": "24", "group": g}
            for w, g in zip(FIRST, ["Poultry", "Meat", "Vegetable", "Other", "Other"])]
    reply = "Chicken, beef, tomato, lentil and mushroom dishes each make up 24 of the recipes."
    out, shown = _turn(RECIPES_QUESTION, reply, FIRST_WORD_CODE, rows, path)
    assert shown.startswith(reply)
    assert shown.endswith("Method: categories taken from the first word of title.")
    assert out["verification_report"]["derived_categories"]["withheld"] is False


FREE_TEXT_MAP_CODE = """import pandas as pd

def main(df):
    df['cuisine'] = df['title'].map({'Chicken soup with fresh herbs': 'Soup', 'Beef stew for the winter': 'Stew',
                                     'Tomato salad and bread': 'Salad'}).fillna('Other')
    return df.groupby('cuisine', as_index=False).size()
"""


def test_a_value_mapping_on_free_text_that_misses_most_rows_is_withheld(tmp_path):
    path = tmp_path / "titles.csv"
    pd.DataFrame({"title": TITLES}).to_csv(path, index=False)
    rows = [{"cuisine": "Other", "size": "57"}, {"cuisine": "Soup", "size": "1"}, {"cuisine": "Stew", "size": "1"},
            {"cuisine": "Salad", "size": "1"}]
    reply = "Other dominates with 57 recipes, and Soup, Stew and Salad have 1 each."
    out, shown = _turn(RECIPES_QUESTION, reply, FREE_TEXT_MAP_CODE, rows, path)
    assert "57 recipes" not in shown
    assert "Method: categories assigned by a value mapping on title; 95% unmatched." in shown
    assert "95% of the rows have no entry in the mapping, more than half" in shown


def test_a_recode_of_a_small_category_column_is_untouched(tmp_path):
    path = tmp_path / "patients.csv"
    pd.DataFrame({"gender": ["M"] * 30 + ["F"] * 25 + ["U"] * 5, "age": range(60)}).to_csv(path, index=False)
    code = """import pandas as pd

def main(df):
    df['sex'] = df['gender'].map({'M': 'Male', 'F': 'Female', 'U': 'Unknown'})
    return df.groupby('sex', as_index=False).size()
"""
    rows = [{"sex": "Male", "size": "30"}, {"sex": "Female", "size": "25"}, {"sex": "Unknown", "size": "5"}]
    reply = "Male patients are the largest group with 30."
    out, shown = _turn("how are patients split by sex?", reply, code, rows, path)
    assert shown.startswith(reply)
    assert "Method: categories" not in shown
    assert "derived_categories" not in out["verification_report"]


# --------------------------------------------------------------------------- #
# rweb6 groq recipes_jsonl, verbatim: "und" as the most common recipe type
# --------------------------------------------------------------------------- #
# Title tokens reported as recipe types, led by the German stopword "und",
# beside the first ingredient of each recipe. On 4882e1d9 the ingredient
# labels were judged against Name's words, half the labels read as fragments
# (45%, under the 50% bar) and the breakdown was shown with its method line.

UND_REPLY = 'The most common recipe types are title tokens, with “und” appearing 1,210 times, followed by “pfannkuchen” (552) and “tomaten” (455).  \n- **Leader:** “und” 1,210, showing a very high frequency of generic or placeholder titles.  \n- **Top non‑generic titles:** “pfannkuchen” 552, “tomaten” 455, “gefüllte” 438, “auflauf” 402.  \n- **Main ingredients:** “g kartoffel” 486 dominates, then “g fischfilet” 356 and “zwiebel” 306.  \n- **Surprise:** The most frequent title token (“und”) far exceeds any specific recipe name, indicating many titles are incomplete or generic.'
UND_CODE = 'import pandas as pd\nimport re\nimport json\n\ndef _load_input():\n    """\n    Load the dataset. Prefer a JSON‑L file named \'input.jsonl\';\n    if it does not exist, fall back to a CSV file named \'input.csv\'.\n    """\n    try:\n        return pd.read_json(\'input.jsonl\', lines=True)\n    except ValueError:\n        return pd.read_csv(\'input.csv\')\n    except FileNotFoundError:\n        return pd.read_csv(\'input.csv\')\n\ndef _parse_ingredients(col):\n    """\n    Ensure the Ingredients column is a list.\n    If the entry is a JSON string, parse it; otherwise return as‑is.\n    """\n    if pd.isna(col):\n        return None\n    if isinstance(col, list):\n        return col\n    try:\n        parsed = json.loads(col)\n        if isinstance(parsed, list):\n            return parsed\n    except Exception:\n        pass\n    return None\n\ndef main(df: pd.DataFrame) -> pd.DataFrame:\n    df = df[df[\'Name\'].notna()].copy()\n    stop_words = {\'the\', \'and\', \'or\', \'ein\', \'eine\', \'einer\', \'einem\', \'einen\', \'der\', \'die\', \'das\', \'den\', \'dem\', \'des\', \'ein\', \'einige\', \'einiger\', \'einige\', \'für\', \'mit\', \'auf\', \'in\', \'zu\', \'von\', \'aus\', \'bei\', \'nach\', \'über\', \'unter\', \'zwischen\', \'ohne\', \'gegen\', \'am\', \'im\', \'an\', \'ist\', \'wie\', \'auch\', \'sind\', \'was\', \'wenn\', \'dann\', \'diese\', \'dieser\', \'dieses\'}\n\n    def tokenize(text):\n        tokens = re.findall(\'[A-Za-zÄÖÜäöüß]+\', text.lower())\n        return [t for t in tokens if t not in stop_words]\n    df[\'title_tokens\'] = df[\'Name\'].astype(str).apply(tokenize)\n    token_counts = df.explode(\'title_tokens\').rename(columns={\'title_tokens\': \'value\'}).groupby(\'value\', as_index=False).size().rename(columns={\'size\': \'count\'})\n    token_counts = token_counts.sort_values(\'count\', ascending=False).head(10)\n    token_counts[\'type\'] = \'title_token\'\n    df[\'ingredients_parsed\'] = df[\'Ingredients\'].apply(_parse_ingredients)\n\n    def extract_main(ing_list):\n        if not ing_list:\n            return None\n        first = ing_list[0]\n        if not isinstance(first, str):\n            return None\n        s = re.sub(\'^\\\\d+[\\\\s/.]*\', \'\', first)\n        s = re.sub(\'\\\\(.*?\\\\)\', \'\', s)\n        parts = re.split(\',|für\', s, flags=re.IGNORECASE)\n        main = parts[0].strip().lower()\n        return main if main else None\n    df[\'main_ingredient\'] = df[\'ingredients_parsed\'].apply(extract_main)\n    ingredient_counts = df.dropna(subset=[\'main_ingredient\']).rename(columns={\'main_ingredient\': \'value\'}).groupby(\'value\', as_index=False).size().rename(columns={\'size\': \'count\'})\n    ingredient_counts = ingredient_counts.sort_values(\'count\', ascending=False).head(10)\n    ingredient_counts[\'type\'] = \'main_ingredient\'\n    result = pd.concat([token_counts, ingredient_counts], ignore_index=True)\n    result = result[[\'type\', \'value\', \'count\']]\n    return result\nif __name__ == \'__main__\':\n    df = _load_input()\n    df = main(df)\n    df.to_csv(\'output.csv\', index=False)'
UND_ROWS = [{"type": "title_token", "value": "und", "count": "1210"}, {"type": "title_token", "value": "pfannkuchen", "count": "552"}, {"type": "title_token", "value": "tomaten", "count": "455"}, {"type": "title_token", "value": "gefüllte", "count": "438"}, {"type": "title_token", "value": "auflauf", "count": "402"}, {"type": "title_token", "value": "fischsuppe", "count": "385"}, {"type": "title_token", "value": "herzhafte", "count": "374"}, {"type": "title_token", "value": "zucchini", "count": "373"}, {"type": "title_token", "value": "sahne", "count": "324"}, {"type": "title_token", "value": "pfanne", "count": "323"}, {"type": "main_ingredient", "value": "g kartoffel", "count": "486"}, {"type": "main_ingredient", "value": "g fischfilet", "count": "356"}, {"type": "main_ingredient", "value": "zwiebel", "count": "306"}, {"type": "main_ingredient", "value": "g hackfleisch", "count": "267"}, {"type": "main_ingredient", "value": "g nudeln", "count": "252"}, {"type": "main_ingredient", "value": "g mehl", "count": "245"}, {"type": "main_ingredient", "value": "kg kartoffel", "count": "230"}, {"type": "main_ingredient", "value": "ei", "count": "206"}, {"type": "main_ingredient", "value": "g hähnchenbrustfilet", "count": "148"}, {"type": "main_ingredient", "value": "paprikaschote", "count": "137"}]

UND_DISHES = ["Pfannkuchen", "Tomaten mit Sahne", "Auflauf", "Fischsuppe", "Zucchini und Paprika", "Pfanne mit Speck",
              "Kartoffeln und Quark", "Nudeln mit Tomaten", "Pilze in Sahne", "Eier und Spinat"]
UND_ADJECTIVES = ["Herzhafte", "Gefüllte", "Schnelle", "Bunte", "Würzige", "Feine"]
#: First ingredients, as the recipes list them: the result's main_ingredient values.
UND_INGREDIENTS = ["500 g Kartoffel(n)", "400 g Fischfilet", "1 Zwiebel(n)", "300 g Hackfleisch", "250 g Nudeln",
                   "200 g Mehl", "1 kg Kartoffel(n)", "2 Ei(er)", "300 g Hähnchenbrustfilet", "1 Paprikaschote(n)"]


@pytest.fixture
def und_path(tmp_path):
    """60 distinct German recipe names with ingredient lists, as the recipes file has them."""
    path = tmp_path / "recipes.jsonl"
    names = [f"{a} {d}" for a in UND_ADJECTIVES for d in UND_DISHES] * 3
    with path.open("w", encoding="utf-8") as fh:
        for i, name in enumerate(names):
            ingredients = [UND_INGREDIENTS[i % len(UND_INGREDIENTS)], "Salz und Pfeffer", "etwas Öl oder Butter"]
            fh.write(json.dumps({"Name": name, "Ingredients": ingredients, "Year": 2020}, ensure_ascii=False) + "\n")
    return path


def test_rweb6_und_as_the_leading_recipe_type_is_withheld(und_path):
    out, shown = _turn(RECIPES_QUESTION, UND_REPLY, UND_CODE, UND_ROWS, und_path)
    assert "1,210" not in shown and "“und”" not in shown.split("Why:")[0], shown
    assert "I've held back the category breakdown" in shown
    assert "Method: categories taken from the words of Name." in shown
    assert "Its leading label, “und”, is a fragment" in shown
    verdict = out["verification_report"]["derived_categories"]
    assert verdict["fragments"] == ["und", "ei"]          # ingredients judged against Ingredients, not Name
    assert out["verification_safe_to_present"] is False



# --------------------------------------------------------------------------- #
# rweb16 rw7b recipes_jsonl, verbatim: a long-format result (#584)
# --------------------------------------------------------------------------- #
# The code stacked five breakdowns into one Category/Value/Count table: Month,
# Weekday and Year beside the word tokens of Name and the first token of each
# ingredient. The reply led with "g" 5,552 and "kg" 1,232 and called the
# stopword "in" the top name token. On d199d120 label_derivation() found the
# tokens, but the label was a variable (``name_tokens.value_counts()``), which
# breakdown_reported() did not recognise, so review() returned None and the
# reply went out with no method line.

LONG_REPLY = 'The most common recipe elements are ingredient units, with “g” appearing 5,552 times and “kg” 1,232 times.  \n\n- **Ingredient units dominate**: “g” far outpaces other categories (5,552 vs. 1,232 for “kg”).  \n- **Weekday usage is high**: Tuesday leads with 2,042 mentions, while Monday trails at 1,531.  \n- **Months cluster in early year**: February (1,353) and January (1,342) are the top two, with October and November near the bottom (905–910).  \n- **Name tokens are sparse**: the most frequent token “in” appears 1,014 times, far below the top ingredient counts.'
LONG_CODE = 'import re\nimport pandas as pd\n\ndef _validate_columns(df: pd.DataFrame, required):\n    missing = [c for c in required if c not in df.columns]\n    if missing:\n        raise ValueError(f\'Missing required columns: {missing}\')\n\ndef _tokenizer(text: str):\n    """Return list of alphabetic tokens in lower‑case."""\n    return re.findall(\'[a-zA-Z]+\', text.lower())\n\ndef main(df: pd.DataFrame) -> pd.DataFrame:\n    required_cols = [\'Month\', \'Weekday\', \'Year\', \'Name\', \'Ingredients\']\n    _validate_columns(df, required_cols)\n    month_counts = df[\'Month\'].dropna().value_counts()\n    weekday_counts = df[\'Weekday\'].dropna().value_counts()\n    year_counts = df[\'Year\'].dropna().astype(str).value_counts()\n    stop_words = {\'the\', \'und\', \'mit\', \'ein\', \'eine\'}\n    name_tokens = df[\'Name\'].dropna().astype(str).apply(_tokenizer).explode().loc[lambda s: ~s.isin(stop_words)]\n    name_counts = name_tokens.value_counts().head(5)\n    ingredients_exploded = df[\'Ingredients\'].explode()\n    main_ingredients = ingredients_exploded.dropna().astype(str).apply(_tokenizer).apply(lambda toks: toks[0] if toks else None).dropna()\n    ingredient_counts = main_ingredients.value_counts().head(5)\n\n    def _make_df(series: pd.Series, category: str) -> pd.DataFrame:\n        tmp = series.reset_index()\n        tmp.columns = [\'Value\', \'Count\']\n        tmp[\'Category\'] = category\n        tmp[\'Category\'] = tmp[\'Category\'].astype(str)\n        tmp[\'Value\'] = tmp[\'Value\'].astype(str)\n        return tmp[[\'Category\', \'Value\', \'Count\']]\n    dfs = [_make_df(month_counts, \'Month\'), _make_df(weekday_counts, \'Weekday\'), _make_df(year_counts, \'Year\'), _make_df(name_counts, \'NameToken\'), _make_df(ingredient_counts, \'Ingredient\')]\n    dominant_df = pd.concat(dfs, ignore_index=True)\n    dominant_df = dominant_df.sort_values(\'Count\', ascending=False).reset_index(drop=True)\n    return dominant_df\nif __name__ == \'__main__\':\n    df = pd.read_csv(\'input.csv\')\n    result = main(df)\n    result.to_csv(\'output.csv\', index=False)'
#: (Category, Value, Count): the 46 result rows.
LONG_ROWS = [{"Category": c, "Value": v, "Count": str(n)} for c, v, n in
             [("Ingredient", "g", 5552), ("Weekday", "Tuesday", 2042), ("Weekday", "Friday", 1851), ("Weekday", "Thursday", 1731),
               ("Weekday", "Saturday", 1716), ("Weekday", "Sunday", 1661), ("Weekday", "Wednesday", 1658), ("Weekday", "Monday", 1531),
               ("Month", "February", 1353), ("Month", "January", 1342), ("Ingredient", "kg", 1232), ("Month", "September", 1018),
               ("NameToken", "in", 1014), ("Month", "April", 978), ("Month", "May", 962), ("Month", "July", 958),
               ("Month", "December", 950), ("Month", "August", 945), ("Month", "March", 935), ("Month", "June", 934),
               ("Month", "November", 910), ("Month", "October", 905), ("Year", "2005", 762), ("Year", "2012", 761),
               ("Year", "2017", 746), ("Year", "2010", 744), ("Year", "2006", 740), ("Year", "2016", 740),
               ("Year", "2004", 733), ("Year", "2003", 725), ("Year", "2008", 722), ("Year", "2009", 720),
               ("Year", "2013", 713), ("Year", "2011", 704), ("Year", "2019", 703), ("Year", "2015", 682),
               ("Year", "2018", 672), ("Year", "2007", 668), ("Year", "2014", 655), ("NameToken", "e", 602),
               ("NameToken", "pfannkuchen", 552), ("NameToken", "se", 539), ("NameToken", "gef", 499), ("Ingredient", "f", 395),
               ("Ingredient", "zwiebel", 310), ("Ingredient", "m", 288)]]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@pytest.fixture
def long_path(tmp_path):
    """60 distinct German recipe names with ingredient lists, a month, a weekday and a year."""
    path = tmp_path / "recipes.jsonl"
    names = [f"{a} {d}" for a in UND_ADJECTIVES for d in UND_DISHES] * 3
    with path.open("w", encoding="utf-8") as fh:
        for i, name in enumerate(names):
            fh.write(json.dumps({"Name": name, "Ingredients": [UND_INGREDIENTS[i % len(UND_INGREDIENTS)], "Salz"],
                                 "Month": MONTHS[i % 12], "Weekday": WEEKDAYS[i % 7], "Year": 2003 + i % 17},
                                ensure_ascii=False) + "\n")
    return path


def test_rweb16_long_format_token_breakdown_is_withheld(long_path):
    out, shown = _turn(RECIPES_QUESTION, LONG_REPLY, LONG_CODE, LONG_ROWS, long_path)
    assert "5,552" not in shown and "1,014" not in shown, shown
    assert "I've held back the category breakdown" in shown
    assert "Method: categories taken from the words of Name." in shown
    verdict = out["verification_report"]["derived_categories"]
    # Only the derived blocks are labels: Month, Weekday and Year are columns the data has.
    assert verdict["labels"] == ["g", "kg", "in", "e", "pfannkuchen", "se", "gef", "f", "zwiebel", "m"]
    assert verdict["leading_label"] == "g"
    assert "in" in verdict["fragments"]                    # the stopword
    assert out["verification_safe_to_present"] is False


def test_a_long_format_result_of_existing_columns_only_is_untouched(long_path):
    rows = [r for r in LONG_ROWS if r["Category"] in ("Month", "Weekday")]
    reply = "Tuesday leads the weekdays with 2,042 recipes, and February the months with 1,353."
    out, shown = _turn(RECIPES_QUESTION, reply, LONG_CODE, rows, long_path)
    assert "held back the category breakdown" not in shown
    assert "derived_categories" not in out["verification_report"]
