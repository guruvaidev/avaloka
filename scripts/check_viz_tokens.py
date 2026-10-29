from dotenv import load_dotenv
load_dotenv(".env")                                  # explicit path: works from any runner
import pandas as pd
import app.agents.visualization_agent as va

try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    count = lambda s: len(enc.encode(s))
except Exception:
    count = lambda s: len(s) // 4

captured = []
def fake_llm(prompt, effort=None, *a, **k):
    captured.append(prompt)
    raise RuntimeError("captured")
va.invoke_viz_llm = fake_llm

def prompt_tokens(df):
    captured.clear()
    try:
        va.build_visualization_config_from_sample(
            dataset_id="t", sample_rows=df.to_dict("records"),
            schema={c: str(t) for c, t in df.dtypes.items()},
            task_type="unsupervised", target_column=None)
    except Exception:
        pass
    if not captured:
        return "no LLM call (still disabled?)"
    p = captured[0]
    return f"{count(p):>6} tokens, all columns in prompt: {all(str(c) in p for c in df.columns)}"

df = pd.read_csv("data/filestore/healtcaredataset (1).csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
print("file shape:", df.shape)

print("\nROWS test (all columns):")
for n in (100, 2000, 7000):
    print(f"  rows={n:>5}  ->", prompt_tokens(df.iloc[:n]))

print("\nCOLUMNS test (7000 rows):")
for c in (15, 21, 42, df.shape[1]):
    print(f"  cols={c:>3}  ->", prompt_tokens(df.iloc[:7000, :c]))
