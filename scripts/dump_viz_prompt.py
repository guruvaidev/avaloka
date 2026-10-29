from dotenv import load_dotenv; load_dotenv(".env")
import pandas as pd
import app.agents.visualization_agent as va
captured = []
def fake(prompt, effort=None, *a, **k):
    captured.append(prompt); raise RuntimeError("captured")
va.invoke_viz_llm = fake
df = pd.read_csv("data/filestore/healtcaredataset (1).csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].iloc[:7000]
try:
    va.build_visualization_config_from_sample(dataset_id="t", sample_rows=df.to_dict("records"),
        schema={c: str(t) for c, t in df.dtypes.items()}, task_type="unsupervised", target_column=None)
except Exception:
    pass
open("/tmp/viz_prompt.txt", "w").write(captured[0])
print("saved", len(captured[0]), "chars")
