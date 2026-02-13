"""Quick integration test for Phase 0 data pipeline + reward."""

from grubrics_science.data.adapters import get_adapter, ADAPTERS
from grubrics_science.rewards.gsm8k_reward import compute_score, parse_rubric_items

print("=== Available adapters ===")
for name in sorted(ADAPTERS):
    a = get_adapter(name)
    print(f"  {name}: data_source={a.data_source}, domain_type={a.domain_type}")

print("\n=== VerifiableMath adapter (local CSV) ===")
adapter = get_adapter("olympiad_math")
items = adapter.load_raw()
print(f"Loaded {len(items)} items")
if items:
    row = adapter.to_verl_format(items[0])
    print(f"  data_source: {row['data_source']}")
    print(f"  domain_type: {row['extra_info']['domain_type']}")
    q = row["extra_info"]["question"]
    print(f"  question: {q[:100]}{'...' if len(q) > 100 else ''}")

print("\n=== FrontierScience adapter (local JSONL) ===")
fs_adapter = get_adapter("frontierscience")
fs_items = fs_adapter.load_raw()
print(f"Loaded {len(fs_items)} items")
if fs_items:
    row = fs_adapter.to_verl_format(fs_items[0])
    print(f"  data_source: {row['data_source']}")
    print(f"  has golden_rubric: {bool(row['extra_info']['golden_rubric'])}")

print("\n=== Reward function test ===")
good_rubric = (
    "Points: 3.0, Item: The answer correctly identifies the key equation\n"
    "Points: 2.5, Item: The answer shows all derivation steps\n"
    "Points: 2.0, Item: The answer arrives at the correct numerical result\n"
    "Points: 1.5, Item: The answer handles boundary conditions properly\n"
    "Points: 1.0, Item: The answer is clearly written and well-organized"
)
score = compute_score("gsm8k", good_rubric, extra_info={"question": "Solve 2x+3=7"})
print(f"Good rubric reward: {score:.3f}")

bad_rubric = "This is not a rubric at all."
score2 = compute_score("gsm8k", bad_rubric, extra_info={"question": "Solve 2x+3=7"})
print(f"Bad rubric reward:  {score2:.3f}")

parsed = parse_rubric_items(good_rubric)
total = sum(p for p, _ in parsed)
print(f"Parsed: {len(parsed)} items, total={total}")

print("\n=== Parquet generation test (VerifiableMath, 5 items) ===")
path = adapter.to_parquet(
    output_dir="./data/processed/test",
    max_items=5,
    split="test",
)
print(f"Written: {path}")

import pandas as pd
df = pd.read_parquet(path)
print(f"Parquet shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n=== All tests passed ===")
