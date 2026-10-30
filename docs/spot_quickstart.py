from typing import Any
import sys
sys.path.append("docs/source/SPOT-dLLM/")
import time
import torch
# pyrefly: ignore[missing-import]
from spot.decode import DecodeConfig, generate
# pyrefly: ignore[missing-import]
from spot.drafters import SlotDrafter
# pyrefly: ignore[missing-import]
from spot.models import load_adapter
# pyrefly: ignore[missing-import]
from spot.tasks import load_task, score_task


device = "cuda"
dtype = torch.bfloat16
target = load_adapter("sdar17", device=device, dtype=dtype)
drafter = SlotDrafter("docs/source/SPOT-dLLM/drafters/sdar1.7b_b32.pt", target)
GEN_LENGTH = 256
print(f"SDAR-1.7B-Chat-b32 on {device} in {str(dtype).replace('torch.', '')}, with its SPOT drafter")

"""## 4. One prompt: draft-free decoding vs SPOT

`budget` is the number of drafted block states the model checks per pass. `gamma` turns on **SPOT-γ**, which
drafts fewer states when the drafter is unsure, so each pass is cheaper. The first call warms up the GPU.
"""

def spot_decode(prompt, budget=0, gamma=0.0, temperature=0.0, seed=0, gen_length=None, pad=None):
    """One generation. budget=0 is draft-free decoding (one commit per pass); gamma>0 is SPOT-γ."""
    drafting = budget > 0
    if pad is None:  # SPOT pads every pass to the full budget; SPOT-γ does not
        pad = (0 if gamma > 0 else budget) if drafting else 0
    cfg = DecodeConfig(
        cache="on", unmasking="single", verify="match", suffix="skip", draft_layout="concat",
        draft="slot" if drafting else "none", slot_mode="ddtree_b", slot_tokgate=0.6,
        draft_budget=max(budget, 1), pad_pieces=pad, slot_gamma=gamma,
        gen_length=gen_length or GEN_LENGTH, block_length=32, temperature=temperature, seed=seed)
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    ids = target.encode_prompt(messages)
    return generate(target, ids, cfg, drafter=drafter if drafting else None)


question = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
            "in May. How many clips did Natalia sell altogether in April and May?")

# warm-up
spot_decode(question, budget=8, gen_length=32)

runs = {
    "draft-free": spot_decode(question),
    "SPOT, budget 8": spot_decode(question, budget=8),
    "SPOT-γ, γ = 0.1": spot_decode(question, budget=8, gamma=0.1),
}
base = runs["draft-free"]["wall_s_net"]
print(f"{'':18}{'passes':>8}{'seconds':>10}{'speed-up':>10}")
for name, r in runs.items():
    print(f"{name:18}{r['nfe']:>8}{r['wall_s_net']:>10.2f}{base / r['wall_s_net']:>9.2f}x")
print("\n" + runs["SPOT, budget 8"]["text"])

reference = spot_decode(question, budget=0, pad=8)  # draft-free, padded to SPOT's pass shape
print("SPOT output identical to draft-free decoding:", reference["text"] == runs["SPOT, budget 8"]["text"])

# A small GSM8K benchmark
n = 10
samples = load_task("gsm8k", limit=n)


def benchmark(**kw):
    total = {"accuracy": 0.0, "passes": 0.0, "seconds": 0.0}
    for s in samples:
        r = spot_decode(s.messages, **kw)
        total["accuracy"] += score_task("gsm8k", s, r["text"])
        total["passes"] += r["nfe"]
        total["seconds"] += r["wall_s_net"]
    return {k: v / len(samples) for k, v in total.items()}


results: dict[str, Any] = {
    "draft-free": benchmark(),
    "SPOT, budget 8": benchmark(budget=8),
    "SPOT-γ, γ = 0.1": benchmark(budget=8, gamma=0.1),
}
base = results["draft-free"]["seconds"]
print(f"{'':18}{'accuracy':>9}{'passes':>8}{'seconds':>10}{'speed-up':>10}")
for name, m in results.items():
    print(f"{name:18}{m['accuracy']:>9.2f}{m['passes']:>8.1f}{m['seconds']:>10.2f}{base / m['seconds']:>9.2f}x")

## Sampling at T = 1
for seed in range(3):
    r = spot_decode(question, budget=8, temperature=1.0, seed=seed)
    answer = " ".join(r["text"].split())
    print(f"seed {seed}: {r['nfe']} passes | ...{answer[-90:]}")
