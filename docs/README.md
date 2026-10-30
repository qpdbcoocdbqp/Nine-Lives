## SPOT-dLLM

- [SPOT-dLLM](https://github.com/Eliezer318/SPOT-dLLM)

- Model
```bash
hf download JetLM/SDAR-1.7B-Chat-b32
hf download openai/gsm8k --repo-type dataset
```

- Clone repository
```bash
git clone https://github.com/Eliezer318/SPOT-dLLM docs/source/SPOT-dLLM
```

- Run RTX 5070 Ti

| Method | Accuracy | Passes/question | Sec/question | Tokens/s | Overall speedup | Median per-question speedup | Per-question speedup range |
|---|---:|---:|---:|---:|---:|---:|---:|
| draft-free | 0.64 | 269.3 | 8.97 | 19.9 | 1.00x | — | — |
| SPOT, budget 8 | 0.62 | 74.5 | 4.19 | 43.1 | 2.14x | 2.13x | 1.65–2.94x |
| SPOT-γ, γ=0.1 | 0.70 | 82.2 | 3.80 | 47.3 | 2.36x | 2.36x | 1.59–3.15x |

