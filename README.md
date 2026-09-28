# Resume → JSON: QLoRA fine-tuning of Qwen2.5-1.5B-Instruct

Fine-tunes a small open model to turn resume text into a fixed JSON schema, and measures how much the fine-tuning helps against a zero-shot baseline of the same model.

```json
{"name": "...", "email": "...", "skills": ["..."],
 "education": [{"institution": "...", "degree": "...", "field": "..."}],
 "experience": [{"company": "...", "title": "..."}]}
```

**Read this first: what this project is and isn't.** The training text is *rendered from structured JSON* (the source dataset has no raw resume text) and the dataset is ~97% synthetic. Scores on the synthetic test sets are therefore near-ceiling and say little about real-world resume parsing. The more meaningful number is the small held-out set of anonymized real records (section 4). This is a fine-tuning and evaluation exercise, not a production resume parser. See [Limitations](#6-limitations).

---

## 1. Results

Same model, same prompt, same test resumes for both rows. "Field score" is the mean of five per-field scores (exact match for name and email, set-F1 for skills, education entries and experience entries). Unparseable output scores 0.

| test set | model | valid JSON | exact schema | skills | education | experience | **field score** | s / resume |
|---|---|---|---|---|---|---|---|---|
| seen layouts (n=100) | zero-shot | 96% | 20% | 0.46 | 0.56 | 0.71 | **0.71** | 6.6 |
| | fine-tuned | 100% | 100% | 1.00 | 0.99 | 1.00 | **0.995** | 2.7 |
| unseen layout (n=98) | zero-shot | 86% | 1% | 0.54 | 0.47 | 0.80 | **0.70** | 6.5 |
| | fine-tuned | 100% | 100% | 1.00 | 0.99 | 1.00 | **0.997** | 2.8 |
| real records, deduplicated (n=22) | zero-shot | 100% | 45% | 0.80 | 0.69 | 0.89 | **0.79** | - |
| | fine-tuned | 100% | 100% | 0.97 | 0.88 | 0.95 | **0.935** | - |

Name and email are excluded from the real-records score because those resumes have no name or email (anonymized). `sec / resume` is batched generation on a T4 with the model in 4-bit.

> The real-records row uses the 22 resumes left after removing repeated resumes and any resume identical to a training example (section 5). On the original 40 the scores were 0.80 (zero-shot) and 0.94 (fine-tuned), so the cleanup barely moved them. The seen and unseen rows use all resumes; removing the 2 that matched a training resume gives 0.998 (n=98) and 1.000 (n=96).

**What the zero-shot model gets wrong.** It usually writes valid JSON but adds keys that aren't in the schema (for example `major`, `dates`, `achievements`), so exact-schema validity is low. Fine-tuning fixes that, and lifts the fields it was weakest on (skills, education).

---

## 2. Approach

1. **Data.** [`datasetmaster/resumes`](https://huggingface.co/datasets/datasetmaster/resumes): 4,817 nested-JSON resumes (4,803 usable). Each record is converted to a trimmed target JSON (schema above). Placeholder values ("Unknown", "N/A") are treated as missing.
2. **Rendering.** Each record is rendered to resume-like text in three layouts: `Label: value` lines, values-only joined with ` | `, and Markdown with bold labels. Layouts 0 and 1 are used for training; layout 2 is held out entirely. Section order is shuffled.
3. **Splits (by record, before rendering).** 1,000 train / 100 validation / 100 test, plus 40 anonymized real records held out of training. The test resumes are rendered twice (seen layouts and the unseen layout), so those two sets share resumes.
4. **Label check.** Every target string appears verbatim in the rendered input (2,202 of 2,202 checked), so the model is extracting, not guessing.
5. **Baseline.** Untrained Qwen2.5-1.5B-Instruct (4-bit) with the schema in the system prompt.
6. **Fine-tuning.** QLoRA: frozen 4-bit base + LoRA adapters. Loss is computed **only on the JSON answer tokens** (prompt tokens are masked with -100).
7. **Evaluation.** Same prompt and test sets, greedy decoding, up to 768 new tokens.

### Training configuration

| | |
|---|---|
| base model | Qwen/Qwen2.5-1.5B-Instruct |
| quantization | 4-bit NF4, double quantization, fp16 compute |
| LoRA | r=16, alpha=32, dropout=0.05, all attention and MLP projections |
| trainable parameters | 18,464,768 (1.18% of 1.56B) |
| batch | 2 per step × 8 accumulation = 16 effective |
| schedule | 2 epochs (126 steps), lr 2e-4, cosine, 6 warmup steps |
| optimizer / precision | paged AdamW 8-bit, fp16, gradient checkpointing |
| max sequence length | 1,536 tokens |
| hardware / time | free Colab T4, 2,600 s (43 min) of training, 8.0 GB peak GPU memory |
| loss (train / val) | step 50: 0.0028 / 0.0034 · step 100: 0.0006 / 0.0027 · end: 0.0005 (train, step 120) / 0.0026 (val, step 126) |

---

## 3. What the failure analysis found

Reading the worst outputs found problems in *my own labels* before the model's:

- **Skills labels included proficiency levels and spoken languages** ("intermediate", "English", "fluent"), so the model was penalized for correctly leaving them out. Fixed by extracting skill names only.
- **Degree labels glued fields together** (`"BSc Computer Science Node.Js"`). Fixed by splitting into `degree` (level) and `field`.
- **Education was capped at 3 entries**, so a correct fourth school was marked wrong. Cap raised to 5.
- **An institution label built from unrelated fields** (location plus accreditation body). Fixed.
- **Generation was cut off at 400 tokens**, which unfairly made the wordy zero-shot model look worse. Raised to 768.

Remaining model errors are rare. Skills recall on the synthetic sets is 0.998 with precision 1.0 (it occasionally stops one skill early). On real records the weak field is education, mostly ambiguous or noisy labels (for example institution and field swapped in the source data), plus one clear model habit: emitting a blank education entry instead of an empty list.

---

## 4. Real-records test

40 anonymized real resumes (no name or email in the data) were held out of training. After deduplication (section 5), 22 remain. Zero-shot scores **0.79** on them and the fine-tuned model **0.935** (skills 0.97, education 0.88, experience 0.95; 100% valid JSON, 100% exact schema). With n=22 the uncertainty is roughly ±5 to 8 points, so treat this as a sanity check, not a precise estimate. The gap between models is larger than that noise. These resumes are still text rendered from JSON, not parsed from real files, and the weakest field is education, where several errors trace to ambiguous or noisy source labels.

---

## 5. Data leakage / duplicates

The source data contains near-duplicate resumes (for example one copy with a name and one anonymized). Duplicates can appear inside a test set or across train and test, which inflates scores.

Resumes were compared by a fingerprint of their answer key (sorted skills, education entries, experience entries). Findings:

- Whole dataset: 4,803 usable records contain only 4,703 unique fingerprints. The anonymized real portion is far worse: 153 records, 73 unique.
- Real test set (40): 10 repeated another test resume, and 13 were identical to a training or validation resume. Only 22 unique, non-leaked resumes remained.
- Synthetic seen set (100): no repeats, and 2 matched a training resume.

Effect on scores: negligible. Real records went from 0.94 (n=40) to 0.935 (n=22); the seen set from 0.995 to 0.998 and the unseen set from 0.997 to 1.000 after removing the leaked resumes. Both the original and cleaned numbers are reported above. The fingerprint only catches exact duplicates, so near-duplicates may remain. The 22-resume set could not be enlarged, because the dataset has only 73 unique anonymized records.

---

## 6. Reproduce

1. Open `resume_json_qwen_qlora.ipynb` in Google Colab with a T4 runtime and run all cells (set `SMOKE = True` first for a 5-minute pipeline check). The full run takes about 1.5 hours including the baseline.
2. Results and per-resume predictions are written to `resume-qlora/full/` (Google Drive): `results.csv`, `baseline_results.json`, `preds_*.jsonl`, `train_log.json` and the adapter.

### Use the trained adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(base, "path/to/qwen-resume-lora-final")   # pass the base model explicitly
```
Use the same system prompt as in the notebook and the chat template (`tok.apply_chat_template(..., add_generation_prompt=True)`), then decode greedily (`do_sample=False`).

---

## 7. Files

```
resume_json_qwen_qlora.ipynb   # full pipeline, outputs kept
results.csv                    # final results table
baseline_results.json          # zero-shot summaries
preds_ft_*.jsonl               # fine-tuned per-resume outputs and scores
train_log.json                 # loss log
README.md
```

---


