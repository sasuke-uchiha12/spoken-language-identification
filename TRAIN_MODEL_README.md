# `train_model.py` Detailed Code Walkthrough

This README explains `train_model.py` as a real code walkthrough.

It is organized as:
1. Overall pipeline logic
2. Class-by-class and function-by-function breakdown
3. Artifact outputs and practical checks

---

## 1) Overall Pipeline Logic

`train_model.py` is a complete training + evaluation + diagnostics pipeline for spoken language identification.

Default profile in this file:
- Model backbone: `utter-project/mHuBERT-147`
- Dataset: `badrex/nnti-dataset-full`
- DANN: enabled by default
- Train augmentation: disabled by default
- Best-checkpoint metric: `macro_f1`
- Diagnostics: per-language, confusion matrix, speaker diagnostics, t-SNE

### End-to-end execution order (`main()`)

```python
cfg = CFG
set_all_seeds(cfg.seed)
maybe_login_hf(cfg)
wandb_enabled = maybe_setup_wandb(cfg)

feature_extractor = AutoFeatureExtractor.from_pretrained(...)
dataset = load_dataset(cfg.dataset_name)

train_ds_encoded = train_ds.map(train_preprocess, ...)
valid_ds_encoded = valid_ds.map(valid_preprocess, ...)

model = DANNForAudioClassification(...) or AutoModelForAudioClassification(...)
trainer = build_trainer(...)

train_result = trainer.train()
final_eval_metrics = trainer.evaluate(...)
pred_output = trainer.predict(...)

save per-language / confusion / speaker diagnostics / t-SNE
```

What this means in plain engineering terms:
1. Prepare deterministic run settings and optional logging integrations.
2. Load dataset and convert raw audio into model-ready tensors.
3. Train model, evaluate, predict.
4. Export analysis artifacts used in reporting.

---

## 2) Detailed Breakdown: Class and Function by Function

Below is the file order, with code snippets and part-by-part explanation.

## `TrainConfig` (dataclass)

### Code shape
```python
@dataclass
class TrainConfig:
    model_id: str = "utter-project/mHuBERT-147"
    dataset_name: str = "badrex/nnti-dataset-full"
    ...
    enable_dann: bool = True
    dann_speaker_loss_weight: float = 0.1
    ...
```

### What each part does
1. Model/data fields choose backbone and dataset.
2. Training fields define batch sizes, epochs, LR, warmup, save/eval cadence.
3. Augmentation fields define probability + speed/noise ranges.
4. Diagnostics fields control confusion/per-language/speaker/t-SNE exports.
5. DANN fields control adversarial speaker head behavior.
6. `timestamp` is auto-created and used in run naming.

---

## `_default_run_name(cfg)`

### Code
```python
model_slug = cfg.model_id.replace("/", "-")
return f"SLID_{model_slug}_{cfg.learning_rate}_{cfg.timestamp}"
```

### Explanation
1. Converts model ID into filesystem-safe format.
2. Creates a unique, traceable run name.

---

## `slugify_label(label)`

### Code
```python
return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
```

### Explanation
1. Normalizes text to lowercase.
2. Replaces non-alphanumeric sequences with underscore.
3. Removes leading/trailing underscores.
4. Used for metric keys like `acc_lang_konkani`.

---

## `set_all_seeds(seed)`

### Code
```python
set_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Explanation
1. Aligns randomness in Transformers, NumPy, and PyTorch.
2. CUDA seed branch keeps behavior consistent on GPU.

---

## `print_device_info()`

### Code
```python
cuda_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {cuda_available}")
```

### Explanation
1. Prints runtime hardware context.
2. Helps later debugging when runs are compared.

---

## `maybe_login_hf(cfg)`

### Code
```python
token = os.getenv("HF_TOKEN")
if token and hf_login is not None:
    hf_login(token=token)
```

### Explanation
1. Optional login for Hub rate limits/private access.
2. Safely skips when token/package is missing.

---

## `maybe_setup_wandb(cfg)`

### Code
```python
if not cfg.report_to_wandb: return False
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)
wandb.init(project=..., name=..., config=asdict(cfg))
```

### Explanation
1. Enables W&B only when fully configured.
2. Logs config for reproducibility.
3. Returns a boolean to decide `report_to` mode.

---

## `infer_input_features_key(model_id)`

### Code
```python
if model_id == "facebook/w2v-bert-2.0":
    return "input_features"
return "input_values"
```

### Explanation
1. Handles model-specific input interface differences.
2. Prevents feature-key mismatch in preprocessing/collation.

---

## `AudioDataCollator`

## `__init__(feature_extractor, input_features_key)`
Stores extractor and expected key.

## `__call__(features)`

### Code
```python
batch = {
    self.input_features_key: [...],
    "attention_mask": [...],
}
batch = self.feature_extractor.pad(...)
batch["labels"] = torch.tensor(...)
if "speaker_labels" in features[0]:
    batch["speaker_labels"] = torch.tensor(...)
```

### Explanation
1. Collects variable-length sequences from examples.
2. Pads them to uniform length.
3. Adds language labels for supervised objective.
4. Adds speaker labels only if present (DANN path).

---

## `GradientReversalFunction`

## `forward(ctx, x, lambda_)`
Stores `lambda_`, returns `x` unchanged.

## `backward(ctx, grad_output)`
Returns `-lambda_ * grad_output`.

### Why this matters
- Forward pass behaves like identity.
- Backward pass flips/weights gradient to enforce adversarial invariance.

---

## `grad_reverse(x, lambda_)`

Thin wrapper around `GradientReversalFunction.apply`.

---

## `DANNForAudioClassification`

### `__init__(...)`

### Code
```python
self.encoder = AutoModel.from_pretrained(model_id)
self.language_classifier = nn.LazyLinear(num_labels)
self.speaker_classifier = nn.Sequential(
    nn.Dropout(...),
    nn.LazyLinear(num_speakers),
)
```

### Explanation
1. Shared encoder extracts speech representation.
2. Language head predicts language class.
3. Speaker head predicts speaker class (adversarial branch).
4. `LazyLinear` adapts input dimension automatically after first forward.

## `_pool(sequence_output, attention_mask)`

### Code
```python
if attention_mask is None:
    return sequence_output.mean(dim=1)
masked = sequence_output * mask.unsqueeze(-1)
return masked.sum(dim=1) / denom
```

### Explanation
1. Converts frame-level outputs into one vector/sample.
2. Uses mask-aware mean when mask shape is compatible.
3. Falls back to plain mean when mask is unavailable/mismatched.

## `forward(...)`

### Code
```python
encoder_outputs = self.encoder(**encoder_inputs)
sequence_output = encoder_outputs.last_hidden_state or encoder_outputs[0]
pooled = self._pool(sequence_output, attention_mask)
language_logits = self.language_classifier(pooled)
speaker_logits = self.speaker_classifier(grad_reverse(pooled, grl_lambda))
```

### Explanation
1. Builds encoder inputs for either `input_values` or `input_features`.
2. Runs encoder with compatibility fallback for `attention_mask` support.
3. Pools sequence output.
4. Produces two logits:
   - language logits (main task)
   - speaker logits through gradient reversal (adversarial task)

---

## `DANNTrainer`

## `__init__(...)`
Stores DANN hyperparameters.

## `_current_grl_lambda()`

### Code
```python
progress = global_step / max_steps
schedule = 2/(1+exp(-10*progress)) - 1
return base_lambda * schedule
```

### Explanation
1. If scheduling is disabled, uses fixed lambda.
2. If enabled, ramps adversarial strength during training.
3. Avoids strong adversarial pressure at initial unstable phase.

## `compute_loss(model, inputs, ...)`

### Code
```python
language_loss = F.cross_entropy(language_logits, labels)
total_loss = language_loss

if model.training and speaker_labels is not None:
    valid = speaker_labels >= 0
    speaker_loss = F.cross_entropy(speaker_logits[valid], speaker_labels[valid])
    total_loss = language_loss + weight * speaker_loss
```

### Explanation
1. Always computes language classification loss.
2. Computes speaker loss only in train mode and only for valid speaker labels.
3. Ignores invalid labels (`-100`) to prevent invalid CE targets.
4. Combines losses using configured speaker-loss weight.
5. Optionally returns detached component losses for logging.

---

## `WaveformAugmenter`

## `__init__(cfg, sample_rate)`
- Loads `torchaudio` only if augmentation is enabled.
- Raises clear import error if augmentation requested without torchaudio.

## `maybe_augment(waveform, sample_index)`

### Code
```python
rng = np.random.default_rng(seed + sample_index)
if rng.random() > augmentation_prob:
    return waveform

speed = rng.uniform(speed_min, speed_max)
resample(...)

noise_std = rng.uniform(noise_std_min, noise_std_max)
x = clip(x + noise)
```

### Explanation
1. Uses deterministic per-sample random state for reproducibility.
2. Applies a global augmentation gate by probability.
3. Performs speed perturbation via resampling.
4. Adds Gaussian waveform noise and clips to valid amplitude range.

---

## `make_preprocess_function(...)`

### Returned function: `preprocess_function(examples, indices=None)`

### Code
```python
audio_np = np.asarray(audio_obj["array"], dtype=np.float32)
if apply_augmentation: audio_np = augmenter.maybe_augment(...)

inputs = feature_extractor(audio_arrays, ...)
inputs["label"] = [str_to_int[x] for x in examples["language"]]
inputs["speaker_labels"] = [...]
inputs["length"] = [len(f) for f in inputs[input_features_key]]
```

### Explanation
1. Converts each decoded audio into float32 numpy waveform.
2. Optionally applies train-only augmentation.
3. Extracts model features with truncation at max duration.
4. Attaches language labels and optional speaker labels.
5. Saves `length` for optional length-based batching.

---

## `make_compute_metrics(int_to_str)`

### Returned function: `compute_metrics(eval_pred)`

### Code
```python
preds = np.argmax(logits, axis=1)
metrics = {
    "accuracy": accuracy_score(...),
    "macro_f1": f1_score(..., average="macro"),
}
for each label_id:
    metrics[f"acc_lang_{slug}"] = per-class accuracy
```

### Explanation
1. Converts logits to predicted class IDs.
2. Computes global metrics.
3. Computes per-language accuracy for diagnostics and reports.

---

## `pool_sequence_representation(last_hidden_state, attention_mask)`

### Core logic
- If sequence tensor is not 3D, return as-is.
- If no valid mask, mean over time axis.
- Else masked mean over time axis.

### Why it exists
- Reusable pooling helper for t-SNE extraction path.

---

## `extract_last_layer_representations(...)`

### Code
```python
for batch in dataloader:
    remove labels/speaker_labels
    move to model device

    if DANN model:
        run model.encoder(..., output_hidden_states=True)
    else:
        run model(..., output_hidden_states=True)

    last_hidden = ...
    pooled = ...
    all_repr.append(pooled.cpu().numpy())
```

### Explanation
1. Iterates encoded dataset without gradients.
2. Supports both DANN and non-DANN model structures.
3. Gets last-layer hidden states robustly across output formats.
4. Pools to one vector/sample.
5. Returns stacked representation matrix used by t-SNE.

---

## `sample_indices_for_tsne(languages, max_samples, seed)`

### Code
```python
if n <= max_samples: return all indices

# pass 1: stratified-ish quota per language
for label in unique_languages:
    pick min(len(label_idx), quota)

# pass 2: random fill from remaining pool
if len(selected) < max_samples:
    pick extras
```

### Explanation
1. Limits t-SNE size for runtime/memory.
2. Avoids domination by high-frequency languages.
3. Fills leftover slots randomly for better coverage.

---

## `run_tsne_and_save(...)`

### Part A: prepare sampled data
```python
indices = sample_indices_for_tsne(...)
reps = representations[indices]
langs = ...
speakers = ...
```

### Part B: run t-SNE
```python
perplexity = min(cfg.tsne_perplexity, max_perplexity)
tsne = TSNE(...)
coords = tsne.fit_transform(reps)
```

### Part C: save points CSV
```python
rows = [{"x": ..., "y": ..., "language": ..., "speaker_id": ...}, ...]
write_rows_csv(...)
```

### Part D: save language-colored plot
- One scatter color per language.
- Saves `tsne_validation_by_language.png`.

### Part E: save speaker-colored plot
- Highlights top-k speakers by sample count.
- Others are plotted as gray background.
- Saves `tsne_validation_by_speaker_topk.png`.

---

## `write_confusion_matrix_csv(path, cm, label_names)`

### Code
```python
writer.writerow(["true\\pred"] + label_names)
for label, row in zip(label_names, cm.tolist()):
    writer.writerow([label] + row)
```

### Explanation
- Stores confusion matrix in readable tabular form.

---

## `plot_confusion_matrix(...)`

### Core steps
1. Create dynamic figure size based on number of labels.
2. Draw heatmap with axis labels.
3. Save PNG.

### Purpose
- Visual inspection of dominant confusion patterns.

---

## `compute_per_language_rows(preds, refs, int_to_str)`

### Code
```python
for each label_id:
    mask = refs == label_id
    n_samples = sum(mask)
    n_correct = sum(preds[mask] == refs[mask])
    accuracy = n_correct / n_samples
```

### Explanation
- Builds table rows for per-language performance diagnostics.

---

## `write_rows_csv(path, rows, fieldnames)`

Generic CSV utility used by multiple diagnostics.

---

## `run_speaker_diagnostic(...)`

### Part A: aggregate by speaker
```python
for each validation sample:
    update speaker sample count and correct count
```

### Part B: write per-speaker CSV
- Contains sample count, accuracy, number of languages, seen-in-train flag.

### Part C: overlap and seen/unseen analysis
```python
overlap_speakers = train_speaker_ids & valid_speaker_set
seen_mask = [speaker in train_speaker_ids]
unseen_mask = ~seen_mask
seen_accuracy = ...
unseen_accuracy = ...
```

### Part D: warning rules
- Warn if overlap exists.
- Warn if overlap ratio too high.
- Warn if seen-vs-unseen gap is too large.

### Part E: write summary JSON
- Saves aggregate statistics and warnings.

---

## `build_training_arguments(cfg, output_dir, report_to)`

### Code skeleton
```python
kwargs = {
    "output_dir": ...,
    "eval_steps": cfg.eval_steps,
    "save_steps": cfg.save_steps,
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
    ...
}

# compatibility guards
if "evaluation_strategy" in signature: ... else: ...
if "label_names" in signature: kwargs["label_names"] = ["labels"]
```

### Explanation
1. Defines core HF Trainer behavior.
2. Uses macro-F1 for best checkpoint selection.
3. Handles differences between transformers versions safely.
4. Ensures evaluation metrics use language labels.

---

## `build_trainer(...)`

### Code
```python
trainer_cls = DANNTrainer if cfg.enable_dann else Trainer
if cfg.enable_dann:
    pass DANN hyperparameters
return trainer_cls(**kwargs)
```

### Explanation
1. Picks correct trainer implementation.
2. Injects DANN-specific arguments only when needed.

---

## `initialize_lazy_parameters_with_sample(...)`

### Code
```python
has_uninitialized = any(isinstance(p, UninitializedParameter) for p in model.parameters())
if has_uninitialized:
    batch = data_collator([dataset_encoded[0]])
    remove labels
    model(**batch)
```

### Explanation
1. Required because DANN heads use `LazyLinear`.
2. Prevents Trainer failure when counting trainable parameters before first forward.

---

## `main()`

`main()` is split with explicit stage comments in code.

### Stage 1: Run setup
- Resolve run name
- Print timestamp and device
- Seed everything
- Optional HF/W&B setup

### Stage 2: Load feature extractor + dataset
- Load extractor from model ID
- Load dataset splits
- Shuffle and cast audio column sample rate

### Stage 3: Label mapping
- Build language list and `str_to_int` mapping
- If DANN is enabled, build train speaker mapping

### Stage 4: Preprocessing functions
- Build train preprocessing function (optionally with augmentation)
- Build validation preprocessing function (always no augmentation)
- Encode train and validation sets via `map()`

### Stage 5: Model + trainer
- Build model config and model instance
- Create data collator
- Initialize lazy params if needed
- Build training args and trainer

### Stage 6: Train and save
- `trainer.train()`
- Save model, feature extractor, state, train metrics

### Stage 7: Final eval + predict
- Evaluate and save eval metrics
- Predict validation and save predict metrics

### Stage 8: Diagnostics export
- Per-language accuracy CSV
- Confusion matrix CSV + PNG
- Speaker diagnostic CSV + JSON
- t-SNE outputs (CSV + two plots), wrapped in safe try/except

### Stage 9: Cleanup
- Finish W&B run if enabled
- Print output artifact path

---

## 3) Files Produced by the Script

Inside each run directory (`<output_root>/<run_name>/`):
- Training outputs: model files, checkpoints, trainer state
- Metrics: `train_results.json`, `eval_results.json`, `predict_results.json`, `all_results.json`
- Diagnostics:
  - `diagnostics/per_language_accuracy_validation.csv`
  - `diagnostics/confusion_matrix_validation.csv`
  - `diagnostics/confusion_matrix_validation.png`
  - `diagnostics/speaker_accuracy_validation.csv`
  - `diagnostics/speaker_diagnostic_summary.json`
  - `diagnostics/tsne_validation_points.csv`
  - `diagnostics/tsne_validation_by_language.png`
  - `diagnostics/tsne_validation_by_speaker_topk.png`

---

## 4) Practical Notes for Reporting

1. `metric_for_best_model` is `macro_f1`, so selected checkpoint is macro-F1 best checkpoint.
2. Accuracy/loss/F1 reported should come from that same selected checkpoint.
3. DANN uses speaker labels only in training loss; evaluation metrics remain language-based.
4. t-SNE export is diagnostic; if it fails, training/evaluation still complete.
