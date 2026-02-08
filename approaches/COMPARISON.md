# SemEval 2026 Task 5 - Approaches Comparison

## Quick Reference Summary

| Approach       | Model                     | Size  |  Key Innovation                                    |
| :------------- | :------------------------ | :---- |  :------------------------------------------------ |
| **deberta**    | DeBERTa-v3-large          | 435M  |  BCE bounded regression + Accuracy-aware soft loss |
| **deberta-a2** | DeBERTa-v3-large          | 435M  | Custom batch sampler + MSE/Spearman hybrid loss   |
| **flan-t5**    | Flan-T5-XL                | 3B    |  Regression via Classification (Expected Value)    |
| **flan-t5-xl** | Flan-T5-XL (Encoder-only) | ~1.5B | Encoder-only + Contrastive Loss                   |
| **mistral-7b** | Mistral-7B-Instruct       | 7B    | Semantic priming + Decoder regression head        |

---

## 1. Model Architecture Comparison

| Aspect                  | deberta                            | deberta-a2                     | flan-t5                         | flan-t5-xl                   | mistral-7b                         |
| :---------------------- | :--------------------------------- | :----------------------------- | :------------------------------ | :--------------------------- | :--------------------------------- |
| **Base Model**          | microsoft/deberta-v3-large         | microsoft/deberta-v3-large     | google/flan-t5-xl               | google/flan-t5-xl            | mistralai/Mistral-7B-Instruct-v0.2 |
| **Architecture Type**   | Encoder-only                       | Encoder-only                   | Seq2Seq                         | Encoder-only (adapted)       | Decoder-only                       |
| **Parameters**          | ~435M                              | ~435M                          | ~3B                             | ~3B (encoder only ~1.5B)     | ~7B                                |
| **Pre-training**        | RTD (Replaced Token Detection)     | RTD                            | Instruction-tuned (FLAN)        | Instruction-tuned (FLAN)     | Instruction-tuned                  |
| **Regression Head**     | Linear on [CLS] with Sigmoid→[1,5] | Configurable (CLS recommended) | Expected Value over token probs | MLP with masked mean pooling | AutoModelForSequenceClassification |
| **Attention Mechanism** | Disentangled attention             | Disentangled attention         | Standard Transformer            | Standard Transformer         | Grouped-Query Attention            |

---

## 2. Fine-tuning Strategy Comparison

| Aspect                         | deberta          | deberta-a2               | flan-t5                 | flan-t5-xl              | mistral-7b                   |
| :----------------------------- | :--------------- | :----------------------- | :---------------------- | :---------------------- | :--------------------------- |
| **Training Type**              | Full fine-tuning | Full / LoRA (optional)   | QLoRA (4-bit NF4)       | LoRA (PEFT)             | QLoRA (4-bit NF4)            |
| **Optimizer**                  | **Adafactor**    | AdamW                    | **Adafactor**           | **Adafactor**           | AdamW                        |
| **Learning Rate**              | 2e-6 to 2e-5     | 1e-6 to 5e-4             | 2e-4                    | 3e-4                    | 5e-5                         |
| **LLRD (Layer-wise LR Decay)** | ❌ No            | ✅ Yes (aggressive 0.56) | ❌ No                   | ❌ No                   | ❌ No                        |
| **Gradient Checkpointing**     | ✅ Yes           | ✅ Yes                   | ✅ Yes                  | ❌ N/A                  | ✅ Implicit via quantization |
| **Mixed Precision**            | FP16             | FP16                     | BF16                    | FP16                    | FP16                         |
| **LoRA Configuration**         | N/A              | r=8, α=8, dropout=0.1    | r=16, α=32, dropout=0.1 | r=32, α=64, dropout=0.2 | r=16, α=32, dropout=0.05     |
| **LoRA Target Modules**        | N/A              | Optional                 | q, v, wi_0, wi_1, wo    | q, v, k, o              | q, k, v, o, gate, up, down   |

---

## 3. Loss Function Comparison

| Aspect                    | deberta                  | deberta-a2         | flan-t5               | flan-t5-xl                | mistral-7b          |
| :------------------------ | :----------------------- | :----------------- | :-------------------- | :------------------------ | :------------------ |
| **Primary Loss**          | BCE with Logits          | MSE                | Weighted MSE          | Weighted SmoothL1         | Weighted MSE        |
| **Secondary Loss**        | Accuracy-aware soft loss | Spearman rank loss | Cross-Entropy         | Contrastive loss          | N/A                 |
| **Loss Weights**          | configurable             | w_nll + w_spearman | 0.1 CE + 0.9 wMSE     | 0.8 Reg + 0.2 Contrastive | single weighted MSE |
| **Uncertainty Weighting** | ✅ w = exp(-λσ)          | ❌ No              | ✅ w = 1/(σ + ε)      | ✅ w = exp(-λσ)           | ✅ w = 1/(σ + ε)    |
| **Uncertainty Scale (ε)** | 1.5 - 3.0 (learnable)    | configurable       | 0.5                   | 0.5                       | 0.1                 |
| **Bounded Output**        | ✅ Sigmoid → [1,5]       | ✅ Sigmoid → [1,5] | ✅ via Expected Value | ❌ No (unbounded head)    | ❌ Clipped post-hoc |

### Loss Function Mathematical Formulations

| Approach       | Formula                                                                                                          |
| :------------- | :--------------------------------------------------------------------------------------------------------------- |
| **deberta**    | $\mathcal{L} = w_i \cdot \text{BCE}(\hat{y}, y) + \alpha \cdot (1 - \sigma(k \cdot (\theta - \|y - \hat{y}\|)))$ |
| **deberta-a2** | $\mathcal{L} = w_{nll} \cdot \text{MSE} + w_{spearman} \cdot \mathcal{L}_{SpearmanRank}$                         |
| **flan-t5**    | $\mathcal{L} = 0.1 \cdot \text{CE} + 0.9 \cdot \frac{1}{N}\sum \frac{(\hat{y}_i - y_i)^2}{\sigma_i + 0.5}$       |
| **flan-t5-xl** | $\mathcal{L} = 0.8 \cdot \sum w_i \cdot \text{SmoothL1} + 0.2 \cdot \mathcal{L}_{contrastive}$                   |
| **mistral-7b** | $\mathcal{L} = \frac{1}{N}\sum \frac{(y_{pred} - y_{true})^2}{\sigma + 0.1}$                                     |

---

## 4. Input Pipeline & Tokenization

| Aspect                   | deberta                                                                                 | deberta-a2                                             | flan-t5                                                      | flan-t5-xl                                     | mistral-7b                                                               |
| :----------------------- | :-------------------------------------------------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------------- | :--------------------------------------------- | :----------------------------------------------------------------------- |
| **Max Sequence Length**  | **140** (optimized)                                                                     | 512                                                    | 1024                                                         | configurable                                   | 512                                                                      |
| **Truncation Strategy**  | **Left truncation**                                                                     | only_first                                             | standard                                                     | standard                                       | standard                                                                 |
| **Input Format**         | Single sequence                                                                         | Sequence-pair                                          | Structured prompt                                            | Structured prompt                              | Instruction format                                                       |
| **Template**             | `{homonym}: {meaning} Example: {example} Story: {precontext} {sentence} [SEP] {ending}` | `[CLS] <Meaning+Example> <Story> [SEP] <Ending> [SEP]` | Task → Scale → Story → Homonym → Sense → Constraint → Answer | Story → Target Word → Specific Sense → Example | `[INST] <Instructions> Story: <Content> Target Word: <Word> ... [/INST]` |
| **Homonym Highlighting** | ✅ UPPERCASE                                                                            | ❌ No                                                  | ❌ No                                                        | ❌ No                                          | ✅ in prompt context                                                     |
| **Prompt Repetition**    | ❌ No                                                                                   | ❌ No                                                  | ✅ Yes (bidirectional sim)                                   | ❌ No                                          | ❌ No                                                                    |
| **Semantic Priming**     | ❌ No                                                                                   | ❌ No                                                  | ✅ via instructions                                          | ✅ via instructions                            | ✅ "Expert annotator" role                                               |

---

## 5. Data Handling & Augmentation

| Aspect                      | deberta | deberta-a2                            | flan-t5          | flan-t5-xl       | mistral-7b       |
| :-------------------------- | :------ | :------------------------------------ | :--------------- | :--------------- | :--------------- |
| **Semantic Group Split**    | ✅ Yes  | ✅ Yes                                | ❌ Not mentioned | ❌ Not mentioned | ❌ Not mentioned |
| **K-Fold Cross-Validation** | ❌ No   | ✅ 5-Fold                             | ❌ No            | ❌ No            | ❌ No            |
| **Back Translation**        | ❌ No   | ✅ (but harmful)                      | ❌ No            | ❌ No            | ❌ No            |
| **Round-Robin Learning**    | ❌ No   | ✅ Available                          | ❌ No            | ❌ No            | ❌ No            |
| **Pretraining**             | ❌ No   | ✅ Available (no benefit)             | ❌ No            | ❌ No            | ❌ No            |
| **Custom Batch Sampler**    | ❌ No   | ✅ SmartUniqueGroupBatchSampler (+6%) | ❌ No            | ❌ No            | ❌ No            |
| **Auto Data Download**      | ✅ Yes  | ❌ Manual                             | ❌ Manual        | ❌ Manual        | ❌ Manual        |

---

## 6. Regularization Strategies

| Aspect              | deberta           | deberta-a2                  | flan-t5          | flan-t5-xl       | mistral-7b            |
| :------------------ | :---------------- | :-------------------------- | :--------------- | :--------------- | :-------------------- |
| **Weight Decay**    | 0.05 - 0.15       | 0.01 - 0.3                  | configurable     | configurable     | implicit in AdamW     |
| **Dropout (Head)**  | via model default | 0.01 - 0.3                  | 0.1 (LoRA)       | 0.2 (LoRA)       | 0.05 (LoRA)           |
| **Early Stopping**  | ✅ Patience=4     | ❌ Not mentioned            | ❌ Not mentioned | ❌ Not mentioned | ✅ on validation loss |
| **Layer Freezing**  | ❌ No             | ✅ Optional (frozen epochs) | ✅ via LoRA      | ✅ via LoRA      | ✅ via LoRA           |
| **Label Smoothing** | ❌ No             | ❌ No                       | ❌ No            | ❌ No            | ❌ No                 |

---

## 7. Inference Optimization

| Aspect                      | deberta         | deberta-a2       | flan-t5                 | flan-t5-xl      | mistral-7b                  |
| :-------------------------- | :-------------- | :--------------- | :---------------------- | :-------------- | :-------------------------- |
| **Prediction Clipping**     | ✅ [1.99, 4.01] | ❌ Not mentioned | ❌ Not mentioned        | ✅ [1.99, 4.01] | ✅ [1.99, 4.01]             |
| **Calibration**             | ❌ No           | ❌ No            | ❌ No                   | ❌ No           | ✅ Linear regression on dev |
| **Ensemble**                | ❌ No           | ✅ Available     | ❌ No                   | ❌ No           | ❌ No                       |
| **Expected Value Decoding** | ❌ N/A          | ❌ N/A           | ✅ Yes (key innovation) | ❌ N/A          | ❌ N/A                      |

### Metric Exploitation Strategy

All approaches exploit the competition metric: prediction correct if `|pred - true| < max(σ, 1.0)`

By clipping to `[1.99, 4.01]`:

- Prediction **1.99** for truth **1.0**: `|1.99 - 1.0| = 0.99 < 1.0` ✅
- Prediction **1.99** for truth **2.5**: `|1.99 - 2.5| = 0.51 < 1.0` ✅

---

## 8. Hardware Requirements

| Aspect               | deberta                  | deberta-a2         | flan-t5     | flan-t5-xl   | mistral-7b |
| :------------------- | :----------------------- | :----------------- | :---------- | :----------- | :--------- |
| **Min VRAM**         | **6GB** (GTX 1060)       | 8GB+ (with LoRA)   | 16GB (T4)   | 40GB (A100)  | 16GB (T4)  |
| **Recommended VRAM** | 24GB (RTX 3090/4090)     | 24GB               | 40GB (A100) | 40GB (A100)  | 24GB+      |
| **Quantization**     | ❌ None (full precision) | ❌ None            | 4-bit NF4   | ❌ None      | 4-bit NF4  |
| **Training Time**    | Fast                     | Moderate           | Moderate    | Moderate     | Moderate   |
| **Colab Compatible** | ✅ Yes                   | ✅ Yes (with LoRA) | ✅ T4/A100  | ✅ A100 only | ✅ T4      |

---

## 9. Hyperparameter Optimization

| Aspect        | deberta                     | deberta-a2        | flan-t5                | flan-t5-xl | mistral-7b |
| :------------ | :-------------------------- | :---------------- | :--------------------- | :--------- | :--------- |
| **Framework** | Optuna (Bayesian)           | Optuna (Bayesian) | ❌ Manual              | ❌ Manual  | ❌ Manual  |
| **Objective** | 0.2 Spearman + 0.8 Soft Acc | configurable      | 0.7 Acc + 0.3 Spearman | N/A        | N/A        |
| **Trials**    | tunable                     | up to 200         | N/A                    | N/A        | N/A        |
| **Pruning**   | ❌ Optional                 | ❌ Optional       | N/A                    | N/A        | N/A        |

---

## 10. Results Comparison

| Approach           | Accuracy (within SD) | Spearman ρ | p-Value   | Dataset          |
| :----------------- | :------------------- | :--------- | :-------- | :--------------- |
| **deberta**        | 0.7957 (740/930)     | 0.6866     | 1.23e-130 | Test             |
| **deberta-a2**     | N/A (see ablations)  | N/A        | N/A       | N/A              |
| **flan-t5 (v6.0)** | **0.8107**           | 0.6841     | 2.44e-129 | Test (codabench) |
| **flan-t5-xl**     | 0.8452               | **0.7241** | N/A       | Dev              |
| **mistral-7b**     | **0.8570** (797/930) | **0.7623** | 1.44e-177 | Test             |

### Winner Analysis

- **Best Accuracy**: `mistral-7b` (0.8570)
- **Best Spearman**: `mistral-7b` (0.7623)
- **Best Efficiency**: `deberta` (runs on 6GB GPU)
- **Best Research Clarity**: `deberta` (extensive documentation)

---

## 11. Failed Experiments & Negative Results

| Approach       | Experiment                                | Outcome                         | Lesson                                                                    |
| :------------- | :---------------------------------------- | :------------------------------ | :------------------------------------------------------------------------ |
| **deberta-a2** | Back Translation                          | ❌ Increased noise, overfitting | Synthetic paraphrases introduce semantic drift                            |
| **deberta-a2** | NLI-pretrained checkpoints                | ❌ No benefit                   | NLI doesn't transfer to plausibility regression                           |
| **deberta-a2** | Pretraining on synthetic data             | ❌ No improvement               | Distant supervision quality insufficient                                  |
| **deberta-a2** | Alternative pooling (LSTM, GRU, Weighted) | ❌ Worse than CLS               | Pretrained [CLS] already optimal for this task                            |
| **flan-t5**    | KL-Divergence loss                        | ❌ Model confused               | Conflicting gradients between distribution matching and mean optimization |
| **flan-t5**    | Flan-T5-Large                             | ❌ Insufficient capacity        | XL needed for semantic nuances                                            |

---

## 12. Key Innovations by Approach

### deberta

1. **Left truncation** preserving critical ending information
2. **BCE as bounded regression loss** for numerical stability
3. **Accuracy-aware soft loss** directly optimizing competition metric
4. **Runs on consumer hardware** (6GB VRAM)

### deberta-a2

1. **SmartUniqueGroupBatchSampler** (+6% improvement)
2. **Aggressive LLRD** (0.56 decay) with high LR (+5% improvement)
3. **Comprehensive ablation studies** documenting failed approaches

### flan-t5

1. **Regression via Classification** using Expected Value
2. **Prompt Repetition** for simulated bidirectional attention
3. **Comprehensive version tracking** showing iteration improvements

### flan-t5-xl

1. **Encoder-only T5** reducing memory by ~50%
2. **Contrastive Loss** for embedding space structure
3. **Deep MLP head** with masked mean pooling

### mistral-7b

1. **Semantic priming** via "expert annotator" persona
2. **Calibration step** on dev set before inference
3. **Crash & Recovery training** strategy (Ordering before Scaling)
4. **Highest accuracy** among all approaches (0.8570)

---

## 14. Common Design Patterns Across Approaches

1. **Uncertainty Weighting**: Most approaches weight samples inversely to annotator disagreement (except `deberta-a2`)
2. **Bounded Outputs**: Most approaches constrain predictions to [1,5] range
3. **Inference Clipping**: Multiple approaches use [1.99, 4.01] clipping to exploit metrics
4. **Memory Optimization**: All use some form of gradient checkpointing, LoRA, or quantization
5. **Semantic Grouping**: Encoder-based approaches use semantic splits to prevent leakage
