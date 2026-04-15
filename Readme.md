# 🔍 Tool Retrieval with LLM Attention 

> Model: `meta-llama/Llama-3.2-1B-Instruct` · Seed: `64` · GPU: A100 (float16)

A research-grade pipeline that benchmarks **classical and neural approaches to tool retrieval** — selecting the correct API/tool from a corpus of 100 tools given a natural language query. The project culminates in a mechanistic analysis of the **"Lost in the Middle"** phenomenon in LLMs and a **Retrieval Heads** technique that recovers a ~30× improvement in Recall@1 using just 10 specialized attention heads.

---

## 📊 Key Results at a Glance

| Part | Method | Recall@1 | Recall@5 |
|------|--------|:--------:|:--------:|
| 1 | BM25 | 0.1874 | 0.3486 |
| 1 | msmarco-MiniLM | 0.3514 | 0.5716 |
| 1 | **UAE-Large-V1** | **0.6142** | **0.8650** |
| 2 | Llama-3.2 (all 512 heads) | 0.0126 | 0.2376 |
| 3 | **Retrieval Heads (K=10)** | **0.3856** | **0.6542** |
| 3 | Retrieval Heads (K=20) | 0.3194 | 0.6260 |
| 3 | Retrieval Heads (K=30) | 0.1396 | 0.5856 |

> **Headline finding:** Selecting just **10 attention heads** via MRR on 200 training queries lifts Recall@1 from `0.0126 → 0.3856` — a **~30× improvement** — and matches retrieval-trained bi-encoders like msmarco-MiniLM, using zero-shot attention signals alone.

---

## 🧠 What This Project Does

Given a query like *"Show me today's NASA picture of the day"*, the system must retrieve the correct tool (`nasa`) from a corpus of 100 API tool descriptions. Three progressively sophisticated approaches are studied:

| Part | Method | Key Idea |
|------|--------|----------|
| **Part 1** | Classical IR — BM25 + Dense Embeddings | Sparse & dense retrieval baselines |
| **Part 2** | LLM Attention-Based Retrieval | Use raw attention weights from Llama-3.2 as a ranker; visualize "Lost in the Middle" |
| **Part 3** | Retrieval Heads | Identify the most retrieval-useful heads via MRR on training data; use only those |
| **Bonus** | Effect of K on performance | Sharp quality boundary found at K=10 |

---

## 📂 Repository Structure

```
tool-retrieval-lost-in-the-middle/
│
├── data/
│   ├── tools.json              # 100 API tool descriptions (corpus)
│   ├── train_queries.json      # 1,500 training queries with gold labels
│   └── test_queries.json       # 5,000 test queries with gold labels
│
├── run1.py                     # Part 1: BM25 + Dense retrieval (MiniLM, UAE-Large-V1)
├── run2.py                     # Part 2: Attention-based retrieval + Lost-in-Middle plot
├── run3.py                     # Part 3: Retrieval Heads — head selection + evaluation
├── code3.py                    # Part 3: select_retrieval_heads() + get_query_span()
├── utils.py                    # Shared: model loading, PromptUtils, data loaders
│
├── plot2/
│   └── gold_attention_plot.png # Generated: Lost-in-the-Middle visualization
│
├── results/
│   └── q3/
│       ├── selected_heads.json # Saved top-K retrieval head indices
│       └── test_results.json   # Recall@1, Recall@5 for Part 3
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (≥ 16GB VRAM recommended — experiments ran on A100)
- HuggingFace account with access to `meta-llama/Llama-3.2-1B-Instruct`

### Install Dependencies

```bash
git clone https://github.com/<your-username>/tool-retrieval-lost-in-the-middle.git
cd tool-retrieval-lost-in-the-middle

pip install rank_bm25 sentence-transformers transformers accelerate tqdm matplotlib numpy pandas
```

### HuggingFace Authentication

```bash
huggingface-cli login
# Enter your HF token with read access to meta-llama/Llama-3.2-1B-Instruct
```

---

## ▶️ Reproducing All Experiments

### Part 1 — Classical Retrieval

```bash
python run1.py
```

Evaluates BM25, msmarco-MiniLM, and UAE-Large-V1 on 5,000 test queries:

```
============================================================
Method               Recall@1   Recall@5
------------------------------------------
BM25                   0.1874     0.3486
msmarco-MiniLM         0.3514     0.5716
UAE-Large-V1           0.6142     0.8650
============================================================
```

---

### Part 2 — Attention-Based Retrieval & Lost-in-the-Middle

```bash
python run2.py --model meta-llama/Llama-3.2-1B-Instruct --seed 64
```

- Loads Llama-3.2-1B-Instruct and extracts all 16 layers × 32 heads of attention
- Scores tools by average query→tool attention weight across all layers and heads
- Visualizes how attention to the gold tool degrades based on its **position** in the 100-tool prompt
- Saves plot to `plot2/gold_attention_plot.png`

```
Recall@1: 0.0126    Recall@5: 0.2376
```

Add `--debug` to print sample prompts and token spans for the first 5 queries.

---

### Part 3 — Retrieval Heads (vary K)

```bash
for K in 10 20 30; do
  python run3.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --seed 64 \
    --max_heads $K \
    --train_samples 200
done
```

**Phase 1** scores every `(layer, head)` pair on 200 training queries using MRR, selects top-K.  
**Phase 2** evaluates those heads on the full 5,000-query test set.

```
[K=10]  Recall@1: 0.3856    Recall@5: 0.6542
[K=20]  Recall@1: 0.3194    Recall@5: 0.6260
[K=30]  Recall@1: 0.1396    Recall@5: 0.5856
```

---

## 📉 Lost in the Middle — Visualization

![Lost in the Middle](gold_attention_plot.png)

**What the plot shows:**

- **Position 0 is a massive outlier** — the gold tool receives ~0.002 average attention when placed first, roughly **10× higher** than any middle position (~0.0001–0.0002). This is classic **primacy bias**: early tokens accumulate attention from all subsequent positions in a causal transformer.
- **Positions 1–90 are flat and suppressed** — the model largely ignores tools buried in the middle of the 100-tool context window.
- **Slight recency recovery at positions 95–99** (~0.0003–0.0005) — consistent with **recency bias** observed in long-context transformers.
- The **degree-3 polynomial trendline (red)** captures the U-shape: high start → deep trough at positions 40–55 → slight recovery at the end.

This directly explains the Recall@1 of 0.0126: the gold tool is only scored highly when it lands at position 0 (~1% of queries by chance), while for the remaining 99% it is buried and ignored.

---

## 🔬 Technical Deep Dive

### Dataset
Derived from the [MetaTool benchmark](https://arxiv.org/abs/2310.03128). 100 API tools with text descriptions; 5,000 test queries and 1,500 training queries, each with a `gold_tool_name` label.

### Part 1 — Classical IR

- **BM25** tokenizes queries and tool `(name + description)` strings and ranks via TF-IDF-style term overlap. Fails when query phrasing differs from tool description (e.g., *"make a document public"* vs. *"PDF sharing tool"*).
- **Dense retrieval** encodes queries and tools with pretrained bi-encoders and ranks by cosine similarity of L2-normalized embeddings. UAE-Large-V1's richer semantic space nearly doubles BM25's Recall@1.

**Evaluation metric:**

```
Recall@k = (1 / |Q|) * Σ 1[gold ∈ top-k(q)]
```

### Part 2 — Attention Scoring

For each query, Llama-3.2-1B produces 16 attention tensors of shape `[1, 32, N, N]`. Each tool `d` is scored as the average attention from query tokens to tool tokens, averaged over all heads and all layers. Averaging all 512 heads introduces severe noise — most heads encode syntactic/positional patterns unrelated to retrieval — collapsing Recall@1 to 0.0126.

The query span is identified via a **reverse sliding-window match** over token IDs, making it robust to variable prompt lengths.

### Part 3 — Retrieval Head Selection

For each head `(layer, head)` and training query, compute the reciprocal rank of the gold tool under that head's attention scores. The head's final score is its **Mean Reciprocal Rank (MRR)** across all 200 training queries. Select top-K heads by MRR. At test time, score tools using **only those K heads**.

MRR is preferred over mean attention score because it is rank-based and robust to scale differences across heads.

#### Top-20 Selected Heads (K=20, trained on 200 queries)

| Rank | Layer | Head | Rank | Layer | Head |
|------|-------|------|------|-------|------|
| 1 | 8 | 19 | 11 | 8 | 20 |
| 2 | 6 | 11 | 12 | 5 | 9 |
| 3 | 7 | 12 | 13 | 10 | 9 |
| 4 | 2 | 20 | 14 | 8 | 17 |
| 5 | 4 | 16 | 15 | 5 | 10 |
| 6 | 8 | 13 | 16 | 6 | 9 |
| 7 | 6 | 8 | 17 | 10 | 11 |
| 8 | 5 | 11 | 18 | 12 | 13 |
| 9 | 10 | 1 | 19 | 5 | 5 |
| 10 | 10 | 23 | 20 | 10 | 13 |

**Notable pattern:** Layers 5, 6, 8, and 10 dominate — mid-to-later layers specialize in semantic retrieval. Layer 8 alone contributes 4 of the top 20 heads (heads 19, 13, 20, 17), suggesting it is a key layer for tool-query matching. This is consistent with mechanistic interpretability findings on **induction heads** and **retrieval heads** in transformers.

### Bonus — Effect of K

Performance **monotonically decreases** as K grows from 10 → 30. Heads ranked 11–30 don't merely fail to help — they actively hurt by introducing conflicting attention patterns. This sharp quality boundary at K=10 confirms that retrieval-relevant computation in Llama-3.2 is concentrated in a tiny fraction of its 512 total heads — **quality over quantity**.

---

## 🧩 Memory Management

Parts 2 and 3 process 5,000 queries on GPU. After each query, tensors are explicitly released:

```python
del attentions, doc_scores, inputs
torch.cuda.empty_cache()
gc.collect()
```

This is essential to avoid OOM errors even on an 80GB A100 across 5,000 long-context forward passes.

---

## 💡 Concepts Demonstrated

- Sparse retrieval (BM25 / Okapi)
- Bi-encoder dense retrieval with sentence transformers
- Extracting and slicing LLM attention tensors across all layers and heads
- Reverse sliding-window token span identification in autoregressive LMs
- "Lost in the Middle" effect — primacy and recency bias quantification
- Attention head specialization — MRR-based retrieval head selection
- Mechanistic interpretability of transformer circuits

---

## 📚 References

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., 2023
- [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2406.19913) — Wu et al., 2024
- [MetaTool Benchmark](https://arxiv.org/abs/2310.03128) — Huang et al., 2023
- [UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)
- [msmarco-MiniLM-L6-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

---

## 🪪 License

For academic/educational use. Dataset derived from MetaTool; model weights subject to the Meta Llama 3 Community License.
