import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # remove this line when downloading fresh
import argparse
import json
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items


# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_to_docs_attention(attentions, query_span, doc_spans, debug=False):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    device = attentions[0].device
    num_layers = len(attentions)
    doc_scores = torch.zeros(len(doc_spans), device=device)

    q_start, q_end = query_span
    N = attentions[0].shape[2]
    
    if debug:
        print(f"\n[attention_debug] query_span: {query_span}, N (seq_len): {N}")
        print(f"[attention_debug] num_layers: {num_layers}, num_docs: {len(doc_spans)}")
        print(f"[attention_debug] attentions tuple length: {len(attentions)}")
        print(f"[attention_debug] attentions[0] shape: {attentions[0].shape}")
    
    query_mask = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
    query_mask = (query_mask >= q_start) & (query_mask < q_end)  # [1, N]
    
    if debug:
        print(f"[attention_debug] query_mask shape: {query_mask.shape}")
        print(f"[attention_debug] query_mask sum (num_query_tokens): {query_mask.sum().item()}")
        print(f"[attention_debug] query_mask: {query_mask}")

    for layer_idx in range(num_layers):
        A = attentions[layer_idx][0]  # [heads, N, N]
        A_mean = A.mean(dim=0)  # [N, N]  mean over heads

        if debug and layer_idx == 0:
            print(f"[attention_debug] Layer {layer_idx}: A shape: {A.shape}")
            print(f"[attention_debug] Layer {layer_idx}: A_mean shape: {A_mean.shape}")
            print(f"[attention_debug] Layer {layer_idx}: A_mean min/max: {A_mean.min():.6f} / {A_mean.max():.6f}")
            print(f"[attention_debug] Layer {layer_idx}: A_mean sum: {A_mean.sum():.6f}")

        # attention from query tokens to the whole sequence
        q2ctx = A_mean[query_mask[0], :]  # [num_query_toks, N]
        
        if debug and layer_idx == 0:
            print(f"[attention_debug] Layer {layer_idx}: q2ctx (before mean) shape: {q2ctx.shape}")
            print(f"[attention_debug] Layer {layer_idx}: q2ctx (before mean) min/max: {q2ctx.min():.6f} / {q2ctx.max():.6f}")
        
        q2ctx = q2ctx.mean(dim=0, keepdim=True)  # [1, N]
        
        if debug and layer_idx == 0:
            print(f"[attention_debug] Layer {layer_idx}: q2ctx (after mean) shape: {q2ctx.shape}")
            print(f"[attention_debug] Layer {layer_idx}: q2ctx (after mean) min/max: {q2ctx.min():.6f} / {q2ctx.max():.6f}")
            print(f"[attention_debug] Layer {layer_idx}: q2ctx (after mean) sum: {q2ctx.sum():.6f}")

        for doc_idx, (d_start, d_end) in enumerate(doc_spans):
            doc_contrib = q2ctx[0, d_start:d_end].sum()  # scalar
            
            if debug and layer_idx == 0 and doc_idx < 3:
                doc_span_vals = q2ctx[0, d_start:d_end]
                print(f"[attention_debug] Layer {layer_idx}: Doc {doc_idx} span [{d_start}:{d_end}], values shape: {doc_span_vals.shape}")
                print(f"[attention_debug] Layer {layer_idx}: Doc {doc_idx} contribution: {doc_contrib.item():.6f}")
            
            doc_scores[doc_idx] += doc_contrib

    if debug:
        print(f"[attention_debug] doc_scores (before avg): min/max: {doc_scores.min():.6f} / {doc_scores.max():.6f}")
        print(f"[attention_debug] doc_scores (before avg) sum: {doc_scores.sum():.6f}")
    
    doc_scores /= num_layers  # average over layers
    
    if debug:
        print(f"[attention_debug] doc_scores (after avg): min/max: {doc_scores.min():.6f} / {doc_scores.max():.6f}")
        print(f"[attention_debug] doc_scores (after avg) sum: {doc_scores.sum():.6f}")
        print(f"[attention_debug] top 5 doc_scores: {torch.topk(doc_scores, min(5, len(doc_scores))).values}")
    
    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    """
    result: list of dicts with keys: gold_position, gold_score, gold_rank
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    positions = [r["gold_position"] for r in result]
    scores = [r["gold_score"].item() if hasattr(r["gold_score"], "item") else r["gold_score"]
              for r in result]
    ranks = [r["gold_rank"] for r in result]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. attention score vs position
    axes[0].scatter(positions, scores, c="blue", alpha=0.6)
    axes[0].set_xlabel("Gold tool position in prompt")
    axes[0].set_ylabel("Query → Gold tool attention score")
    axes[0].set_title("Attention vs. Position")

    # 2. rank vs position
    axes[1].scatter(positions, ranks, c="green", alpha=0.6)
    axes[1].set_xlabel("Gold tool position in prompt")
    axes[1].set_ylabel("Gold tool rank (lower = better)")
    axes[1].set_title("Rank vs. Position")

    # 3. distribution of ranks
    axes[2].hist(ranks, bins=30, color="orange", alpha=0.7)
    axes[2].set_xlabel("Gold tool rank")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of gold tool ranks")

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def get_query_span(inputs, question, tokenizer, debug=False):
    """
    Identifies the token span corresponding to the query in the prompt.
    Searches backwards since the query is near the end of the prompt.
    """
    input_ids = inputs.input_ids[0].cpu().tolist()
    query_ids = tokenizer(question, add_special_tokens=False).input_ids
    
    if debug:
        print(f"\n[query_span_debug] input_ids length: {len(input_ids)}")
        print(f"[query_span_debug] question: '{question}'")
        print(f"[query_span_debug] query_ids: {query_ids}")
        print(f"[query_span_debug] query_ids length: {len(query_ids)}")
    
    query_len = len(query_ids)
    
    # Search for the exact sequence of query tokens in the input_ids
    # Searching backwards because the question is near the end
    for i in range(len(input_ids) - query_len, -1, -1):
        if input_ids[i:i+query_len] == query_ids:
            if debug:
                print(f"[query_span_debug] Found query match at indices [{i}:{i+query_len}]")
                print(f"[query_span_debug] Matched tokens: {tokenizer.decode(input_ids[i:i+query_len])}")
            return (i, i + query_len)
    
    # Fallback if exactly matching fails
    if debug:
        print(f"[query_span_debug] No exact match found! Using fallback span.")
        print(f"[query_span_debug] Fallback span: [{len(input_ids) - query_len}:{len(input_ids)}]")
    return (len(input_ids) - query_len, len(input_ids))


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"

    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads // model.config.num_key_value_heads
    softmax_scaling = d**-0.5

    train_queries, test_queries, tools = get_queries_and_items()

    print("\n" + "="*60)
    print("---- COMPREHENSIVE DEBUG INFO ----")
    print("="*60)
    print(f"seed: {args.seed}, model: {model_name}")
    print(f"device: {device}")
    print(f"model.config._attn_implementation: {model.config._attn_implementation}")
    print(f"num_heads: {num_heads}")
    print(f"num_layers: {num_layers}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"head_dim (d): {d}")
    print(f"num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"num_key_value_groups: {num_key_value_groups}")
    print(f"softmax_scaling (d^-0.5): {softmax_scaling}")
    print(f"num_test_queries: {len(test_queries)}")
    print(f"num_tools: {len(tools)}")
    print("="*60 + "\n")

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []

    recall_at_1 = 0.0
    recall_at_5 = 0.0
    total = 0

    for qix in tqdm(range(len(test_queries))):
        sample = test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]
        
        is_debug_sample = args.debug and qix < 5
        
        if is_debug_sample:
            print(f"\n{'*'*60}")
            print(f"SAMPLE {qix}: qid={qid}")
            print(f"{'*'*60}")

        # -------------------- Do NOT change the shuffling here --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v: k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)

        gold_tool_id = map_docname_id[gold_tool_name]
        
        if is_debug_sample:
            print(f"\n[sample_debug] qid: {qid}")
            print(f"[sample_debug] question: {question}")
            print(f"[sample_debug] gold_tool_name: {gold_tool_name}")
            print(f"[sample_debug] gold_tool_id: {gold_tool_id}")
            print(f"[sample_debug] num_docs: {len(item_spans)}")
            print(f"[sample_debug] doc_spans first 3: {item_spans[:3]}")
            print(f"[sample_debug] doc spans: {item_spans}")

        prompt = putils.create_prompt(query=question)
        
        if is_debug_sample:
            print(f"\n[tokenizer_debug] FULL PROMPT LENGTH (chars): {len(prompt)}")
            print(f"[tokenizer_debug] First 300 chars of prompt:")
            print(repr(prompt[:300]))
            print(f"\n[tokenizer_debug] Last 300 chars of prompt:")
            print(repr(prompt[-300:]))
            
            # Test tokenizer on special tokens first
            test_header = '<|start_header_id|>'
            test_header_ids = tokenizer(test_header, add_special_tokens=False).input_ids
            print(f"\n[tokenizer_debug] Testing special token '<|start_header_id|>': {test_header_ids}")
        
        # Tokenize WITHOUT truncation
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
        
        if is_debug_sample:
            ip_ids = inputs.input_ids[0].cpu()
            print(f"\n[tokenizer_debug] After tokenization:")
            print(f"[tokenizer_debug] input_ids shape: {ip_ids.shape}")
            print(f"[tokenizer_debug] input_ids length: {len(ip_ids)}")
            print(f"[tokenizer_debug] First 30 token IDs: {ip_ids[:30].tolist()}")
            print(f"[tokenizer_debug] Last 20 token IDs: {ip_ids[-20:].tolist()}")
            if len(ip_ids) > 0:
                print(f"[tokenizer_debug] Decoded first 150 chars: {repr(tokenizer.decode(ip_ids[:100])[:150])}")
                print(f"[tokenizer_debug] Decoded last 100 chars: {repr(tokenizer.decode(ip_ids[-50:]))}")
            print(f"\n[sample_debug] input_ids shape: {ip_ids.shape}")
            print(f"[sample_debug] input_ids length: {len(ip_ids)}")
            print("\n--- PROMPT SNIPPET ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("--- END PROMPT ---\n")
            if len(ip_ids) >= max(max(s) for s in item_spans) if item_spans else False:
                print("---- doc1 ----")
                print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]])[:200])
                print("---- lastdoc ----")
                print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]])[:200])
            else:
                print(f"[ERROR] item_spans exceed input_ids length! Max span: {max(max(s) for s in item_spans)}, input length: {len(ip_ids)}")

        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions  # tuple of [1, heads, N, N]
        
        if is_debug_sample:
            print(f"\n[sample_debug] attentions type: {type(attentions)}")
            print(f"[sample_debug] attentions length: {len(attentions)}")
            if len(attentions) > 0:
                print(f"[sample_debug] attentions[0] shape: {attentions[0].shape}")
                print(f"[sample_debug] attentions[0] dtype: {attentions[0].dtype}")
                print(f"[sample_debug] attentions[0] min/max: {attentions[0].min():.6f} / {attentions[0].max():.6f}")

        # Get query span
        query_span = get_query_span(inputs, question, tokenizer, debug=is_debug_sample)

        # Compute query -> doc attention scores
        doc_scores = query_to_docs_attention(attentions, query_span, item_spans, debug=is_debug_sample)
        
        if is_debug_sample:
            print(f"\n[sample_debug] FINAL doc_scores shape: {doc_scores.shape}")
            print(f"[sample_debug] FINAL doc_scores: {doc_scores}")
            print(f"[sample_debug] FINAL doc_scores min/max: {doc_scores.min():.6f} / {doc_scores.max():.6f}")

        # Rank documents by score
        _, indices = torch.sort(doc_scores, descending=True)
        
        if is_debug_sample:
            print(f"\n[ranking_debug] sorted indices (top 5): {indices[:5].cpu().tolist()}")
            print(f"[ranking_debug] sorted scores (top 5): {doc_scores[indices[:5]]}")

        # Gold rank and score
        gold_rank = (indices == gold_tool_id).nonzero(as_tuple=True)[0].item() + 1  # 1‑based
        gold_score = doc_scores[gold_tool_id]
        
        if is_debug_sample:
            print(f"[ranking_debug] gold_tool_id: {gold_tool_id}")
            print(f"[ranking_debug] gold_score: {gold_score.item():.6f}")
            print(f"[ranking_debug] gold_rank: {gold_rank}")
            print(f"[ranking_debug] gold_tool_name: {gold_tool_name}")
            print(f"\n{'*'*60}\n")

        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # Update recall@1, recall@5
        if gold_rank <= 1:
            recall_at_1 = (recall_at_1 * total + 1) / (total + 1)
        else:
            recall_at_1 = (recall_at_1 * total) / (total + 1)

        if gold_rank <= 5:
            recall_at_5 = (recall_at_5 * total + 1) / (total + 1)
        else:
            recall_at_5 = (recall_at_5 * total) / (total + 1)

        total += 1

        if args.debug and qix % 100 == 0:
            print(f"After {qix + 1} samples: R@1 = {recall_at_1:.4f}, R@5 = {recall_at_5:.4f}")

    # Final metrics
    print(f"\nFinal metrics over {total} queries:")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")

    # Save and plot results
    analyze_gold_attention(results, save_path="plot2/gold_attention_plot.png")
    print("Plot saved to plot2/gold_attention_plot.png")