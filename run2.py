'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
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

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)

    Strategy:
        Average attention across all heads within each layer,
        then sum query-to-doc attention, then average over layers,
        then normalise by document length.
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)

    # TODO 1: implement to get final query to doc attention stored in doc_scores
    q_start, q_end = query_span

    device = attentions[0].device

    doc_lengths = torch.tensor(
        [end - start for start, end in doc_spans], device=device
    ).clamp(min=1.0)

    num_layers = len(attentions)

    for layer_attention in attentions:
        # layer_attention: [1, heads, N, N]  -> average over heads -> [N, N]
        avg_attn   = layer_attention[0].mean(dim=0)
        query_attn = avg_attn[q_start:q_end, :]          # [q_len, N]

        for doc_idx, (d_start, d_end) in enumerate(doc_spans):
            score = query_attn[:, d_start:d_end].sum()
            doc_scores[doc_idx] += score

    doc_scores = doc_scores / num_layers   # average over layers
    doc_scores = doc_scores / doc_lengths  # normalise by doc length

    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    # TODO 2: visualize graph
    """
    input -> result: list of dicts with keys:
                        - gold_position
                        - gold_score
                        - gold_rank
    GOAL: Using the results data, generate a visualization that shows how attention to the gold tool varies with its position in the prompt.

    Requirements:
        - The plot should clearly illustrate whether position affects attention.
        - Save the plot as an image file under folder plot2.
        - You are free to choose how to aggregate and visualize the data.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    positions = np.array([r["gold_position"] for r in result])
    scores    = np.array([r["gold_score"]    for r in result])
    ranks     = np.array([r["gold_rank"]     for r in result])

    unique_positions    = np.unique(positions)
    mean_score_per_pos  = []
    std_score_per_pos   = []
    mean_rank_per_pos   = []

    for pos in unique_positions:
        mask = positions == pos
        mean_score_per_pos.append(scores[mask].mean())
        std_score_per_pos.append(scores[mask].std())
        mean_rank_per_pos.append(ranks[mask].mean())

    mean_score_per_pos = np.array(mean_score_per_pos)
    std_score_per_pos  = np.array(std_score_per_pos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Attention and Rank of Gold Tool vs Position in Prompt", fontsize=14)

    # plot 1: mean attention score vs position
    ax = axes[0]
    ax.plot(unique_positions, mean_score_per_pos, color="steelblue", lw=2)
    ax.fill_between(
        unique_positions,
        mean_score_per_pos - std_score_per_pos,
        mean_score_per_pos + std_score_per_pos,
        color="steelblue", alpha=0.3, label="Std Dev"
    )
    ax.set_xlabel("Position of Gold Tool in Prompt", fontsize=12)
    ax.set_ylabel("Mean Attention Score to Gold Tool", fontsize=12)
    ax.set_title("Attention Score vs Position", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # plot 2: mean rank vs position
    ax = axes[1]
    ax.plot(unique_positions, mean_rank_per_pos, color="coral", lw=2)
    ax.set_xlabel("Position of Gold Tool in Prompt", fontsize=12)
    ax.set_ylabel("Mean Rank of Gold Tool", fontsize=12)
    ax.set_title("Rank vs Position", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # plot 3: rank distribution
    ax = axes[2]
    ax.hist(ranks, bins=50, color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Gold Rank")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Gold Tool Rank")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Part 2] Plot saved -> {save_path}")


def get_query_span(input_ids, tokenizer, query_text):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Searches for the sub-sequence "Query: <query_text>" in the token stream
    and returns its (start, end) span (end is exclusive).
    """
    query_marker = f"text: {query_text}"
    marker_ids   = tokenizer(query_marker, add_special_tokens=False).input_ids
    marker_len   = len(marker_ids)
    ids_list     = input_ids.tolist()
    n            = len(ids_list)

    for i in range(n - marker_len + 1):
        if ids_list[i: i + marker_len] == marker_ids:
            return (i, i + marker_len)

    # Fallback: assume query is at the tail of the sequence
    return (n - marker_len, n)


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
    softmax_scaling = d ** -0.5
    train_queries, test_queries, tools = get_queries_and_items()

    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []

    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):
        sample   = test_queries[qix]
        qid      = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )
        item_spans     = putils.doc_spans
        doc_lengths    = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v: k for k, v in map_docname_id.items()}
        db_lengths_pt  = torch.tensor(doc_lengths, device=device)

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------" * 5)
            print(prompt)
            print("-------" * 5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------" * 5)

        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''

        query_span = get_query_span(
            input_ids=inputs.input_ids[0].cpu(),
            tokenizer=tokenizer,
            query_text=question,
        )

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # find gold_rank and gold_score
        ranked_doc_indices = torch.argsort(doc_scores, descending=True).cpu().tolist()
        gold_rank  = ranked_doc_indices.index(gold_tool_id)
        gold_score = doc_scores[gold_tool_id].item()

        results.append({
            "qid":           qid,
            "gold_position": gold_tool_id,
            "gold_score":    gold_score,
            "gold_rank":     gold_rank
        })

        # recall@1 and recall@5
        if gold_rank == 0:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        total += 1

        del attentions
        torch.cuda.empty_cache()

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    print(f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}")

    analyze_gold_attention(results)