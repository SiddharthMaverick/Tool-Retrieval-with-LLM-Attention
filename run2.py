'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
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

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    
    # TODO 1: implement to get final query to doc attention stored in doc_scores
    q_start, q_end = query_span
    num_layers = len(attentions)

    device = attentions[0].device
    doc_lengths = torch.tensor(
        [end - start for start, end in doc_spans], device=device
    ).clamp(min=1.0)
    
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
    import os
    os.makedirs("plot2", exist_ok=True)

    positions = [r["gold_position"] for r in result]
    scores    = [float(r["gold_score"]) for r in result]
    ranks     = [r["gold_rank"] for r in result]

    df = pd.DataFrame({"position": positions, "score": scores, "rank": ranks})

    # Bin positions (each position = one slot among ~100 tools)
    grouped = df.groupby("position").agg(
        mean_score=("score", "mean"),
        count=("score", "count")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean attention score vs gold tool position
    axes[0].scatter(grouped["position"], grouped["mean_score"],
                    alpha=0.7, s=grouped["count"] * 2, color="steelblue", edgecolors="k", linewidths=0.4)
    axes[0].set_xlabel("Position of Gold Tool in Prompt")
    axes[0].set_ylabel("Mean Attention Score")
    axes[0].set_title("Gold Tool Attention Score vs Position\n(bubble size ∝ query count)")

    axes[0].set_ylim(bottom=0, top=0.005)  # Ensure y-axis starts at 0
    # Add a smoothed trend line
    sorted_pos = grouped["position"].values
    sorted_scores = grouped["mean_score"].values
    axes[0].plot(sorted_pos, sorted_scores, color="salmon", linewidth=1.5, alpha=0.8)

    # Plot 2: Histogram of gold ranks to show retrieval quality
    axes[1].hist(ranks, bins=30, color="mediumseagreen", edgecolor="k", alpha=0.8)
    axes[1].set_xlabel("Rank of Gold Tool")
    axes[1].set_ylabel("Number of Queries")
    axes[1].set_title("Distribution of Gold Tool Ranks")
    axes[1].axvline(x=0.5, color="red", linestyle="--", label="Rank 1 boundary")
    axes[1].axvline(x=4.5, color="orange", linestyle="--", label="Rank 5 boundary")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {save_path}")

def get_query_span(input_ids, tokenizer, question):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    query_prompt = f"Query: {question}\nCorrect tool_id:"
    query_tokens = tokenizer(query_prompt, add_special_tokens=False).input_ids
    query_len = len(query_tokens)
    ids = input_ids.tolist()

    # Search from the end since query is near the end of the prompt
    for i in range(len(ids) - query_len, -1, -1):
        if ids[i : i + query_len] == query_tokens:
            return (i, i + query_len)

    # Fallback: just use the tail
    return (len(ids) - query_len, len(ids))

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
    num_key_value_groups = num_heads//model.config.num_key_value_heads
    softmax_scaling=d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()
    
    print("\n" + "="*70)
    print("[MODEL & DATA LOADED]")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Model Config: {num_layers} layers, {num_heads} heads, head_dim={d}")
    print(f"Attention Implementation: {model.config._attn_implementation}")
    print(f"Total test queries: {len(test_queries)}")
    print(f"Total tools available: {len(tools)}")
    print("="*70 + "\n")

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    
    print("[STARTING] Processing test queries...\n")
    
    for qix in tqdm(range(len(test_queries))):
        sample =  test_queries[qix]
        qid = sample["qid"]
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
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------"*5)
            print(prompt)
            print("-------"*5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------"*5)


        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        query_span = get_query_span(
            input_ids=inputs.input_ids[0].cpu(),
            tokenizer=tokenizer,
            question=question) 

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # Verify doc_scores are reasonable
        if args.debug and qix < 3:
            print(f"  [QIX {qix}] doc_scores range: [{doc_scores.min():.4f}, {doc_scores.max():.4f}]")
            print(f"  [QIX {qix}] gold_tool_id: {gold_tool_id}, total tools: {len(item_spans)}")

        # TODO: find gold_rank- rank of gold tool in doc_scores
        # TODO: find gold_score - score of gold tool
        # Rank all docs by score (descending)
        ranked_docs = torch.argsort(doc_scores, descending=True)

        # Rank of the gold tool (0-indexed)
        gold_rank  = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        gold_score = doc_scores[gold_tool_id].item()
        
        if args.debug and qix < 3:
            print(f"  [QIX {qix}] gold_rank: {gold_rank}, gold_score: {gold_score:.6f}")

        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
        if gold_rank == 0:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        total += 1
        
        # Print progress every 500 queries
        if (qix + 1) % 500 == 0:
            running_r1 = correct_at_1 / total
            running_r5 = correct_at_5 / total
            print(f"[Progress] Processed {qix + 1}/{len(test_queries)} - Running Recall@1: {running_r1:.4f}, Recall@5: {running_r5:.4f}")

        del attentions
        torch.cuda.empty_cache()

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    
    print("\n" + "="*70)
    print("[FINAL RESULTS]")
    print("="*70)
    print(f"Total Queries Processed: {total}")
    print(f"Recall@1: {recall_at_1:.4f} ({correct_at_1}/{total})")
    print(f"Recall@5: {recall_at_5:.4f} ({correct_at_5}/{total})")
    print("="*70)
    
    print("\n[GENERATING] Attention visualization plot...")
    analyze_gold_attention(results)
    
    print("\n" + "="*70)
    print("[COMPLETED] All processing done!")
    print("="*70)


    