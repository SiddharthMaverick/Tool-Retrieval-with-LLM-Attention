'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
#os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
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
    """
    device = attentions[0].device
    doc_scores = torch.zeros(len(doc_spans), device=device)
    
    # TODO 1: implement to get final query to doc attention stored in doc_scores
    query_start, query_end = query_span
    num_layers = len(attentions)

    doc_lengths = torch.tensor(
        [max(end - start, 1) for start, end in doc_spans],
        dtype=torch.float32,
        device=device,
    )

    for layer_attn in attentions:
        # layer_attn: [1, heads, N, N]
        avg_over_heads = layer_attn[0].mean(dim=0)              # [N, N]
        query_attn     = avg_over_heads[query_start:query_end]  # [q_len, N]

        for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
            doc_scores[doc_idx] += query_attn[:, doc_start:doc_end].sum()

    doc_scores /= num_layers
    doc_scores /= doc_lengths
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
    df = pd.DataFrame(result)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df["gold_position"] = df["gold_position"].astype(int)
    df["gold_score"]    = df["gold_score"].astype(float)

    grouped = (
        df.groupby("gold_position")["gold_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.4, zorder=0)

    jitter = np.random.normal(0, 0.1, size=len(df))
    plt.scatter(df["gold_position"] + jitter, df["gold_score"],
                alpha=0.15, s=15, color="steelblue", label="Per-query gold score", zorder=1)

    lower_bound = np.clip(grouped["mean"] - grouped["std"], a_min=0, a_max=None)
    upper_bound = grouped["mean"] + grouped["std"]
    plt.fill_between(grouped["gold_position"], lower_bound, upper_bound,
                     color="crimson", alpha=0.2, label="+/-1 Std Dev", zorder=2)
    plt.plot(grouped["gold_position"], grouped["mean"],
             color="crimson", linewidth=2.5, marker="o", label="Mean gold score", zorder=3)

    plt.xlabel("Gold Tool Position in Prompt", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Attention Score",               fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Gold Tool Attention vs. Prompt Position\n(Lost-in-the-Middle Analysis)",
              fontsize=14, pad=15)

    max_score = df["gold_score"].max()
    min_score = df["gold_score"].min()
    y_pad = (max_score - min_score) * 0.1 if max_score > min_score else max_score * 0.1
    plt.ylim(bottom=max(0, min_score - y_pad), top=max_score + y_pad)
    if max_score < 0.01:
        plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="lightgray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")

def get_query_span(input_ids, putils):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    total_len     = len(input_ids)
    last_doc_end  = putils.doc_spans[-1][1]

    query_start = last_doc_end + 1 + putils.add_text1_length + 1
    query_end   = total_len - putils.prompt_suffix_length

    query_start = max(0, min(query_start, total_len - 1))
    query_end   = max(query_start + 1, min(query_end, total_len))

    return (query_start, query_end)

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
    total        = 0
    
    
    
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
            outputs    = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        query_span = get_query_span(input_ids=inputs.input_ids[0].cpu(),putils=putils)
        
        if query_span[0] >= query_span[1]:
            del inputs
            continue
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
            # --- CRITICAL MEMORY FIX ---
            # Move the massive 9GB attention maps to CPU instantly.
            attentions = tuple(layer_attn.cpu() for layer_attn in outputs.attentions)
            
            # Aggressively delete the outputs (which holds the ~1GB logits) and inputs
            del outputs
            del inputs
            
            # Force PyTorch to release the VRAM back to the GPU before the next calculation
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # ---------------------------

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank- rank of gold tool in doc_scores
        # TODO: find gold_score - score of gold tool
        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        gold_score = doc_scores[gold_tool_id].item()
        
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

        
        
        
    if total > 0:
        recall_at_1 = correct_at_1 / total
        recall_at_5 = correct_at_5 / total
        print(f"Recall@1: {recall_at_1:.4f}")
        print(f"Recall@5: {recall_at_5:.4f}")
        analyze_gold_attention(results)
    else:
        print("Total valid queries was 0. Check your tokenizer/prompt logic.")        

    analyze_gold_attention(results)

    