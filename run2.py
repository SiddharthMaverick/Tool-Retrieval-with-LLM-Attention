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
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    for attn_layer in attentions:
        attn_layer = attn_layer.squeeze(0)  # shape: [heads, N, N]
        query_attn = attn_layer[:, query_span[0]:query_span[1], :]  # shape: [heads, query_len, N]
        doc_attn_scores = []

        for doc_span in doc_spans:
            doc_attn = query_attn[:, :, doc_span[0]:doc_span[1]]  # shape: [heads, query_len, doc_len]
            doc_attn_score = doc_attn.mean().item()  # average over heads, query and document tokens
            doc_attn_scores.append(doc_attn_score)

        doc_scores += torch.tensor(doc_attn_scores, device=attentions[0].device)

    doc_scores /= len(attentions)  # average over layers

    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    """
    Generates a clean, readable visualization of the 'Lost in the Middle' effect.
    """
    position_scores = {}
    for res in result:
        pos = res["gold_position"]
        score = res["gold_score"]
        position_scores.setdefault(pos, []).append(score)

    positions = sorted(position_scores.keys())
    avg_scores = [np.mean(position_scores[p]) for p in positions]
    counts = [len(position_scores[p]) for p in positions]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Add horizontal gridlines for readability
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # 2. Plot the bars (softer color, no borders to reduce visual noise)
    bars = ax.bar(positions, avg_scores, color="#4C72B0", alpha=0.6, width=0.85, 
                  zorder=2, label="Avg Score per Position")

    # 3. Add a Polynomial Trendline (Degree 3 fits the 'U-Shape' perfectly)
    # This clearly illustrates the "Lost in the Middle" effect to the viewer
    z = np.polyfit(positions, avg_scores, 3)
    p = np.poly1d(z)
    ax.plot(positions, p(positions), color="#C44E52", linewidth=3, linestyle="-", 
            zorder=3, label="Trendline (Poly Fit)")

    # 4. Fix X-Axis Clutter (Show ticks every 5 positions instead of every single one)
    ax.set_xticks(np.arange(min(positions), max(positions) + 1, 5))
    
    # 5. Clean up Labels, Titles, and Fonts
    ax.set_xlabel("Gold Tool Position (index in shuffled list)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel("Average Attention Score", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("Average Query → Gold-Tool Attention by Gold Tool Position\n(Lost in the Middle Effect)", 
                 fontsize=15, fontweight='bold', pad=15)

    # 6. Remove the cluttered 'n=' text above every bar. 
    # Instead, summarize the sample size in a clean text box.
    avg_n = int(np.mean(counts))
    ax.text(0.02, 0.95, f"Avg samples per position ≈ {avg_n}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
            zorder=4)

    # 7. Clean up the borders (remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # Increased DPI for a crisper image
    plt.close()
    
    print(f"Plot saved to {save_path}")

def get_query_span(tokenized_query, tokenized_prompt):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    start = len(tokenized_prompt) - 1
    end = len(tokenized_prompt) - 1

    i = len(tokenized_prompt) - 1
    j = len(tokenized_query) - 1

    while(i >= 0):
        if tokenized_prompt[i] == tokenized_query[j]:
            if j == len(tokenized_query) - 1:
                end = i + 1
            j -= 1
            if j < 0:
                start = i
                break
        i -= 1  

    return (start, end)

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
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    i=0
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
        modified_question = "Query: " + question + "\n"

        tokenized_query = tokenizer(modified_question, add_special_tokens=False).input_ids

        query_span = get_query_span(tokenized_query=tokenized_query, tokenized_prompt=inputs.input_ids.squeeze(0).tolist())
        
        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank- rank of gold tool in doc_scores
        # TODO: find gold_score - score of gold tool
        gold_score = doc_scores[gold_tool_id].item()
        ranked_indices = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_indices == gold_tool_id).nonzero(as_tuple=True)[0].item() + 1
        
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
        recall_at_1 += 1.0 if gold_rank == 1 else 0.0
        recall_at_5 += 1.0 if gold_rank <= 5 else 0.0

        del attentions, doc_scores, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    num_queries = len(test_queries)
    recall_at_1 /= num_queries
    recall_at_5 /= num_queries

    

    print(f"Recall@1: {recall_at_1:.4f}, Recall@5: {recall_at_5:.4f}")

    analyze_gold_attention(results)

    