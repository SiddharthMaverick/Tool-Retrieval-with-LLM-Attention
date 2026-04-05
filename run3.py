'''
Part 3: Retrieval Heads

Goal:
    - Select a subset of attention heads using training data
    - Use ONLY those heads to rank tools at test time
'''

import json
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import time
import random
import numpy as np
import torch
from tqdm import tqdm

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

from code3 import select_retrieval_heads

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads):
    # TODO 2: Head-based scoring
    """
    Compute document scores using ONLY selected_heads.

    Inputs:
        attentions: tuple of (num_layers) each [1, heads, N, N]
        query_span: (start, end)
        doc_spans: list of (start, end)
        selected_heads: list of (layer, head)

    Output:
        doc_scores: tensor of shape [num_docs]
    """

    selected_heads = [(int(layer.replace("layer", "")), int(head.replace("head", ""))) for layer, head in selected_heads]

    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)

    for layer, head in selected_heads:
        layer_attn = attentions[layer][0]  # shape: [h, N, N]
        query_attn = layer_attn[head, query_span[0]:query_span[1], :]  # shape: [query_len, N]

        for doc_idx, (start, end) in enumerate(doc_spans):
            doc_attn = query_attn[:, start:end]  # shape: [query_len, doc_len]
            doc_score = doc_attn.mean().item()  # average over query and document tokens
            doc_scores[doc_idx] += doc_score

    doc_scores /= len(selected_heads)  # average over selected heads

    return doc_scores


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
parser.add_argument('--max_heads', type=int, default=20)
parser.add_argument('--train_samples', type=int, default=200)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':

    seed_all(args.seed)
    model_name = args.model
    device = "cuda:0"    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)

    train_queries, test_queries, tools = get_queries_and_items()
    print("\n[Phase 1] Selecting retrieval heads...")

    selected_heads = select_retrieval_heads(
        train_queries=train_queries[:args.train_samples],
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        device=device,
        max_heads=args.max_heads
    )

    print(f"Selected {len(selected_heads)} heads")

    os.makedirs("results/q3", exist_ok=True)
    with open("results/q3/selected_heads.json", "w") as f:
        json.dump(selected_heads, f)

    print(selected_heads)

    print("\n[Phase 2] Evaluating on test set...")
    recall_at_1 = 0
    correct_at_1 = 0
    recall_at_5 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):

        sample = test_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            # dataset="toole",
            # model_name=args.model,
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )

        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions

        modified_question = "Query: " + question + "\n"
        tokenized_query = tokenizer(modified_question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        query_span = get_query_span(tokenized_query=tokenized_query, tokenized_prompt=input_ids)

        doc_scores = query_to_docs_attention_heads(
            attentions,
            query_span,
            item_spans,
            selected_heads
        )


        # TODO: ranking the docs
        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()


        # TODO: measure the recall@1, recall@5
        correct_at_1 += (gold_rank == 0)
        correct_at_5 += (gold_rank < 5)
        total += 1

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total

    with open("results/q3/test_results.json", "w") as f:
        json.dump({
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5
        }, f)

    print(f"\nRecall@1 (selected heads): {recall_at_1:.4f}")
    print(f"\nRecall@5 (selected heads): {recall_at_5:.4f}")