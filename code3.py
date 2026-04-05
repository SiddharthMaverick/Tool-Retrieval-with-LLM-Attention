import torch
from tqdm import tqdm
from utils import PromptUtils
import random

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

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    query_gold_tool_ranks = torch.zeros(num_layers, num_heads, len(train_queries), device=device)

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=tool_ids, 
            dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id # tool name to index mapping
        map_id_docname = {v:k for k, v in map_docname_id.items()} # index to tool name mapping
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line
        modified_question = "Query: " + question + "\n"
        tokenized_query = tokenizer(modified_question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        query_span = get_query_span(tokenized_query=tokenized_query, tokenized_prompt=input_ids)

        doc_scores = torch.zeros((num_layers, num_heads, len(item_spans)), device=device)

        for layer in range(num_layers):
            layer_attn = attentions[layer][0]  # shape: [h, N, N]
            query_attn = layer_attn[:, query_span[0]:query_span[1], :]  # shape: [h, query_len, N]
            for doc_idx, (start, end) in enumerate(item_spans):
                doc_attn = query_attn[:, :, start:end].mean(dim=2)  # shape: [h, query_len]
                doc_scores[layer, :, doc_idx] = doc_attn.mean(dim=1)  # shape: [h]

        # Finding ranks of the gold tool for each head
        gold_tool_id = map_docname_id[gold_tool_name]
        for layer in range(num_layers):
            for head in range(num_heads):
                _, sorted_indices = torch.sort(doc_scores[layer, head], descending=True)
                gold_rank = (sorted_indices == gold_tool_id).nonzero(as_tuple=True)[0].item() + 1 # 1 indexed rank
                query_gold_tool_ranks[layer, head, qix] = 1 / gold_rank # store reciprocal rank for calculating mean reciprocal rank later

    # Calculate mean reciprocal rank for each head across all queries
    head_scores = query_gold_tool_ranks.mean(dim=2)

    # TODO: select top heads
    top_heads = torch.topk(head_scores.view(-1), k=max_heads)
    selected_heads = [(f"layer{(idx // num_heads).item()}", f"head{(idx % num_heads).item()}") for idx in top_heads.indices]

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads