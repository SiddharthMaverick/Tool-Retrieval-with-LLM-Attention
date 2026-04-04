import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
import numpy as np
import pandas as pd

import random

def load_model_tokenizer(model_name, device, dtype = torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                output_attentions = True,
                                                dtype=dtype,  
                                                local_files_only = True, # set True when the model is already downloaded
                                                )
    model.to(device)
    model.eval()
    return tokenizer, model

class PromptUtils:
    def __init__(self, tokenizer, doc_ids, dict_all_docs):
        self.dict_doc_name_id = {key:idx for idx, key in enumerate(doc_ids)}
        self.tokenizer = tokenizer
        self.prompt_seperator = " \n\n"
        user_header = '<|start_header_id|>user<|end_header_id|>'
        asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self.item_instruction = f" Here are all the available tools:"
        self.prompt_prefix = user_header + self.item_instruction
        self.prompt_suffix = asst_header
        self.prompt_prefix_length = len(tokenizer(self.prompt_prefix, add_special_tokens=False).input_ids)
        self.prompt_suffix_length = len(tokenizer(self.prompt_suffix, add_special_tokens=False).input_ids)
        
        self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
        self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."

        (
            self.all_docs_info_string, 
            self.doc_names_str, 
            self.doc_lengths,
            self.doc_spans
        ) = self.create_doc_pool_string(doc_ids, dict_all_docs)
        self.add_text1_length = len(tokenizer(self.add_text1, add_special_tokens=False).input_ids)

    
    def create_prompt(self, query):
        query_prompt = f"Query: {query}"+ "\nCorrect tool_id:"
        prompt = self.prompt_prefix + \
                self.all_docs_info_string + \
                self.prompt_seperator + \
                self.add_text1 + \
                self.prompt_seperator + \
                query_prompt + \
                self.prompt_suffix
        return prompt
        

    def create_doc_pool_string(self, shuffled_keys, all_docs):
        doc_lengths = []
        doc_list_str = []
        map_docname_id, map_id_docname = {}, {}
        all_schemas = ""
        doc_spans = []
        doc_st_index = self.prompt_prefix_length + 1 # inlcudes " \n\n"
        for ix, key in enumerate(shuffled_keys):
            value = all_docs[key]
            doc_list_str.append(key)
            text = self.prompt_seperator
            doc_text = self.doc_text(idx=self.dict_doc_name_id[key] + 1, doc_name=key, doc_info=value).strip()
            doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)
            text += doc_text
            doc_spans.append((doc_st_index, doc_st_index + doc_text_len))
            doc_st_index =  doc_st_index + 1 + doc_text_len
            doc_lengths.append(doc_text_len)
            all_schemas += text
            if ix == len(shuffled_keys)-1:
                end_of_docs_index = doc_st_index
        doc_list_str = ", ".join(doc_list_str)    
        return all_schemas, doc_list_str, doc_lengths, doc_spans
    
    

def get_queries_and_items_check():
    tool_path = "/scratch/deekshak/datasets/MetaTool/dataset/data/all_clean_data.csv"   
    tool_desc_path = "/scratch/deekshak/datasets/MetaTool/dataset/plugin_des.json"
    df =  pd.read_csv(tool_path)
    with open(tool_desc_path) as f:
        dbs = json.load(f)
    queries = []
    map_tool_count = {key: 0 for key in dbs}
    for idx in range(len(df)):
        row = df.iloc[idx]
        queries.append({
            "text": row["Query"],
            "gold_tool_name": row["Tool"],
            "qid": idx
            }
        )
        map_tool_count[row["Tool"]] += 1
    
    tools100 = sorted(map_tool_count.items(), key= lambda x: x[1], reverse=True)[:100]
    tools100 = [i[0] for i in tools100]
    queries_filtered = [i for i in queries if i["gold_tool_name"] in tools100]
    random.shuffle(queries_filtered)
    dbs_filtered = {i:dbs[i] for i in dbs if i in tools100}
    with open("data/test_queries.json", "w") as f: json.dump(queries_filtered[:5000], f)
    with open("data/train_queries.json", "w") as f: json.dump(queries_filtered[5000: 6500], f)
    with open("data/tools.json", "w") as f: json.dump(dbs_filtered, f)
    return queries_filtered, dbs_filtered


def get_queries_and_items():
    with open("data/test_queries.json", "r") as f: test_queries = json.load(f)
    with open("data/train_queries.json", "r") as f: train_queries  = json.load(f)
    with open("data/tools.json", "r") as f: tools = json.load(f)
    return train_queries, test_queries, tools