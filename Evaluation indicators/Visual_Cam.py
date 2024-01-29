from transformers import BertTokenizer, BertModel
from bertviz import head_view
import torch

bert_model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)
text = " " #Your input text is here.

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state

head_view(
    model=model,
    tokenizer=tokenizer,
    layers=[11],
    heads=[5],
    data={"inputs": inputs, "attention_mask": inputs["attention_mask"], "token_type_ids": inputs.get("token_type_ids")},
)
