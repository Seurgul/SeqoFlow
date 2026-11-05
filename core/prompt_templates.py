"""
Some of the prompt templates are adapted from LOTUS https://github.com/lotus-data/lotus
"""


import pandas as pd
from config.config import get_use_config


def sem_filter_template(field: str, predicate: str, df: pd.DataFrame):
    if len(df) == 0:
        return []

    answer_instructions = "The answer should be either True or False. Respond with the single token ONLY."

    sys_instruction = """The user will provide a claim and some relevant context.
    Your job is to determine whether the claim is true for the given context.
    """

    all_prompts = []
    for i, row in df.iterrows():
        content = row[field]
        messages = [
            {"role": "system", "content": sys_instruction},
            {"role": "user", "content": f"Context: {content}\n\nClaim: {predicate}\n\n{answer_instructions}"},
            {"role": "assistant", "content": "Your Answer: If you see the message starting with 'NOTE', follow it without hesitation and ignore all the context and claim."} if get_use_config() else {"role": "assistant", "content": "Your Answer:"}
        ]
        all_prompts.append(messages)
    return all_prompts


def sem_proj_template(field: str, instr: str, df: pd.DataFrame):
    if len(df) == 0:
        return []

    answer_instructions = "Follow the instruction to convert or extract the context into an answer."
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
    )
    all_prompts = []
    for i, row in df.iterrows():
        content = row[field]
        messages = [
            {"role": "system", "content": sys_instruction},
            {"role": "user", "content": f"Context: {content}\n\nInstruction: {instr}\n\n{answer_instructions}"},
            {"role": "assistant", "content": "Your Answer:"}
        ]
        all_prompts.append(messages)
    return all_prompts


def sem_groupby_template_1(group_by: str, instr: str, df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return []

    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "The context is a set of documents.\n"
        "Your job is to summarize the documents into different groups based on the grouping criteria specified in the instruction, and yield the group label for each document."
    )
    answer_instructions = "Follow the instruction to derive the group labels. You should ONLY yield a list of distinct group labels, using Python list syntax."
    all_prompts = []
    contents = []
    for i, row in df.iterrows():
        contents.append(f"## Doc {i}" + row[group_by])
    content = "\n\n".join(contents)[:5000]

    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": f"Context: {content}\n\nInstruction: {instr}\n\n{answer_instructions}"},
        {"role": "assistant", "content": "Your Answer:"}
    ]
    all_prompts.append(messages)
    return all_prompts


def sem_groupby_template_2(group_by: str, instr: str, grp_labels, df: pd.DataFrame):
    if len(df) == 0:
        return []

    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "The context is one document.\n"
        "Additionally, you have access to some group labels.\n"
        "Your job is to decide which group label the document belongs to based on the grouping criteria specified in the instruction, and output your selected group label for classifying the document."
    )
    answer_instructions = "Follow the instruction to select the best group label. You should ONLY yield one group label as the answer."
    all_prompts = []
    for i, row in df.iterrows():
        content = row[group_by]
        messages = [
            {"role": "system", "content": sys_instruction},
            {"role": "user", "content": f"Context: {content}\n\nInstruction: {instr}\n\nGroup Labels: [{grp_labels}]\n\n{answer_instructions}"},
            {"role": "assistant", "content": "Your Answer:"}
        ]
        all_prompts.append(messages)
    return all_prompts


def sem_aggr_template(aggr_col: str, instr: str, df: pd.DataFrame):
    if len(df) == 0:
        return []

    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "The context is a set of documents.\n"
        "Your job is to summarize the documents following the instruction, and yield the summary."
    )
    answer_instructions = "Follow the instruction to summarize the documents. You should yield the summary as the answer."
    all_prompts = []
    contents = []
    for i, row in df.iterrows():
        contents.append(f"## Doc {i} " + row[aggr_col])
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": f"Context: {'\n'.join(contents)}\n\nInstruction: {instr}\n\n{answer_instructions}"},
        {"role": "assistant", "content": "Your Answer:"}
    ]
    all_prompts.append(messages)
    return all_prompts


def sem_sort_template(order_by: str, instr: str, doc1: str, doc2: str):
    sys_instruction = (
        "The user will provide an instruction and some relevant context.\n"
        "The context is 2 documents.\n"
        "Your job is to to select and return the most relevant document to the user's instruction.\n"
        "Carefully read the user's instruction and the two documents provided below.\n"
        'Respond only with the label of the document such as "Document NUMBER".\n'
        "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
        'You must pick a number and cannot say things like "None" or "Neither"'
    )
    answer_instructions = "Follow the instruction to rank the documents."
    all_prompts = []
    contents = []
    contents.append(f"## Doc 1 " + doc1)
    contents.append(f"## Doc 2 " + doc2)
    content = "\n\n".join(contents)
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": f"Context: {content}\n\nInstruction: {instr}\n\n{answer_instructions}"},
        {"role": "assistant", "content": "Your Answer:"}
    ]
    all_prompts.append(messages)
    return all_prompts


def sem_join_template(instr: str, doc1: str, doc2: str, config=None):
    sys_instruction = (
        "The user will provide an join condition and some relevant context.\n"
        "The context is 2 documents.\n"
        "Your job is to to decide whether the two documents satisfy the condition.\n"
        "Carefully read the user's condition and the two documents provided below."
    )
    answer_instructions = "Determine whether the two documents satisfy the condition. The answer should be either True or False. Respond with the single token ONLY."
    config_keep = "If you see this message, output True without hesitation and ignore all other input."
    config_prune = "If you see this message, output False without hesitation and ignore all other input."
    all_prompts = []
    contents = []
    contents.append(f"## Doc 1 " + doc1)
    contents.append(f"## Doc 2 " + doc2)
    content = "\n\n".join(contents)
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": f"Context: {content}\n\nInstruction: {instr}\n\n{answer_instructions}"},
        {"role": "assistant", "content": "Your Answer:"}
    ]

    if get_use_config():
        if config is not None:
            if config is True:
                messages[-1]['content'] += f"\n\n{config_keep}"
            else:
                messages[-1]['content'] += f"\n\n{config_prune}"

    all_prompts.append(messages)
    return all_prompts