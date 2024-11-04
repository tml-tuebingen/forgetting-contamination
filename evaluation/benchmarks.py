from datasets import load_dataset, concatenate_datasets, Dataset

import tiktoken

import numpy as np
import os
import re

from datasets import disable_caching
disable_caching() # required for development


################################################################################
# Winogrande (2019)
#
# The task is to predict the right sentence out of two options.
################################################################################


def render_winogrande(example):
    sentence = example['sentence']
    assert ' _ ' in sentence
    s1 = sentence.replace(' _ ', ' ' + example['option1'] + ' ')
    s2 = sentence.replace(' _ ', ' ' + example['option2'] + ' ')
    return {"options": [s1, s2], "label": int(example['answer'])-1}


def load_winogrande():
    dataset = load_dataset("allenai/winogrande", 'winogrande_xl', trust_remote_code=True)['train']
    dataset = dataset.map(render_winogrande, remove_columns=dataset.column_names)
    return __gpt2__benchmark_safety__(dataset)

################################################################################
# Hellaswag
################################################################################


def render_hellaswag(example):
    def preprocess(text):
        """From https://github.com/allenai/OLMo/blob/main/olmo/eval/downstream.py"""
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text.strip()
    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]
    options = []
    for end in endings:
        if ctx[-1] != ' ' and end[0] != ' ': # insert space if needed
            ctx += ' '
        options.append(preprocess(ctx + end))
    return {"options": options, "label": label}


def load_hellaswag():
    print("Loading Hellaswag validation set...")
    dataset = load_dataset("Rowan/hellaswag", split='validation')
    dataset = dataset.map(render_hellaswag, remove_columns=dataset.column_names, load_from_cache_file=False)
    return __gpt2__benchmark_safety__(dataset)


################################################################################
# Social I-QA (https://arxiv.org/abs/1904.09728)
################################################################################


def render_social_i_qa(example):
    options = []
    ctx = example["context"] + "\nQuestion: " + example["question"] + "\nAnswer: "
    for letter in ["A", "B", "C"]:
        options.append(ctx + example["answer" + letter])
    return {"options": options, "label": int(example["label"])-1}


def load_social_i_qa():
    dataset = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
    dataset = dataset.map(render_social_i_qa, remove_columns=dataset.column_names)
    return __gpt2__benchmark_safety__(dataset)


################################################################################
# Piqa (https://arxiv.org/abs/1911.11641)
################################################################################


def render_piqa(example):
    ctx = f"Question: {example['goal']}\nSolution: "
    options = [ctx + example["sol1"], ctx + example["sol2"]]
    return {"options": options, "label": int(example["label"])}


def load_piqa():
    dataset = load_dataset("piqa", split="train", trust_remote_code=True)
    dataset = dataset.map(render_piqa, remove_columns=dataset.column_names)
    return __gpt2__benchmark_safety__(dataset)


################################################################################
# ARC-Easy (The AI2 Reasoning Challenge)
################################################################################


def render_arc(example):
    options = []
    ctx = f"Question: {example['question']}\nAnswer: "
    for opt in example["choices"]['text']:
        options.append(ctx + " " + opt)
    return {"options": options, "label": example["choices"]['label'].index(example["answerKey"])}


def load_arc_easy(split='all'):
    if split == 'all':
        train = load_arc_easy('train')
        dev = load_arc_easy('validation')
        test = load_arc_easy('test')
        return __gpt2__benchmark_safety__(concatenate_datasets([train, dev, test]))
    dataset = load_dataset("allenai/ai2_arc", 'ARC-Easy', trust_remote_code=True)[split]
    dataset = dataset.map(render_arc, remove_columns=dataset.column_names)
    return __gpt2__benchmark_safety__(dataset)


################################################################################
# BoolQ
################################################################################


def render_boolq(example):
    question =  f"{example['passage']}\nQuestion: {example['question']}?\nAnswer: "
    options = [question + "No", question + "Yes"]
    return {"options": options, "label": int(example['answer'])}


def load_boolq():
    dataset = load_dataset("google/boolq", split='validation', trust_remote_code=True)
    dataset = dataset.map(render_boolq, remove_columns=dataset.column_names)
    return __gpt2__benchmark_safety__(dataset)


################################################################################
# MMLU Source: https://github.com/hendrycks/test/blob/master/evaluate.py
################################################################################


mmlu_subcat = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

mmlu_cat = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

mmlu_stem = [k for k,v in mmlu_subcat.items() if v[0] in mmlu_cat["STEM"]]
mmlu_humanities = [k for k,v in mmlu_subcat.items() if v[0] in mmlu_cat["humanities"]]
mmlu_social_sciences = [k for k,v in mmlu_subcat.items() if v[0] in mmlu_cat["social sciences"]]
mmlu_other = [k for k,v in mmlu_subcat.items() if v[0] in mmlu_cat["other (business, health, misc.)"]]


def render_mml(example):
    options = []
    ctx = f"Question: {example['question']}\nAnswer:"
    for end in example["choices"]:
        options.append(ctx + " " + end)
    return {"options": options, "label": example["answer"]}


def load_mmlu(category=None):
    if category is None: # load full MMLU
        mmlu_stem_ds = None
        for subcat in mmlu_stem:
            if mmlu_stem_ds is None:
                mmlu_stem_ds = load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-stem"})
            else:
                mmlu_stem_ds = concatenate_datasets([mmlu_stem_ds, load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-stem"})])
        mmlu_social_ds = None
        for subcat in mmlu_social_sciences:
            if mmlu_social_ds is None:
                mmlu_social_ds = load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-social"})
            else:
                mmlu_social_ds = concatenate_datasets([mmlu_social_ds, load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-social"})])
        mmlu_humanities_ds = None
        for subcat in mmlu_humanities:
            if mmlu_humanities_ds is None:
                mmlu_humanities_ds = load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-humanities"})
            else:
                mmlu_humanities_ds = concatenate_datasets([mmlu_humanities_ds, load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-humanities"})])
        mmlu_other_ds = None
        for subcat in mmlu_other:
            if mmlu_other_ds is None:
                mmlu_other_ds = load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-other"})
            else:
                mmlu_other_ds = concatenate_datasets([mmlu_other_ds, load_mmlu(subcat).map(lambda x: {"benchmark": "mmlu-other"})])
        return concatenate_datasets([mmlu_stem_ds, mmlu_social_ds, mmlu_humanities_ds, mmlu_other_ds])
    # load a specific subcategory
    dataset = load_dataset("cais/mmlu", category, split='test', verification_mode='no_checks', keep_in_memory=True)
    return __gpt2__benchmark_safety__(dataset.map(render_mml, remove_columns=dataset.column_names))


################################################################################
# Load combinations of the different benchmarks
################################################################################


def load_contamination_split(split_id):
    """Load one of the 11 contamination-splits of the data contamination project by its id."""
    split_files = [
        '0-(0, 10000).parquet',
        '1-(4, 8000).parquet',
        '2-(12, 5000).parquet',
        '3-(36, 2000).parquet',
        '4-(144, 2000).parquet',
        '5-(4, 8000).parquet',
        '6-(12, 5000).parquet',
        '7-(36, 2000).parquet',
        '8-(144, 2000).parquet',
    ]
    fname = 'contamination_splits/' + split_files[split_id]
    dataset_path = __file__.replace("benchmarks.py", fname) # basepath is the same as this script
    return Dataset.from_parquet(dataset_path)


################################################################################
# Utilities
################################################################################


ALL_BENCHMARKS = {
    "winogrande": load_winogrande,
    "hellaswag": load_hellaswag,
    "social_i_qa": load_social_i_qa,
    "piqa": load_piqa,
    "arc_easy": load_arc_easy,
    "mmlu": load_mmlu,
    "boolq": load_boolq,

    # the split used for the data contamination project
    "contamination-split-0": lambda: load_contamination_split(0),
    "contamination-split-1": lambda: load_contamination_split(1),
    "contamination-split-2": lambda: load_contamination_split(2),
    "contamination-split-3": lambda: load_contamination_split(3),
    "contamination-split-4": lambda: load_contamination_split(4),
    "contamination-split-5": lambda: load_contamination_split(5),
    "contamination-split-6": lambda: load_contamination_split(6),
    "contamination-split-7": lambda: load_contamination_split(7),
    "contamination-split-8": lambda: load_contamination_split(8),
    "all-contamination-splits": lambda: concatenate_datasets([load_contamination_split(i) for i in range(9)]),
}


def load_benchmark(benchmark :str, **kwargs):
    if os.path.isfile(benchmark): # support loading from file
        ds = None
        if benchmark.endswith('.json'):
            ds = Dataset.from_json(benchmark)
        elif benchmark.endswith('.parquet'):
            ds = Dataset.from_parquet(benchmark)
        else:
            ds = Dataset.from_file(benchmark)
        return __gpt2__benchmark_safety__(ds)
    return ALL_BENCHMARKS[benchmark](**kwargs)


def random_baseline(dataset):
    acc = []
    for example in dataset:
        acc.append(1 / len(example['options']))
    return np.mean(acc)


def filter_max_tokens(dataset, num_tokens, encoding):
    """Assure that all options in the dataset fit within the context window of the model (1024 for gpt2)."""
    enc = tiktoken.get_encoding(encoding)
    return dataset.filter(lambda x: max([len(enc.encode(opt)) for opt in x["options"]]) <= num_tokens)


def sort_length(dataset, reverse=False):
    """Sort dataset questions from shortest to longest."""
    dataset = dataset.map(lambda x: {"max_option_length": max([len(option) for option in x["options"]])})
    dataset = dataset.sort("max_option_length", reverse=reverse)
    dataset = dataset.remove_columns("max_option_length")
    return dataset


def __gpt2__benchmark_safety__(dataset):
    """Assure all options fit within the 1024 token limit. Also sort questions by the length of the longest option."""
    dataset = filter_max_tokens(dataset, num_tokens=1024, encoding="gpt2")
    return sort_length(dataset)