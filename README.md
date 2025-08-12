# How much can we forget about Data Contamination? 

<p align="center">
  <img src="images/landing.png" width="800" alt=""/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

This is the code repository for the ICML'25 paper ["How much can we forget about Data Contamination?"](https://arxiv.org/abs/2410.03249).     

# Overview

This repository contains the code to reproduce the results in our paper. We also provide the deduplicated benchmark questions, as well as model checkpoints and our training logs for OLMo.

## Using the de-duplicated benchmark questions

Many popular LLM benchmarks contain duplicate questions. For this project, we created a universe of 44000 benchmark questions whose ground-truth answers were deduplicated using fuzzy string matching (for details, see the Section "Filtering Near-Duplicate Benchmark Questions" in our paper). We provide the deduplicated benchmark questions on Huggingface

- https://huggingface.co/datasets/sbordt/forgetting-contamination-hellaswag
- https://huggingface.co/datasets/sbordt/forgetting-contamination-winogrande
- https://huggingface.co/datasets/sbordt/forgetting-contamination-piqa
- https://huggingface.co/datasets/sbordt/forgetting-contamination-boolq
- https://huggingface.co/datasets/sbordt/forgetting-contamination-arc-easy
- https://huggingface.co/datasets/sbordt/forgetting-contamination-mmlu
- https://huggingface.co/datasets/sbordt/forgetting-contamination-social_i_qa

The format of the benchmark questions is the same as in the respective original repositories, with an additional column, "split-id," that can be used to partition the benchmark questions into different subsets. We used the questions with split-id=0 as holdout, and the other splits were contaminated as specified [here](https://github.com/tml-tuebingen/forgetting-contamination/blob/main/llm.c/create_contaminated_dataset.py). 

You can use the provided repositories in the same way that you would use the original repositories (because the format of the questions remains the same).

## Reproducing the results in our paper

Here is a brief overview of the code:

- ```llm.c/```: The code to contaminate, train, and evaluate small models.
- ```evaluation/```: The code to format benchmark questions, de-duplicate them, and generate the differnt splits that we use for contamination.
- ```olmo/```: The code to download specific training batches, contaminate them, and insert them back into the olmo pre-training data.
- ```compute_results.ipynb```: Compute accuracies, confidence intervals.
- ```figures.ipynb```: Generate the figures in the paper.
- ```forgetting_curves.ipynb```: Generate the forgetting curves.

The OLMo experiments depend on https://github.com/allenai/OLMo. We worked with the repository version with the commit hash ```ca81901eca2faa1947ced49ce5c5cef729203db1```.

## Checkpoints

Model checkpoints are available [here](https://drive.google.com/drive/folders/19fERdR4bmfDmNqkdYass21Jd7ccf9XN3?usp=sharing).

## Weights & Biases Logs

For reproducibility, we additionally share the following Weights & Biases Logs:

OLMo-1B: [Link](https://api.wandb.ai/links/train-on-test/1r02w9og)

OLMO-7B: [Link](https://api.wandb.ai/links/train-on-test/60384zi6)

## Absolute Accuracies in Figure 1 of the paper

In Figure 1 in the paper, we report the accuracy differences between the different splits of benchmark questions. Here, we additionally provide the absolute accuracies of the respective splits. 

#### Figure 1(a)

| Model Size | Holdout | 4x    | 12x   | 32x   | 144x  |
|------------|---------|-------|-------|-------|-------|
| 124M       | 44.16   | 49.54 | 54.98 | 73.05 | 93.20 |
| 350M       | 44.72   | 55.69 | 69.90 | 89.20 | 95.50 |
| 774M       | 45.78   | 67.30 | 85.16 | 94.65 | 97.25 |
| 1558M      | 46.90   | 75.48 | 91.04 | 95.70 | 97.55 |

#### Fig 1(b)

| Model Size | Holdout | 4x | 12x | 32x | 144x |
|------------|---------|----|----|-----|------|
| 2x Chinchilla | 43.31 | 50.40 | 59.84 | 80.85 | 94.85 |
| 4x Chinchilla | 44.52 | 50.75 | 58.10 | 78.35 | 93.65 |
| 8x Chinchilla | 45.14 | 49.16 | 51.84 | 64.15 | 85.15 |
| 15x Chinchilla | 46.45 | 48.51 | 47.88 | 51.20 | 67.10 |

#### Fig 1(c) 

| Model Size | Holdout | 4x | 12x | 32x | 144x |
|------------|---------|----|----|-----|------|
| 124M | 42.22 | 48.14 | 56.92 | 80.70 | 96.45 |
| 350M | 44.72 | 55.69 | 69.90 | 89.20 | 95.50 |
| 774M | 49.16 | 64.76 | 81.30 | 92.95 | 96.05 |
| 1.6B | 52.06 | 67.61 | 82.32 | 91.85 | 95.40 |

The result for Figure 1 (c) is Table 1 in the paper.

## Citing our work

If you use the code or the deduplicated benchmark questions in your research, we encourage you to cite our paper. 

```bib
@inproceedings{bordt2025forgetting,
  author    = {Sebastian Bordt and Suraj Srinivas and Valentyn Boreiko and Ulrike von Luxburg},
  title     = {How much can we forget about Data Contamination?},
  booktitle = {ICML},
  year      = {2025}
 }
```
