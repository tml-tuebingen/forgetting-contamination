# How much can we forget about Data Contamination? 

<p align="center">
  <img src="images/landing.png" width="800" alt=""/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

This is the code repository for the ICML'25 paper ["How much can we forget about Data Contamination?"](https://arxiv.org/abs/2410.03249).     

# Overview

TODO

## Using the de-duplicated benchmark questions

Many popular LLM benchmarks contain duplicate questions. For this project, we created a universe of 44000 benchmark questions whose ground-truth answers are sufficiently unique (for details, see the Section "Filtering Near-Duplicate Benchmark Questions" in our paper). We provide these benchmark questions [here](https://drive.google.com/drive/folders/1HN1G4ymhzkJw5VTra5wyYv_USYOaeEms?usp=drive_link). The format of the benchmark questions matches their respective format on Huggingface. 

The benchmark questions contain an additional column, "split-id," that can be used to partition the benchmark questions into different subsets. For example, we used the questions with split-id=0 as holdout. 

## Reproducing the results in our paper

This repository contains the code that can be used to reproduce the results in our paper.

- ```llm.c/```: The code to contaminate, train, and evaluate small models.
- ```evaluation/```: The code to format benchmark questions, de-duplicate them, and generate the differnt splits that we use for contamination.
- ```olmo/```: The code to download specific training batches, contaminate them, and insert them back into the olmo pre-training data.
- ```compute_results.ipynb```: Compute accuracies, confidence intervals.
- ```figures.ipynb```: Generate the figures in the paper.
- ```forgetting_curves.ipynb```: Generate the forgetting curves.

The OLMo experiments depend on https://github.com/allenai/OLMo. We worked with the repository version with the commit hash ```ca81901eca2faa1947ced49ce5c5cef729203db1```.

## Checkpoints

Model checkpoints are available [here](https://drive.google.com/drive/folders/19fERdR4bmfDmNqkdYass21Jd7ccf9XN3?usp=sharing).

## Citing our work

If you use this software in your research, we encourage you to cite our paper.

```bib
@inproceedings{bordt2025forgetting,
  author    = {Sebastian Bordt, Suraj Srinivas, Valentyn Boreiko, and Ulrike von Luxburg},
  title     = {How much can we forget about Data Contamination?},
  booktitle = {ICML},
  year      = {2025}
 }
```
