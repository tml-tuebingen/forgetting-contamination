# How much can we forget about Data Contamination? 

<p align="center">
  <img src="images/landing.png" width="800" alt=""/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)

This is the code repository for the paper ["How much can we forget about Data Contamination?"](https://arxiv.org/abs/2410.03249) by Sebastian Bordt, Suraj Srinivas, Valentyn Boreiko, and Ulrike von Luxburg.     

# Overview

TODO

# Using the de-duplicated benchmark questions for your own project

TODO

# Reproducing the results in our paper

details ...

Here we give a brief overview of the code:

- ```evaluation/```: Contains the code to format benchmark questions, de-duplicate them, and generate the differnt splits that we use for contamination.
- ```llm.c/```: Contains the code to contaminate, train, and evaluate small GPT-3 models.
- ```olmo/```: Contains code to download specific training batches, contaminate them, and insert them back into the the olmo pre-training run. This code depends on https://github.com/allenai/OLMo.
- ```compute_results.ipynb```: Compute accuracies, confidence intervals.
- ```figures.ipynb```: Generate the figures in the paper.
- ```forgetting_curves.ipynb```: Generate the forgetting curves.

# Checkpoints

Model checkpoints are available [here](https://drive.google.com/drive/folders/19fERdR4bmfDmNqkdYass21Jd7ccf9XN3?usp=sharing).

## Citing our work

If you use this software in your research, we encourage you to cite our paper.

```bib
@inproceedings{bordt2025forgetting,
  author    = {Sebastian Bordt, Suraj Srinivas, Valentyn Boreiko, and Ulrike von Luxburg},
  title     = {How much can we forget about Data Contamination? },
  booktitle = {ICML},
  year      = {2025}
 }
```
