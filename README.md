# Recsys Evaluation Of Dozen Datasets
**Authors:** Anonym

This repository implements the methodology presented in the article **Improving RecSys Evaluation: a Dozens-of-datasets Approach to
Reliable Benchmarking** (*In progress*).

Code allows you to run different algorithms with a preset of $29$ datasets from a variety of domains.

Repository supports a lot of different RecSys libraries with different models. A full list will be provided later, but we also support the implementation of the new models. Each model can be optimized via the Optuna framework.

To get the leaderboard, we implemented all the aggregation methods mentioned in the article.

Moreover, results and pre-trained models will be saved.

## Abstract

The field of Recommender Systems (RecSys) advances with the continuous introduction of new algorithms.
Meanwhile, in this domain, to prove the superiority of a new approach, researchers often constrain themselves to a few datasets and baselines, providing limited experimental evidence.
In machine learning, existing benchmarks enable a principled comparison of various methods across diverse scenarios, yet such benchmarks are mainly absent in the RecSys domain.
Our paper proposes a comprehensive benchmarking methodology for evaluating and ranking RecSys methods.
Specifically, we: (a) collect and prepare $27$ open datasets from various domains and introduce $2$ novel ones for benchmarking; (b) establish a pipeline to extract evaluation metrics for a set of ten recommendation algorithms with a straightforward introduction of new methods; and (c) design a principled aggregation methodology for the RecSys context, enabling an accurate ranking.
Through experimental analysis, we identify how evaluation procedures influence algorithm rankings, illustrate the relationship between algorithm performance and dataset characteristics, and identify the top-performing algorithms.

## Prerequirements

As some libraries have compatibility issues, we strongly recommend to use a docker image. You can follow next steps to ensure the correct install:

1. Modify file `docker_scripts/credentials` with your names and forwarded port.
2. Launch the `docker_scripts/build` script. Wait for the image complete building.
3. Launch the `docker_scripts/launch_container`.
4. Now you can simply attach to the running container and launch any experiment you want.

## Run the experiments

To set up the parameters of your experiment, one should modify the YAML configuration files located in the `config` folder. Each file corresponds to a single module, such as a dataset, library, or model. You can explore these files for more details. To run experiments, use the following command:

```
python main.py
```

Additionally, you can use [Hydra](https://hydra.cc/docs/intro/) notation to modify parameters in the console (for example, running multiple experiments).

## Supported models

| Library | Model | Documentation |
| ------- | ----- | ------ |
| *Our implementation* | MostPopular| |
| | Random | |
| **LightFM** | LightFM | https://making.lyst.com/lightfm/docs/home.html |
| **Implicit** | ALS | https://benfred.github.io/implicit/ |
| | BPR | |
| **RecBole**| EASE | https://www.recbole.io/|
| | ItemKNN | |
| | MultiVAE | |
| | SLIMElastic | |
| **MSRec** | SASRec | https://github.com/recommenders-team/recommenders |
