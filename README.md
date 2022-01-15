# LIFA: A Multi-source Transfer Learning Framework for Vietnamese Sentiment Classification
This is a Pytorch implementation for the paper "Learning for Almagamation: A Multi-source Transfer Learning Framework for Vietnamese Sentiment Classification", which is accepted to the journal Information Sciences. Please find the corresponding references: https://www.sciencedirect.com/science/article/pii/S0020025521012809?dgcid=author

## Requirement
* python                    3.7.3
* pytorch                   1.4.0
* pytorch-transformers      1.2.0
* tensorflow                2.0.0
* torchtext                 0.4.0
* torchvision               0.4.0
* scikit-image              0.15.0
* scikit-learn              0.20.3
* nltk                      3.4.5
* fairseq                   0.9.0
* vncorenlp                 1.0.3

## Data preparation
* In this work, we use two datasets:
  * AIVIVN: this is the publish dataset from AIVIVN 2019 Sentiment Challenge, including approximately 160K training reviews with the available labels and 11K testing reviews without the available labels. We manually did labelling for the testing dataset.
  * Our dataset: this is our new dataset which was crawled from the Vietnamese e-commerce websites, the reviews are started from Jan 2019 and includes all product categories. We trained all the methods with 10K, 15K, 20K training reviews respectively and tested on about 170K reviews.
  * The validation dataset is randomly selected from the training dataset, with 20%.
  * The two datasets are placed at the folders */dataset/aivivn/ and */dataset/tiki/.

## Pre-trained Models preparation
* BERT: which is the pre-trained BERT model with the version of bert-base-multilingual-uncased and automatically downloaded from Huggingface Transformers.
* PhoBERT: which is the state-of-the-art pre-trained BERT model for the Vietnamese language. To run with the pre-trained PhoBert models, we need to do:
  * Download https://public.vinai.io/PhoBERT_base_transformers.tar.gz, extract and place at */phobert
  * Download vncorenlp from https://github.com/VinAIResearch/PhoBERT#vncorenlp, extract and place at */vncorenlp

## Training
* For training and evaluating MoE methods, go to /moe folder and run the file run.sh
* For training and evaluating Transfer Learning methods (BERT and PhoBERT), go to /transfer_learning folder and run the file run.sh

## Copyright

For any requests to further use these codes and our proposed algorithms for the Vietnamese Sentiment Classification problem, please kindly contact authors to avoid any misused action or violation to the copyright of all authors and creators of this repository.

Contact Email: ngtbinh@hcmus.edu.vn (Dr. Binh Nguyen)

Please cite our journal paper when using our codes and datasets for further purposes.

```
@article{NGUYEN20221,
title = {Learning for amalgamation: A multi-source transfer learning framework for sentiment classification},
journal = {Information Sciences},
volume = {590},
pages = {1-14},
year = {2022},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.12.059},
url = {https://www.sciencedirect.com/science/article/pii/S0020025521012809},
author = {Cuong V. Nguyen and Khiem H. Le and Anh M. Tran and Quang H. Pham and Binh T. Nguyen},
keywords = {Sentiment classification, Transfer learning, LIFA, Mixture of experts, Low-resource NLP},
abstract = {Transfer learning plays an essential role in Deep Learning, which can remarkably improve the performance of the target domain, whose training data is not sufficient. Our work explores beyond the common practice of transfer learning with a single pre-trained model. We focus on the task of Vietnamese sentiment classification and propose LIFA, a framework to learn a unified embedding from several pre-trained models. We further propose two more LIFA variants that encourage the pre-trained models to either cooperate or compete with one another. Studying these variants sheds light on the success of LIFA by showing that sharing knowledge among the models is more beneficial for transfer learning. Moreover, we construct the AISIA-VN-Review-F dataset, the first large-scale Vietnamese sentiment classification database. We conduct extensive experiments on the AISIA-VN-Review-F and existing benchmarks to demonstrate the efficacy of LIFA compared to other techniques. To contribute to the Vietnamese NLP research, we publish our source code and datasets to the research community upon acceptance.}
}
```
