# DrivenData-Pump-it-Up

This repository contains the code to obtain my results for the Pump it Up competition on DrivenData.
A snakemake pipeline is also added through the Snakefile, allowing one to generate results easily.
The best performing model is RandomForestClassifier, which got a submission score of 0.8048 for its classification rate.

Applied methods in this script are: machine learning using xgboost, tensorflow and pytorch, preprocessing (imputation, one-hot encoding, ordinal encoding, feature engineering), data exploration, hyperparameter optimisation, and building a snakemake pipeline.

DrivenData competition: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/

Data at: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/
