# irrigation-pred-kaggle-comp documentation!

## Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Irrigation Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://test-pred-irr/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://test-pred-irr/data/` to `data/`.


