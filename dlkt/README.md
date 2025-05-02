# Overview
This document outlines the steps to reproduce the DLKT-based preference tracing model results (specifically DKT, DKVMN, SAKT) reported in our paper. Our experiments utilize a modified version of the `pykt-toolkit` library. We forked the original repository (https://github.com/pykt-team/pykt-toolkit) and adapted it for our specific dataset(`ml-1m`) and experimental setup.  

The following instructions detail the necessary modifications and procedures to replicate our findings. You can also refer to the official documentation provided by pykt-team for more general information about the library: https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html#installation


## 0. Prerequisites
1. Clone the `pykt-toolkit` repository:
```bash
git clone https://github.com/pykt-team/pykt-toolkit.git
```

2. Install required dependencies:
- Ensure that you are in the directory containing `pyproject.toml` file.
- Run `uv sync` to install dependencies based on the `pyproject.toml` file.
```bash
uv sync
cd pykt-toolkit/examples
```

3. Prepare the dataset:
- Download the MovieLens 1M data (`ml-1m`). You can usually find this dataset publicly available (e.g., from the GroupLens website: https://grouplens.org/datasets/movielens/1m/)
- Place the `ratings.csv`, which is preprocessed by `preprocessing.ipynb`, into the `pykt-toolkit/data/ml-1m`.
- Create the necessary directories if they do not exist.


## 1. Data Preparation
1. Modify `data_preprocess.py`:
Since the original library only can handle the benchmark datasets in knowledge tracing domain, you need to modify the `data_preprocess.py` to include our customized dataset.
- Open the file: `pykt-toolkit/examples/data_preprocess.py`
- Add an entry for the `ml-1m` dataset in the dictionary named `dname2paths`, pointing the location of your `ratings.csv`
```python
dname2paths = {
    # ... other benchmark datasets ...
    'ml-1m': '../data/ml-1m/ratings.csv' # Add this line 
}
```  

2. Run the modified `data_preprocess.py`:
- Execute the script, specifying `ml-1m` as the dataset name.
```bash
python data_preprocess.py --dataset-name=ml-1m
```
This will process the `ratings.csv` file and generate the necessary data files, suited for the required format, for the model training.


## 2. Model Training
Now you can train the specific DLKT-based preference tracing models used in our paper.
1. Execute the model training script:
- Use the appropriate `wandb_{model_name}_train.py` script.
- Replace the `{model_name}` with the actual model you want to train (e.g., `dkt`, `dkvmn`, `sakt`).
- Specify the dataset using `--dataset-name=ml-1m`.
- [Optional] You can specify other options; please refer to the official documents for the available options.
```bash
# Example command line for training DKT model
python wandb_dkt_train.py --dataset-name=ml-1m

# Example command line for training DKVMN model
python wandb_dkvmn_train.py --dataset-name=ml-1m

# Example command line for training SAKT model
python wandb_sakt_train.py --dataset-name=ml-1m
```


## 3. Model Evaluation
After the model training, we can evaluate the performance of the saved models.
1. Run the prediction/evaluation script:
- Use the `wandb_predict.py` script
- Use the `--save_dir` argument to specify the directory where the trained model checkpoints were saved during the training phase. You'll need to identify this directory based on the output from the training step.
```bash
python wandb_predict.py --save_dir='saved_model/...'
```

2. Check the results:
- The evaluation results will be printed to the console.
- Look for the dictionary output containing the `window_testauc` key.
- [Optional] If `wandb` integration is active, results are also logged to the `wandb_predict` project on the WandB website. We uploaded the log files from our experiment as evidence. The reported performance of our proposed models can be found in each log file's `window_testauc`.