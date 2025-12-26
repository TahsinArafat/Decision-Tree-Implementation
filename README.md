# Decision Tree Implementation (CART vs ID3)

This repository demonstrates decision-tree classification on a tabular dataset, comparing two common split criteria:

- **CART** (Gini impurity)
- **ID3-style** decision tree (Information Gain / Entropy)

The work is implemented in a single Jupyter notebook and uses scikit-learn for model training, hyperparameter tuning, and evaluation.

## What’s Included

- [210116_DT.ipynb](210116_DT.ipynb): end-to-end workflow (data loading → preprocessing → training → evaluation → visualization)
- [data.csv](data.csv): dataset used by the notebook

## Dataset

The notebook uses a breast-cancer diagnostic dataset format with:

- a target column: `diagnosis` (Benign/Malignant)
- an `id` column (dropped before training)
- numeric feature columns (30 features)
- an extra empty column in some exports (the notebook drops `Unnamed: 32` if present)

## Method

1. Load the CSV into a pandas DataFrame.
2. Drop non-feature columns (`id`, and `Unnamed: 32` if present).
3. Label-encode the target label (`diagnosis`).
4. Split the data into train/validation/test.
5. Train two decision trees with `GridSearchCV`:
	- `criterion='gini'` (CART)
	- `criterion='entropy'` (ID3-style)
6. Evaluate on the test set using:
	- confusion matrix
	- Accuracy, Precision, Recall, F1
	- ROC curve and AUC
7. Visualize the learned tree structure.

## How to Run

### Option A: VS Code (recommended)

1. Open [210116_DT.ipynb](210116_DT.ipynb) in VS Code.
2. Select a Python kernel.
3. Run cells from top to bottom.

### Option B: Jupyter Notebook

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

jupyter notebook
```

Then open [210116_DT.ipynb](210116_DT.ipynb) and run all cells.

## Notes

- The notebook currently loads the dataset from GitHub raw URL, but the repository also includes a local copy at [data.csv](data.csv). If you want to run fully offline, replace the URL read with `pd.read_csv('data.csv')`.
- `random_state=42` is used for reproducible splits and model training.

## Expected Outputs

Running the notebook produces:

- confusion matrices for CART and ID3-style trees
- ROC curves with AUC for both models
- a bar chart comparing metrics (Acc/Prec/Rec/F1/AUC)
- a plotted decision tree visualization