# This notebook contains the code to generate the submission for the "Pump it Up: Data Mining the Water Table" competition.
# It is essentially a copy of the xgboost notebook, but in script form. See that notebook for more details on the training.

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Add command line arguments
import argparse
parser = argparse.ArgumentParser(description='Run the preprocessing script.')
parser.add_argument('--input_dir', '-i', action='store',
                    help='The name of the directory where the preprocessed data are saved.')
parser.add_argument('--output_file', '-o', action='store',
                    help='The directory and filename of the submission.')
arguments = parser.parse_args()
input_dir = arguments.input_dir
output_file = arguments.output_file

X_train = pd.read_csv(f'{input_dir}/X_train.csv')
y_train = pd.read_csv(f'{input_dir}/y_train.csv')
X_val = pd.read_csv(f'{input_dir}/X_val.csv')
y_val = pd.read_csv(f'{input_dir}/y_val.csv')

# Best is then 1100, 0.05
model_fin = XGBClassifier(n_estimators=1100, learning_rate=0.05, n_jobs=-1, early_stopping_rounds=5, random_state=42)
model_fin.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

# Load test data
X_test = pd.read_csv(f'{input_dir}/X_test.csv')

# Prepare submission
output = pd.DataFrame(X_test["id"])
X_test.drop(columns=["id"], inplace=True)

y_test = model_fin.predict(X_test)
output["status_group"] = y_test
# Map to right strings again
output["status_group"] = output["status_group"].map({0: "non functional", 1: "functional needs repair", 2: "functional"})

# Save to csv
output.to_csv(output_file, index=False)