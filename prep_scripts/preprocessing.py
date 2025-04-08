# This notebook contains the code to generate the preprocessed data for the "Pump it Up: Data Mining the Water Table" competition.
# It is essentially a copy of the preprocessing notebook, but in script form. See that notebook for more details on 
# the data and the preprocessing steps.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Add command line arguments
import argparse
parser = argparse.ArgumentParser(description='Run the preprocessing script.')
parser.add_argument('--input_dir', '-i', action='store',
                    help='The name of the directory where the raw data are saved.')
parser.add_argument('--output_dir', '-o', action='store',
                    help='The name of the directory where the preprocessed data will be saved.')
arguments = parser.parse_args()
input_dir = arguments.input_dir
output_dir = arguments.output_dir

X = pd.read_csv(f'{input_dir}/training_values.csv')
X_test = pd.read_csv(f'{input_dir}/test_values.csv')
y = pd.read_csv(f'{input_dir}/training_labels.csv')

# Drop useless columns
X.drop(columns=['id', 'wpt_name', 'num_private', 'recorded_by', 'payment_type', 'quantity_group'], inplace=True)
X_test.drop(columns=['wpt_name', 'num_private', 'recorded_by', 'payment_type', 'quantity_group'], inplace=True)

# Drop duplicate columns
X.drop(columns=['subvillage', 'region', 'lga', 'ward'], inplace=True)
X_test.drop(columns=['subvillage', 'region', 'lga', 'ward'], inplace=True)

# Change data_recorded into just the year, then combine that with construction year to get age of the waterpoint
X['date_recorded'] = pd.to_datetime(X['date_recorded'])
X_test['date_recorded'] = pd.to_datetime(X_test['date_recorded'])
X['construction_year'] = X['construction_year'].replace(0, np.nan)  # data contains 0, which must mean unknown
X_test['construction_year'] = X_test['construction_year'].replace(0, np.nan)
X['age'] = X['date_recorded'].dt.year - X['construction_year']
X_test['age'] = X_test['date_recorded'].dt.year - X_test['construction_year']
X.drop(columns=['date_recorded', 'construction_year'], inplace=True)
X_test.drop(columns=['date_recorded', 'construction_year'], inplace=True)

# For y we also drop the id column
y.drop(columns='id', inplace=True)
# Then the status is an ordinal variable, with non functional being the worst, needing repair being better, and functional being best,
# so we can just encode that as 0, 1, 2
y['status_group'] = y['status_group'].map({'non functional': 0, 'functional needs repair': 1, 'functional': 2})

# Before imputing these, check if there are not so many unique values that we can't one-hot encode them
# This applies to non-numeric columns, so all except age
missing_values = X.isna().sum()[ X.isna().sum() > 0 ].index
missing_values = missing_values.drop('age')

cols_to_drop = []
for col in missing_values:
    if X[col].nunique() > 50:  # arbitrary threshold
        cols_to_drop.append(col)

X.drop(columns=cols_to_drop, inplace=True)
X_test.drop(columns=cols_to_drop, inplace=True)

# Then before imputation, split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Only 'age' is numerical, impute mean
imputer = SimpleImputer(strategy='mean')
X_train['age'] = imputer.fit_transform(X_train[['age']])
X_val['age'] = imputer.transform(X_val[['age']])
X_test['age'] = imputer.transform(X_test[['age']])

# For public_meeting, scheme_management, permit, we add the 'Unknown' category for missing values
missing_values = X_train.isna().sum()[ X_train.isna().sum() > 0 ].index

for col in missing_values:
    X_train[col] = X_train[col].fillna('Unknown')
    X_val[col] = X_val[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

# Next deal with the remaining object columns; they are all nominal
# Note that the region_code and district_code are numbers, but they are actually categorical, so we should one-hot encode them as well
object_cols = X_train.select_dtypes(include='object').columns
object_cols = object_cols.append(pd.Index(['region_code', 'district_code']))

X_train = pd.get_dummies(X_train, columns=object_cols, dtype=int)
X_val = pd.get_dummies(X_val, columns=object_cols, dtype=int)
X_test = pd.get_dummies(X_test, columns=object_cols, dtype=int)

# There seem to be 2 missing columns in the validation data
# These are both one-hot encoded columns, so we can just add them with all 0s
X_val['region_code_40'] = 0
X_val['extraction_type_other - mkulima/shinyanga'] = 0
X_val = X_val[X_train.columns]  # put columns in same order

# There seem to be 2 missing columns in X_test too, as it should have 1 more than X_train (id) but instead has 1 less
# These are both one-hot encoded columns, so we can just add them with all 0s
X_test['region_code_40'] = 0
X_test['extraction_type_other - mkulima/shinyanga'] = 0

# Put columns in same order, and keep id
id = X_test['id']
X_test = X_test[X_train.columns]
X_test['id'] = id

# Save the data
X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
X_val.to_csv(f'{output_dir}/X_val.csv', index=False)
X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
y_val.to_csv(f'{output_dir}/y_val.csv', index=False)