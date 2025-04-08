#!/bin/bash
# Bash script to rename the data files so that they can be read by the preprocessing script
# This assumes they are not renamed already after downloading
data_dir="$1"
cd "$data_dir"

# Check if there are three files in the directory
if [ $(ls -1 | wc -l) -ne 3 ]; then
    echo "Please download the data files from the link in the README and place them in the data directory."
    exit 1
fi

echo "Renaming data files to be read by the preprocessing script."
mv "702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv" "test_values.csv"
mv "0bf8bc6e-30d0-4c50-956a-603fc693d966.csv" "training_labels.csv"
mv "4910797b-ee55-40a7-8668-10efd5c1b960.csv" "training_values.csv"
