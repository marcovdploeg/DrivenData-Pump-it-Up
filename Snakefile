# Before running this script, make sure the data is downloaded into the 'data' directory.
# Run using the command (-n for dry-run): snakemake --cores 8 -n

# Rule with standard run, which will be run when no rule is specified
rule run:
    input: "submissions/submission_xgboost.csv"
    # denote the submissions we want to generate; either put xgboost or tree for the wildcard

# Preprocessing (and renaming) step
rule preprocess:
    input: 
        input_dir = "data",
        rename_script = "rename_data.sh",
        preprocess_script = "prep_scripts/preprocessing.py"
    output: directory("prep_data")
    shell:
        """
        mkdir -p {output}
        bash {input.rename_script} {input.input_dir}
        python {input.preprocess_script} -i {input.input_dir} -o {output}
        """

# Rule to run each model
rule model:
    input: 
        preprocessed_data = rules.preprocess.output,
        model_script = "model_py_scripts/model_{model}.py"
    output: "submissions/submission_{model}.csv"
    shell:
        """
        mkdir -p submissions
        python {input.model_script} -i {input.preprocessed_data} -o {output}
        """