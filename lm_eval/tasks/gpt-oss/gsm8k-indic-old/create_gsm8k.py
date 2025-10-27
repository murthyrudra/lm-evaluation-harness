import yaml
from datasets import load_dataset

# Load the YAML file
with open("template.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the dataset splits from Hugging Face
language_list = [
    "bn",
    "bn_roman",
    "gu",
    "gu_roman",
    "hi",
    "hi_roman",
    "kn",
    "kn_roman",
    "ml",
    "ml_roman",
    "mr",
    "mr_roman",
    "or",
    "or_roman",
    "pa",
    "pa_roman",
    "ta",
    "ta_roman",
    "te",
    "te_roman",
]
task_list = []
for task in language_list:
    dataset = load_dataset("sarvamai/gsm8k-indic", task)

    config["task"] = f"gsm8k-indic-{task}"
    config["dataset_name"] = f"{task}"

    config["task"] = config["task"].replace("_", "-")

    task_list.append(config["task"])

    # Save the updated YAML (can create multiple files for each split if needed)
    with open(f"gsm8k-indic-{task}.yaml", "w") as file:
        yaml.dump(config, file)

    print(
        f'Created config_{task}.yaml with task {config["task"]} and dataset_name {config["dataset_name"]}'
    )


# YAML structure
yaml_data = {"group": "gsm8k_indic", "task": task_list, "metadata": {"version": 0.0}}

# Write to gsm8k_indic.yaml
with open("gsm8k_indic.yaml", "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("✅ gsm8k_indic.yaml generated successfully.")
