import yaml
from datasets import load_dataset

# Load the YAML file
with open('template.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the dataset splits from Hugging Face
language_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
task_list = []

for task in language_list:
    dataset = load_dataset('sarvamai/arc-challenge-indic', task)

    config['task'] = f'arc-challenge-indic-{task}-gpt-oss'
    config['dataset_name'] = f'{task}'
    task_list.append(config['task'])

    # Save the updated YAML (can create multiple files for each split if needed)
    with open(f'arc-challenge-indic-{task}-gpt-oss.yaml', 'w') as file:
        yaml.dump(config, file)

    print(f'Created config_{task}.yaml with task {config["task"]} and dataset_name {config["dataset_name"]}')


# YAML structure
yaml_data = {
    'group': 'arc-challenge-indic-gpt-oss',
    'task': task_list,
    'metadata': {
        'version': 0.0
    }
}

# Write to gsm8k_indic.yaml
with open('arc-challenge-indic-gpt-oss.yaml', 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("✅ arc-challenge-indic.yaml generated successfully.")
