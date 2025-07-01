import yaml
from datasets import load_dataset

# Load the YAML file
with open('template.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the dataset splits from Hugging Face
language_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
task_list = []

for task in language_list:
    dataset = load_dataset('sarvamai/trivia-qa-indic-mcq', task)

    config['task'] = f'trivia-qa-indic-mcq-{task}'
    config['dataset_name'] = f'{task}'
    task_list.append(config['task'])

    # Save the updated YAML (can create multiple files for each split if needed)
    with open(f'trivia-qa-indic-mcq-{task}.yaml', 'w') as file:
        yaml.dump(config, file)

    print(f'Created config_{task}.yaml with task {config["task"]} and dataset_name {config["dataset_name"]}')


# YAML structure
yaml_data = {
    'group': 'trivia-qa-indic-mcq',
    'task': task_list,
    'metadata': {
        'version': 0.0
    }
}

# Write to gsm8k_indic.yaml
with open('trivia-qa-indic-mcq.yaml', 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("✅ trivia-qa-indic-mcq.yaml generated successfully.")
