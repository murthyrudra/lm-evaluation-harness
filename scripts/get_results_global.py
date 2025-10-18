import json
import argparse


def compute_average_acc_none(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    
    full = []
    lite = []

    for key, value in results.items():
        if key == "global_mmlu_bn":
            lite.append(value["acc,none"])
        elif key == "global_mmlu_hi":
            lite.append(value["acc,none"])
        elif key == "global_mmlu_full_bn":
            full.append(value["acc,none"])
        elif key == "global_mmlu_full_hi":
            full.append(value["acc,none"])
        elif key == "global_mmlu_full_te":
            full.append(value["acc,none"])
        
    full_average = sum(full) / len(full)
    lite_average = sum(lite) / len(lite)
    print(f"Full Average of 'acc,none': {full_average}")
    print(f"Lite Average of 'acc,none': {lite_average}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average of 'acc,none' from JSON results."
    )
    parser.add_argument("filename", help="Path to the JSON file")
    args = parser.parse_args()

    compute_average_acc_none(args.filename)
