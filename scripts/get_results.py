import json
import argparse


def compute_average_acc_none(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    acc_values = []

    for key, value in results.items():
        print(key)

        if "boolq" in key:
            if key == "indic_boolq":
                acc_values.append(value["acc,none"])
            elif isinstance(value, dict) and "acc,none" in value:
                acc_values.append(value["acc,none"])
                # continue
        else:
            if isinstance(value, dict) and "acc,none" in value:
                acc_values.append(value["acc,none"])
            elif isinstance(value, dict) and "exact_match,strict_match" in value:
                acc_values.append(value["exact_match,strict_match"])

    if not acc_values:
        print("No valid 'acc,none' entries found.")
        return None

    average = sum(acc_values) / len(acc_values)
    print(f"Average of 'acc,none': {average}")
    return average


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average of 'acc,none' from JSON results."
    )
    parser.add_argument("filename", help="Path to the JSON file")
    args = parser.parse_args()

    compute_average_acc_none(args.filename)
