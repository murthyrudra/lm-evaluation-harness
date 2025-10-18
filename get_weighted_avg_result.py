import json
import argparse
import sys

def compute_weighted_average(json_path,metric):
    '''
    Inputs -
    json_path : path to the json file to be considered for calculating weighted avg.
    metric : name of the metric key as in the json file that needs to extracted for weighted avg.

    Returns -
    weighted_avg : weighted average of the metrics where the weight of each subset is proportional to the number of entries in that subset.
    '''
    """Compute weighted average of exact_match,strict_match using n-samples as weights."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at path '{json_path}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", {})
    n_samples = data.get("n-samples", {})

    total_weighted_score = 0.0
    total_samples = 0
    valid_entries = []

    for key, value in results.items():
        score = value.get(metric)
        if score is None:
            # Skip empty or invalid entries (no score)
            continue

        # Fetch the number of samples for this subset
        sample_info = n_samples.get(key, {})
        samples = sample_info.get("effective") or sample_info.get("original")

        if not samples:
            # Skip entries without sample count
            continue

        total_weighted_score += score * samples
        total_samples += samples
        valid_entries.append((key, score, samples))

    if total_samples == 0:
        print("No valid subsets found for weighted average.")
        sys.exit(0)

    weighted_avg = total_weighted_score / total_samples

    print("\nSubset Summary:")
    for k, s, n in valid_entries:
        print(f" - {k}: score={s:.6f}, samples={n}")

    print(f"\nWeighted Average (exact_match,strict_match): {weighted_avg:.6f}")
    return weighted_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute weighted average of exact_match,strict_match scores from a JSON file."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the evaluation JSON file."
    )
    parser.add_argument(
        "metric",
        type=str,
        help="metric to be extracted.",
        default="exact_match,strict_match"
    )

    args = parser.parse_args()
    compute_weighted_average(args.json_path,args.metric)


'''
USAGE : python3 get_weighted_avg_result.py path_to_json_file key_of_metric_to_be_extracted
where key_of_metric_to_be_extracted = 'exact_match,strict_match' for generation based eval and 'acc,none' for log-likelihood based evals and so on.
'''