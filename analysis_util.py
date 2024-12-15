import json
import os


def load_and_extract_metrics(file_path):
    """
    Load a JSON file and extract metrics.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: A dictionary containing metrics for the file
    """
    # Initialize metrics dictionary
    file_metrics = {
        'file_path': file_path,
        'aggregated_metrics': [],
        'server_metrics': [],
        'best_aggregated_metric': None,
        'best_server_metric': None
    }

    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist.")
        return file_metrics

    # Read and parse the JSON file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Extract aggregated evaluate accuracy
            aggregated_eval = data.get('aggregated_evaluate_accuracy', [])
            file_metrics['aggregated_metrics'] = [
                entry['metrics']['accuracy'] for entry in aggregated_eval
            ]

            # Extract server evaluate accuracy
            server_eval = data.get('server_evaluate_accuracy', [])
            file_metrics['server_metrics'] = [
                entry['metrics']['accuracy'] for entry in server_eval
            ]

            # Find best metrics
            file_metrics['best_aggregated_metric'] = (
                max(file_metrics['aggregated_metrics'])
                if file_metrics['aggregated_metrics'] else None
            )
            file_metrics['best_server_metric'] = (
                max(file_metrics['server_metrics'])
                if file_metrics['server_metrics'] else None
            )

    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return file_metrics


def main():
    # Example usage - replace with your actual file paths
    paths = [
        'SimulationOutputs/num_epochs_1/metrics.json',
        'SimulationOutputs/num_epochs_2/metrics.json',
        'SimulationOutputs/num_epochs_3/metrics.json',
        'SimulationOutputs/num_epochs_4/metrics.json',
        'SimulationOutputs/num_epochs_5/metrics.json',
        'SimulationOutputs/num_epochs_10/metrics.json'
    ]

    # Load and extract metrics for each file
    all_file_metrics = []
    for path in paths:
        file_metrics = load_and_extract_metrics(path)
        all_file_metrics.append(file_metrics)

    # Print results for verification
    for metrics in all_file_metrics:
        print(f"\nMetrics for {metrics['file_path']}:")
        print("Aggregated Metrics:")
        print(f"  List: {metrics['aggregated_metrics']}")
        print(f"  Best Metric: {metrics['best_aggregated_metric']}")

        print("\nServer Metrics:")
        print(f"  List: {metrics['server_metrics']}")
        print(f"  Best Metric: {metrics['best_server_metric']}")


# Allow the script to be run directly or imported
if __name__ == "__main__":
    main()