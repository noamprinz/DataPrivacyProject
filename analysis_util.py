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