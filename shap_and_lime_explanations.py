import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

from model import NewNet as Net
from data_util import load_local_datasets


def prepare_torch_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_representative_images(test_set, class_dict, seed=42):
    """
    Get one image from each class in the dataset

    Args:
        test_set: DataLoader containing test images
        class_dict: Dictionary mapping class indices to class names
        seed: Random seed for reproducibility

    Returns:
        List of tuples (image_tensor, label, image_index)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dictionary to store one example of each class
    class_examples = {}

    # Go through the dataset to find examples of each class
    for i, batch_dict in enumerate(test_set):
        for j, (image, label) in enumerate(zip(batch_dict["image"], batch_dict["label"])):
            label_item = label.item()
            if label_item not in class_examples:
                class_examples[label_item] = (image, label, i * len(batch_dict["label"]) + j)

            # Break if we found all classes
            if len(class_examples) == len(class_dict):
                break
        if len(class_examples) == len(class_dict):
            break

    # Sort by label to ensure consistent ordering
    return [class_examples[label] for label in sorted(class_examples.keys())]


def compare_shap_explanations(model_paths, model_names, test_set, output_dir, class_dict, representative_images):
    """
    Generate SHAP explanations comparing multiple models on multiple images
    """
    num_models = len(model_paths)
    assert num_models == len(model_names), "Number of model paths must match number of model names"

    # Load all models
    models = [prepare_torch_model(path) for path in model_paths]

    # Background dataset for SHAP (using first batch)
    background_batch = next(iter(test_set))["image"]
    background = background_batch[:10]

    # Process each representative image
    for image_tensor, label, img_idx in representative_images:
        # Convert tensor to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Create figure: original image + (SHAP + probs) for each model
        fig = plt.figure(figsize=(5 * (num_models + 1), 8))

        # Reverse normalization function
        def reverse_normalize(image):
            return (image / 2 + 0.5) * 255

        # Plot original image
        ax_orig = plt.subplot(2, num_models + 1, 1)
        ax_orig.imshow(reverse_normalize(image_np) / 255)
        ax_orig.set_title(f'Original Image\nTrue Label: {class_dict[label.item()]}')
        ax_orig.axis('off')

        # Generate explanations for each model
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            # SHAP explanation
            explainer_shap = shap.DeepExplainer(model, background)
            shap_values = explainer_shap.shap_values(image_tensor.unsqueeze(0))[0]
            shap_values = shap_values[:, :, :, label.item()]

            # Plot SHAP
            ax_shap = plt.subplot(2, num_models + 1, idx + 2)
            shap_importance = np.abs(shap_values).sum(axis=0)
            shap_importance = (shap_importance - shap_importance.min()) / (
                    shap_importance.max() - shap_importance.min()
            )
            im = ax_shap.imshow(shap_importance, cmap='viridis')
            ax_shap.set_title(f'{model_name}\nSHAP Importance')
            ax_shap.axis('off')
            plt.colorbar(im, ax=ax_shap, fraction=0.046, pad=0.04)

            # Plot probabilities
            ax_prob = plt.subplot(2, num_models + 1, num_models + idx + 3)
            probs = F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy()
            ax_prob.bar(range(len(class_dict)), probs)
            ax_prob.set_title(f'{model_name}\nClass Probabilities')
            ax_prob.set_xticks(range(len(class_dict)))
            ax_prob.set_xticklabels(
                [class_dict[i] for i in range(len(class_dict))],
                rotation=45,
                ha='right'
            )

        plt.tight_layout()

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save with descriptive filename
        output_path = os.path.join(
            output_dir,
            f'shap_comparison_class_{class_dict[label.item()]}_img_{img_idx}.png'
        )
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def compare_lime_explanations(model_paths, model_names, test_set, output_dir, class_dict, representative_images):
    """
    Generate LIME explanations comparing multiple models on multiple images
    """
    num_models = len(model_paths)
    assert num_models == len(model_names), "Number of model paths must match number of model names"

    # Load all models
    models = [prepare_torch_model(path) for path in model_paths]

    # Prepare LIME explainer
    explainer_lime = lime_image.LimeImageExplainer()

    # Process each representative image
    for image_tensor, label, img_idx in representative_images:
        # Convert tensor to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Create figure: original image + (LIME + probs) for each model
        fig = plt.figure(figsize=(5 * (num_models + 1), 8))

        # Reverse normalization function
        def reverse_normalize(image):
            return (image / 2 + 0.5) * 255

        # Plot original image
        ax_orig = plt.subplot(2, num_models + 1, 1)
        ax_orig.imshow(reverse_normalize(image_np) / 255)
        ax_orig.set_title(f'Original Image\nTrue Label: {class_dict[label.item()]}')
        ax_orig.axis('off')

        # Generate explanations for each model
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            # LIME explanation
            def predict_fn(images):
                images_tensor = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(images_tensor)
                return predictions.numpy()

            explanation = explainer_lime.explain_instance(
                image_np,
                classifier_fn=predict_fn,
                top_labels=1,
                num_features=1000
            )
            predicted_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                predicted_label,
                positive_only=True,
                num_features=5,
                hide_rest=True
            )

            # Plot LIME
            ax_lime = plt.subplot(2, num_models + 1, idx + 2)
            ax_lime.imshow(mark_boundaries(reverse_normalize(temp) / 255, mask))
            ax_lime.set_title(f'{model_name}\nLIME Explanation')
            ax_lime.axis('off')

            # Plot probabilities
            ax_prob = plt.subplot(2, num_models + 1, num_models + idx + 3)
            probs = F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy()
            ax_prob.bar(range(len(class_dict)), probs)
            ax_prob.set_title(f'{model_name}\nClass Probabilities')
            ax_prob.set_xticks(range(len(class_dict)))
            ax_prob.set_xticklabels(
                [class_dict[i] for i in range(len(class_dict))],
                rotation=45,
                ha='right'
            )

        plt.tight_layout()

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save with descriptive filename
        output_path = os.path.join(
            output_dir,
            f'lime_comparison_class_{class_dict[label.item()]}_img_{img_idx}.png'
        )
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def main():
    # Load the test dataset
    dataset_path = "Data/bccd_dataset"
    _, _, test_dataset, class_dict = load_local_datasets(0, dataset_path, 10)

    model_paths = [
        "SimulationOutputs/try_num_epochs_1/0_model.pth",
        "SimulationOutputs/try_num_epochs_1/0_model.pth"
    ]

    model_names = [
        "Model 1",
        "Model 2"
    ]

    output_dir = "Outputs/model_comparison"
    # Get representative images once
    representative_images = get_representative_images(test_dataset, class_dict)

    # Generate separate SHAP and LIME explanations using the same images
    compare_shap_explanations(
        model_paths,
        model_names,
        test_dataset,
        output_dir,
        class_dict,
        representative_images
    )
    print("SHAP model comparisons saved")

    compare_lime_explanations(
        model_paths,
        model_names,
        test_dataset,
        output_dir,
        class_dict,
        representative_images
    )
    print("LIME model comparisons saved")


if __name__ == "__main__":
    main()