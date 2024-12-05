import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

from model import NewNet as Net
from data_util import load_local_datasets


def prepare_torch_model(model_path):
    # Load the model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def lime_explanation(model, test_set, output_dir, class_dict, num_samples=5):
    """
    Generate LIME explanations for model predictions
    """
    # Prepare LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Reverse normalization function
    def reverse_normalize(image):
        # Reverse the normalization done earlier
        return (image / 2 + 0.5) * 255

    plt.figure(figsize=(15, 3 * num_samples))

    batch_dict = next(iter(test_set))

    for i in range(num_samples):
        # Get image and label
        image_tensor = batch_dict["image"][i]
        true_label = batch_dict["label"][i]

        # Convert tensor to numpy for LIME
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Prediction function for LIME
        def predict_fn(images):
            # Convert images back to tensor
            images_tensor = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
            with torch.no_grad():
                predictions = model(images_tensor)
            return predictions.numpy()

        # Generate explanation
        # Generate LIME explanation
        explanation = explainer.explain_instance(image_np, classifier_fn=predict_fn, top_labels=1, num_features=1000)
        # Get the explanation for the predicted class
        predicted_label = explanation.top_labels[0] #TODO: REPLACE?
        temp, mask = explanation.get_image_and_mask(predicted_label, positive_only=True, num_features=5, hide_rest=True)

        # Plot original, segmented, and LIME explanation
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(reverse_normalize(image_np) / 255)
        plt.title(f'Original Image\nTrue Label: {class_dict[true_label.item()]}')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mark_boundaries(reverse_normalize(temp) / 255, mask))
        plt.title('LIME Explanation')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.bar(range(10), F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy())
        plt.title('Class Probabilities')
        labels_list = [class_dict[i] for i in range(4)]
        # extend the list to 10 elements
        labels_list.extend(['' for _ in range(6)])
        # TODO: REMOVE
        plt.xticks(range(10), labels_list, rotation=45)

    plt.tight_layout()
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'lime_explanations.png')
    plt.savefig(output_path)
    plt.close()


def shap_explanation(model, test_set, output_dir, class_dict, num_samples=5):
    """
    Generate SHAP explanations for model predictions on image data

    Args:
        model: PyTorch model for image classification
        test_set: DataLoader containing test images
        output_dir: Directory to save explanation plots
        class_dict: Dictionary mapping class indices to class names
        num_samples: Number of images to generate explanations for
    """

    # Reverse normalization function
    def reverse_normalize(image):
        # Reverse the normalization done earlier
        return (image / 2 + 0.5) * 255

    # Background dataset for SHAP (using first batch as background)
    background_batch = next(iter(test_set))["image"]
    background = background_batch[:10]  # Use first 10 images as background

    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)

    plt.figure(figsize=(15, 3 * num_samples))

    batch_dict = next(iter(test_set))

    for i in range(num_samples):
        # Get image and label
        image_tensor = batch_dict["image"][i]
        true_label = batch_dict["label"][i]

        # Convert tensor to numpy for visualization
        # Permute to change from (C, H, W) to (H, W, C) for matplotlib
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Compute SHAP values
        shap_values = explainer.shap_values(image_tensor.unsqueeze(0))
        shap_values = shap_values[0]
        # get the shap values for the true label by last axis
        shap_values = shap_values[:, :, :, true_label.item()]
        # Permute SHAP values to match image dimensions
        if shap_values.ndim == 3:
            shap_values = shap_values.transpose(1, 2, 0)
        # Plot original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(reverse_normalize(image_np) / 255)
        plt.title(f'Original Image\nTrue Label: {class_dict[true_label.item()]}')
        plt.axis('off')
        # Plot SHAP heatmap
        plt.subplot(num_samples, 3, i * 3 + 2)
        # Ensure correct format for SHAP image plot
        shap.image_plot(shap_values, image_np, show=False)
        plt.title('SHAP Explanation')

        # Plot class probabilities
        plt.subplot(num_samples, 3, i * 3 + 3)
        probabilities = F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy()
        plt.bar(range(4), probabilities)
        plt.title('Class Probabilities')
        # Prepare labels
        labels_list = [class_dict[i] for i in range(4)]
        plt.xticks(range(4), labels_list, rotation=45)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'shap_explanations.png')
    plt.savefig(output_path)
    plt.close()


def shap_explanation_improved(model, test_set, output_dir, class_dict, num_samples=5):
    """
    Generate SHAP explanations for model predictions on image data

    Args:
        model: PyTorch model for image classification
        test_set: DataLoader containing test images
        output_dir: Directory to save explanation plots
        class_dict: Dictionary mapping class indices to class names
        num_samples: Number of images to generate explanations for
    """

    # Reverse normalization function
    def reverse_normalize(image):
        # Reverse the normalization done earlier
        return (image / 2 + 0.5) * 255

    # Background dataset for SHAP (using first batch as background)
    background_batch = next(iter(test_set))["image"]
    background = background_batch[:10]  # Use first 10 images as background

    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)

    # Create a figure with subplots for each image
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    batch_dict = next(iter(test_set))

    for i in range(num_samples):
        # Get image and label
        image_tensor = batch_dict["image"][i]
        true_label = batch_dict["label"][i]

        # Convert tensor to numpy for visualization
        # Permute to change from (C, H, W) to (H, W, C) for matplotlib
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Compute SHAP values for the entire image
        shap_values = explainer.shap_values(image_tensor.unsqueeze(0))[0]
        # get the shap values for the true label by last axis
        shap_values = shap_values[:, :, :, true_label.item()]

        # Original Image
        ax_orig = axes[i, 0]
        ax_orig.imshow(reverse_normalize(image_np) / 255)
        ax_orig.set_title(f'Original Image\nTrue Label: {class_dict[true_label.item()]}')
        ax_orig.axis('off')

        # SHAP Heatmap (simplified)
        ax_shap = axes[i, 1]
        # Take absolute values and sum across color channels to get overall importance
        shap_importance = np.abs(shap_values).sum(axis=0)

        # Normalize for better visualization
        shap_importance = (shap_importance - shap_importance.min()) / (shap_importance.max() - shap_importance.min())

        # Create a heatmap
        im = ax_shap.imshow(shap_importance, cmap='viridis')
        ax_shap.set_title('SHAP Feature Importance')
        ax_shap.axis('off')
        plt.colorbar(im, ax=ax_shap, fraction=0.046, pad=0.04)

        # Class Probabilities
        ax_prob = axes[i, 2]
        probabilities = F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy()
        ax_prob.bar(range(len(class_dict)), probabilities)
        ax_prob.set_title('Class Probabilities')
        ax_prob.set_xticks(range(len(class_dict)))
        ax_prob.set_xticklabels([class_dict[i] for i in range(len(class_dict))], rotation=45, ha='right')

    plt.tight_layout()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'shap_explanations.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the test dataset
    dataset_path = "Data/bccd_dataset"
    _, _, test_dataset, class_dict = load_local_datasets(0, dataset_path, 10) # number of partitions isn't used here
    # Load the trained model (loading model trained in main.py)
    model_path = "Models/NewNet_5_partitions/1_model.pth"
    output_dir = "Outputs/try"
    model = prepare_torch_model(model_path)

    # Generate LIME explanations
    # lime_explanation(model, test_dataset, output_dir, class_dict)
    print("LIME explanations saved to lime_explanations.png")

    # Generate SHAP explanations
    shap_explanation_improved(model, test_dataset, output_dir, class_dict)
    print("SHAP explanations saved to shap_explanations.png")


if __name__ == "__main__":
    main()