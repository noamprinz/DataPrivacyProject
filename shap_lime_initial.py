import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

# LIME and SHAP libraries
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap


# Import the original model architecture
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_datasets():
    from flwr_datasets import FederatedDataset

    # Load the full CIFAR-10 test set
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    testset = fds.load_split("test")

    # Preprocessing transform
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    testset = testset.with_transform(apply_transforms)

    return testset


def prepare_torch_model(model_path):
    # Load the model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def lime_explanation(model, testset, num_samples=5):
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

    for i in range(num_samples):
        # Get image and label
        image_tensor = testset[i]["img"]
        true_label = testset[i]["label"]

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
        explanation = explainer.explain_instance(
            reverse_normalize(image_np),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )

        # Get the explanation for the predicted class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=True
        )

        # Plot original, segmented, and LIME explanation
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(reverse_normalize(image_np) / 255)
        plt.title(f'Original Image\nTrue Label: {testset.features["label"].int2str([true_label])[0]}')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mark_boundaries(reverse_normalize(temp) / 255, mask))
        plt.title('LIME Explanation')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.bar(range(10), F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy())
        plt.title('Class Probabilities')
        plt.xticks(range(10), testset.features["label"].names, rotation=45)

    plt.tight_layout()
    plt.savefig('lime_explanations.png')
    plt.close()


def custom_shap_explanation(model, testset, num_samples=5):
    """
    Generate SHAP-like explanations using gradient-based approach
    """
    plt.figure(figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        # Get image and label
        image_tensor = testset[i]["img"]
        true_label = testset[i]["label"]

        # Require gradients
        image_tensor.requires_grad_()

        # Forward pass
        output = model(image_tensor.unsqueeze(0))

        # Get the score for the true label
        score = output[0][true_label]

        # Compute gradients
        score.backward()

        # Get gradient as importance map
        saliency = image_tensor.grad.abs().mean(dim=0)

        # Convert tensor to numpy for visualization
        image_np = image_tensor.detach().permute(1, 2, 0).numpy()
        saliency_np = saliency.numpy()

        # Reverse normalization
        image_display = (image_np / 2 + 0.5)

        # Plot original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image_display)
        plt.title(f'Original Image\nTrue Label: {testset.features["label"].int2str([true_label])[0]}')
        plt.axis('off')

        # Plot saliency map
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(image_display)
        plt.imshow(saliency_np, cmap='hot', alpha=0.5)
        plt.title('Gradient-based Importance')
        plt.colorbar()

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.bar(range(10), F.softmax(model(image_tensor.unsqueeze(0)), dim=1)[0].detach().numpy())
        plt.title('Class Probabilities')
        plt.xticks(range(10), testset.features["label"].names, rotation=45)

    plt.tight_layout()
    plt.savefig('custom_shap_explanations.png')
    plt.close()


def main():
    # Load the test dataset
    testset = load_datasets()

    # Load the trained model (from the previous federation script)
    model_path = "Models/federated_model.pth"  # or "initial_model.pth"
    model = prepare_torch_model(model_path)

    # Generate LIME explanations
    lime_explanation(model, testset)
    print("LIME explanations saved to lime_explanations.png")

    # Generate SHAP explanations
    custom_shap_explanation(model, testset)
    print("SHAP explanations saved to shap_explanations.png")


if __name__ == "__main__":
    main()