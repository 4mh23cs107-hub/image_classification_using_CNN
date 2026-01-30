import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import SimpleCNN


# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalization values for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def load_model(model_path='best_model.pth', device='cpu'):
    """
    Load the trained CNN model.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on
    
    Returns:
        model: Loaded CNN model
    """
    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path, device='cpu'):
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained CNN model
        image_path: Path to image file
        device: Device to use for inference
    
    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to 32x32 (CIFAR-10 size)
    img = img.resize((32, 32))
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = confidence.item() * 100
    probabilities = probabilities.squeeze().cpu().numpy()
    
    return predicted_class, confidence, probabilities


def visualize_predictions(model, image_path, device='cpu'):
    """
    Visualize the image and prediction results.
    
    Args:
        model: Trained CNN model
        image_path: Path to image file
        device: Device to use for inference
    """
    predicted_class, confidence, probabilities = predict_image(
        model, image_path, device
    )
    
    # Load image for visualization
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
    ax1.axis('off')
    
    # Display probabilities
    ax2.barh(CLASS_NAMES, probabilities)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"\nAll Probabilities:")
    for class_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"  {class_name}: {prob*100:.2f}%")


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model('best_model.pth', device=device)
    print("Model loaded successfully!")
    
    # To use this for inference on an image:
    # Replace 'path/to/image.jpg' with your actual image path
    # visualize_predictions(model, 'path/to/image.jpg', device=device)
