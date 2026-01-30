import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import SimpleCNN
from data_loader import get_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: CPU or GPU device
    
    Returns:
        average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc='Training')
    
    for images, labels in progress_bar:
        # Move data to device (CPU/GPU)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    average_loss = total_loss / len(train_loader)
    return average_loss


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: CPU or GPU device
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Don't compute gradients during evaluation
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return average_loss, accuracy


def test(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: CPU or GPU device
    
    Returns:
        accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_results(train_losses, val_losses, val_accuracies):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
    print("Training results saved as 'training_results.png'")
    plt.show()


def main():
    """
    Main training loop.
    """
    # Hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0  # Change to 2-4 on Linux/Mac for faster loading
    
    # Device setup (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Dataset loaded successfully!")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # Initialize model
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"\nModel initialized:")
    print(f"  - Architecture: SimpleCNN")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    print("\nStarting training...\n")
    best_val_accuracy = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%\n")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> Best model saved! Accuracy: {val_accuracy:.2f}%\n")
    
    # Load best model and test
    print("Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot results
    plot_results(train_losses, val_losses, val_accuracies)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Model saved as: best_model.pth")
    print("="*50)


if __name__ == "__main__":
    main()
