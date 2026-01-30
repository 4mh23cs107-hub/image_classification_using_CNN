import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=32, num_workers=2):
    """
    Load CIFAR-10 dataset with appropriate transformations.
    
    CIFAR-10 contains 60,000 32x32 color images in 10 classes:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    Args:
        batch_size: Number of images per batch
        num_workers: Number of processes for data loading
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        num_classes: Number of classes (10 for CIFAR-10)
    """
    
    # Data normalization values for CIFAR-10
    # These are the mean and std of the CIFAR-10 dataset
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    
    # Transformations for training data (includes augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
        transforms.RandomHorizontalFlip(),      # Randomly flip horizontally
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)  # Normalize
    ])
    
    # Transformations for validation/test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Download and load CIFAR-10 training set
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load CIFAR-10 test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training data into train and validation (90-10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, 10
