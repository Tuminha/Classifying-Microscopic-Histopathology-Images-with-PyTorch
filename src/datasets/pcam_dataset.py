"""
PCamDataset: Custom PyTorch Dataset for PatchCamelyon images.

This dataset loads 96x96 RGB histopathology patches and their binary labels
(0=Normal, 1=Tumor) from a CSV file.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class PCamDataset(Dataset):
    """
    Custom Dataset for PatchCamelyon (PCam) images.
    
    Args:
        csv_file (str): Path to CSV file with columns ['image_id', 'label']
                        Example: 'data/train_labels.csv'
        img_dir (str): Directory containing images (default: '../data/pcam_images/')
        transform (callable, optional): Optional transform to apply to images
    
    CSV Format:
        image_id,label
        abc123.png,1
        def456.png,0
        ...
    
    Returns:
        tuple: (image, label) where:
            - image is a transformed tensor [C, H, W]
            - label is a scalar (0 or 1)
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ... ])
        >>> dataset = PCamDataset(
        ...     csv_file='data/train_labels.csv',
        ...     transform=transform
        ... )
        >>> image, label = dataset[0]
        >>> print(image.shape, label)
        torch.Size([3, 96, 96]) 1
    """
    
    def __init__(self, csv_file, img_dir='../data/pcam_images/', transform=None):
        """
        Initialize the PCam dataset.
        
        Args:
            csv_file (str): Path to CSV file
            img_dir (str): Directory containing images
            transform (callable): Optional transforms
        """
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Validate CSV has required columns
        required_columns = {'image_id', 'label'}
        if not required_columns.issubset(self.labels_df.columns):
            raise ValueError(
                f"CSV must contain columns: {required_columns}. "
                f"Found: {set(self.labels_df.columns)}"
            )
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (image, label)
                - image: Transformed PIL Image or tensor [C, H, W]
                - label: Binary label (0 or 1) as int
        
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image filename and label from DataFrame
        img_name = self.labels_df.iloc[idx]['image_id']
        label = self.labels_df.iloc[idx]['label']
        
        # Construct full path
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}. "
                f"Check that img_dir='{self.img_dir}' is correct."
            )
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """
        Returns the distribution of classes in the dataset.
        
        Returns:
            dict: {'Normal': count, 'Tumor': count}
        
        Example:
            >>> dataset.get_class_distribution()
            {'Normal': 89117, 'Tumor': 96888}
        """
        label_counts = self.labels_df['label'].value_counts().to_dict()
        return {
            'Normal': label_counts.get(0, 0),
            'Tumor': label_counts.get(1, 0)
        }


# Example usage (not executed when imported)
if __name__ == "__main__":
    from torchvision import transforms
    
    # Define a simple transform
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset instance
    dataset = PCamDataset(
        csv_file='../data/train_labels.csv',
        img_dir='../data/pcam_images/',
        transform=transform
    )
    
    # Test loading
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Load first sample
    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label} ({'Tumor' if label == 1 else 'Normal'})")

