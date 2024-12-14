import os
from typing import List, Tuple
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class CustomImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Root directory containing images and subdirectories.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self._get_all_images(image_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in directory {image_dir}.")

    def _get_all_images(self, root_dir: str) -> List[str]:
        """
        Recursively gather all image paths from root_dir.

        Args:
            root_dir (str): Root directory to search for images.

        Returns:
            List[str]: List of all image file paths.
        """
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(subdir, file))
        print(f"Total images found: {len(image_paths)}")  # Log the number of images
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise RuntimeError(f"Corrupted image: {img_path}")
        if self.transform:
            image = self.transform(image)
        return image
    
def get_data_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        data_dir (str): Root directory containing images (can have subdirectories).
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    # Define transformations for the images
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load the dataset recursively with appropriate transforms
    dataset = CustomImageDataset(image_dir=data_dir, transform=train_transform)  # For training

    # Split into training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Ensure validation dataset uses the validation transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader