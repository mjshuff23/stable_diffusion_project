from src.utils.data_processing import CustomImageDataset
from torchvision import transforms

def main():
    # Define the path to the dataset
    dataset_path = "images/dragon_ball"

    # Define a simple transform (e.g., resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Initialize the dataset
    print(f"Reading images from: {dataset_path}")
    dataset = CustomImageDataset(image_dir=dataset_path, transform=transform)

    # Check the dataset size
    print(f"Total images found: {len(dataset)}")

    # Display some information about the first image
    if len(dataset) > 0:
        image = dataset[0]
        print("Successfully loaded the first image!")
        print(f"Image shape: {image.shape}")

if __name__ == "__main__":
    main()
