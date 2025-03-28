
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms

# Function to load CelebA dataset with specific attributes
def get_celeba_dataloader(root_dir="./data", batch_size=64, selected_attributes=None, image_size=128, num_workers=4, split='train'):
    """
    Creates dataloader for the CelebA dataset.

    Parameters:
        root_dir (str): Directory where CelebA dataset will be downloaded/stored.
        batch_size (int): Batch size for the dataloaders.
        selected_attributes (list): List of attribute names to extract.
        image_size (int): Image resizing size.
        num_workers (int): Number of workers for data loading.
        split (string): data split we want ('train', 'test', or 'valid')

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    
    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # Load the full CelebA dataset
    dataset = CelebA(root=root_dir, split=split, download=True, transform=transform)

    # Get attribute names
    attr_names = dataset.attr_names[:40]

    # Select specific attributes
    if selected_attributes:
        indices = [attr_names.index(attr) for attr in selected_attributes]
    else:
        indices = list(range(len(attr_names)))  # Use all attributes if none are selected

    # Get image indices for each split
    # Create subsets for train, validation, and test

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return dataloader, attr_names, indices