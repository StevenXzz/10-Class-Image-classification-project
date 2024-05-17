from torch.utils.data import Dataset

class Cifar10(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (Tensor): Image data.
            labels (Tensor): label data.
            transform (callable, optional): Optional conversions applied to each sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
