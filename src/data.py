from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # reorder the label of data set
    def __init__(self, root, transform):
        super(ImageFolderWithPaths, self).__init__(root, transform)
        for key, _ in self.class_to_idx.items():
            if key != 'test':
                self.class_to_idx[key] = int(key)
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            if path[path.rfind("/")+1:path.rfind("_")] != 'test':
                target = int(path[path.rfind("/")+1:path.rfind("_")])

        return sample, target, path

def load_data(data_dir = "../data/", input_size = 224, batch_size = 36):
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(input_size),
            #transforms.RandomResizedCrop(input_size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_dataset_train = ImageFolderWithPaths(os.path.join(data_dir, 'train'), data_transforms['train'])
    image_dataset_valid = ImageFolderWithPaths(os.path.join(data_dir, 'val'), data_transforms['valid'])
    image_dataset_test = ImageFolderWithPaths(os.path.join(data_dir, 'test'), data_transforms['test'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(image_dataset_test, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, valid_loader, test_loader
