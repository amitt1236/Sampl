from costumDataset import UltrasoundDataset, Rescale, RandomCrop, ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

directory = "put something"

transformed_dataset = UltrasoundDataset(root_dir=directory,
                                        transform=transforms.Compose([
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))

# Can use dataLoader *or* send the entire dataset to device if small enough
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
