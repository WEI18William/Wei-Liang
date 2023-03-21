import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import nibabel as nib
import numpy as np
import os

class NiiLoader(Dataset):
    def __init__(self, data_path, img_dir, label_dir, file_names,file_label_names):
        self.data_path = data_path
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.file_names = file_names
        self.file_label_names = file_label_names
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Loading the image and label files

        img_file = nib.load(os.path.join(self.data_path, self.img_dir, self.file_names[index]))
        label_file = nib.load(os.path.join(self.data_path, self.label_dir, self.file_label_names[index]))
        # Converting the image and label data to numpy arrays
        img_data = img_file.get_fdata()
        label_data = label_file.get_fdata()

        # Normalizing the image data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

        # Applying the transform to the image data
        img_data = self.transform(img_data)

        # Applying one-hot encoding to the label data
        label_data = np.eye(6)[label_data.astype(int)]
        label_data = np.transpose(label_data, (3, 0, 1, 2))  # change the dimensions to match output of the model

        # Converting the label data to a tensor
        label_data = torch.tensor(label_data, dtype=torch.float32)

        return img_data, label_data

class predict_NiiLoader(Dataset):
    def __init__(self, data_path, img_dir, file_names):
        self.data_path = data_path
        self.img_dir = img_dir

        self.file_names = file_names
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Loading the image and label files

        img_file = nib.load(os.path.join(self.data_path, self.img_dir, self.file_names[index]))

        # Converting the image and label data to numpy arrays
        img_data = img_file.get_fdata()


        # Normalizing the image data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

        # Applying the transform to the image data
        img_data = self.transform(img_data)



        return img_data