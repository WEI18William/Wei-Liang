import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import re
import os
import random
# Importing the UNet model and NiiLoader
from unet import UNet
from nii_loader import NiiLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="Specify the fold to use (0-4)")
parser.add_argument("--gpu", type=int, default=0,
                    help="Specify the GPU device ID to use")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print(f"Using fold {args.fold}")
print(f'{args.gpu}')

# Defining the training parameters
batch_size = 2
learning_rate = 1e-4
num_epochs = 100
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# Set device to the second GPU

# Defining the data paths
data_path = f'/data/disk3/zhanghuiling/ICH_Slices_3d/fold{args.fold}/'
img_dir = 'imagesTr'
label_dir = 'labelsTr'

# Creating a list of all the image and label file names
file_names = os.listdir(os.path.join(data_path, img_dir))

# Shuffling the file names list
random.shuffle(file_names)

# Splitting the file names list into training and validation sets
split = int(0.8 * len(file_names))
train_files = file_names[:split]
train_label_list = [re.sub(r'_0000', '', s) for s in train_files]
val_files = file_names[split:]
val_label_list = [re.sub(r'_0000', '', s) for s in val_files]

# Creating the training and validation data loaders
train_dataset = NiiLoader(data_path, img_dir, label_dir, train_files,train_label_list)
val_dataset = NiiLoader(data_path, img_dir, label_dir, val_files,val_label_list)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Creating the UNet model and sending it to the device
model = UNet(1,6).to(device)

# Defining the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.float().cuda()
        labels = labels.float().cuda()
        labels = torch.squeeze(labels, dim=-1)


        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Training Loss: {epoch_loss:.4f}')

    # Validation phase
    model.eval()
    running_loss = 0.0
    for images, labels in tqdm(val_loader):
        images = images.cuda().float()
        labels = labels.cuda().float()
        labels = torch.squeeze(labels, dim=-1)


        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)



        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(val_dataset)

    print(f'Validation Loss: {epoch_loss:.4f}')
    torch.save(model.state_dict(), os.path.join(f'/data/disk3/liangwei2/unet/output/weight_fold{args.fold}/',str(epoch) + 'model_weights.pth'))
