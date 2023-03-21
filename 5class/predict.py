import torch
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from unet import UNet
from nii_loader import predict_NiiLoader
import os
import matplotlib.pyplot as plt

# Define the path to the model and data
model_path = "/data/disk3/liangwei2/unet/output/weight_fold1/291model_weights.pth"
data_path = '/data/disk3/zhanghuiling/ICH_Slices_3d/fold1/'
# Load the model and send it to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=6).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the dataset for prediction
test_file_names = [f for f in os.listdir(os.path.join(data_path,'imagesTs')) if f.endswith('.nii.gz')]
test_loader = DataLoader(predict_NiiLoader(data_path, 'imagesTs',test_file_names), batch_size=1, shuffle=False)

# Define a function to predict on the test set
def predict(model, test_loader):
    model.eval()
    with torch.no_grad():
        for img in test_loader:
            img = img.cuda().float()
            pred = model(img)
            pred = pred.argmax(dim=1)
            pred = pred.detach().cpu().numpy()
            yield pred

# Predict on the test set and save the predictions as NIfTI files
for i, pred in enumerate(predict(model, test_loader)):
    img_file = nib.load(os.path.join(data_path,'imagesTs', test_file_names[i]))
    pred_nii = nib.Nifti1Image(pred.astype(np.uint8), img_file.affine, img_file.header)
    nib.save(pred_nii, os.path.join('/data/disk3/liangwei2/unet/predict_nii/weight_fold1/', test_file_names[i]))
