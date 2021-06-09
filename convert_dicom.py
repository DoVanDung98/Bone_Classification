import numpy as np 
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data import Dataset
from albumentations import *
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data    
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

class ChestDataset(Dataset):
    def __init__(self, file_paths, image_size):
        super(ChestDataset,self).__init__()
        self.file_paths = file_paths
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_name = self.file_paths[index].split('/')[-1]
        image = read_xray(self.file_paths[index])
        image = np.stack([image, image, image], axis=-1)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, file_name