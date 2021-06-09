import os 
import numpy as np 
import pydicom
import cv2
from matplotlib import pyplot as plt 
from pydicom.pixel_data_handlers.util import apply_voi_lut
from model import MobileNetV1
from utils import constants, get_default_device
from predict import predict

default_device = get_default_device.default_device
# file_paths = "samples"
if __name__ == '__main__':
    chestxray = MobileNetV1(constants.IMAGE_CHANEL, constants.NUM_OF_FEATURES)
    file_paths= []
    for rdir, _, files in os.walk('samples'):
        for idx, file in enumerate(files):
            if '.dcm' not in file and '.dicom' not in file:
                continue
            file_path = os.path.join(rdir, file)
            file_paths.append(file_path)
    outputs = predict(file_paths, chestxray)