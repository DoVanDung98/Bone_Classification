import torch
from data_loader import to_device
import numpy as np 
from tqdm import tqdm
from convert_dicom import ChestDataset
from torch.utils.data import DataLoader
from utils.constants import IMAGE_SIZE, CLS, BATCH_SIZE
from utils.constants import IMAGE_CHANEL, NUM_OF_FEATURES, CHECKPOINT_PATH
from torchvision.datasets import ImageFolder
import cv2
import pandas as pd 

# ImageFile.LOAD_TRUNCATED_IMAGES = True
default_device = torch.device("cuda")

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), default_device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return CLS[preds[0].item()]

def get_ci(stats, a=0.95):
    p =((1.0-a)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (a + ((1.0 - a)/ 2.0)) * 100 #p is limits
    upper = min(1.0, np.percentile(stats,p))
    return lower, upper

def predict(file_path, model):
    model.cuda()
    model.eval()
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_true = []
    y_pred = []
    index = 0
    correct = 0
    wrong_total = 0
    test_dataset = ChestDataset(file_path, IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    items = []
    for image, file_name in tqdm(test_dataset):
        index += 1
        pred_label = predict_image(image, model)
        img = np.array(image)
        maxi = img.max()
        img = img*255./maxi
        img = img.transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(f'images/{index}_{pred_label}.jpg',img)
        item = (file_name, pred_label)
        items.append(item)
    df = pd.DataFrame(items, columns=['flie_name', 'pred_label']).to_csv('result.csv', index=None)
    