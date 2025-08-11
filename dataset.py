import os
import numpy as np
import cv2  
from torch.utils.data import Dataset

class RoadExtractionDataset(Dataset):
    def __init__(self, root_sat, root_mask, transform=None):
        self.root_sat = root_sat
        self.root_mask = root_mask
        
        self.images = sorted(os.listdir(root_sat))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        mask_filename = img_filename.replace('sat.jpg', 'mask.png')
        
        
        img_path = os.path.join(self.root_sat, img_filename)
        mask_path = os.path.join(self.root_mask, mask_filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        mask[mask > 0] = 1.0

        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        
        
        return image, mask
    
