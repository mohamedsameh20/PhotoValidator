import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader

from wmdetection.utils import read_image_rgb


class ImageDataset(Dataset):
    
    def __init__(self, objects, classifier_transforms):
        self.objects = objects
        self.classifier_transforms = classifier_transforms
        
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]
        assert isinstance(obj, (str, np.ndarray, Image.Image))
        
        if isinstance(obj, str):
            pil_img = read_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            pil_img = Image.fromarray(obj)
        elif isinstance(obj, Image.Image):
            pil_img = obj
        
        resnet_img = self.classifier_transforms(pil_img).float()
        
        return resnet_img
    
    
class WatermarksPredictor:
    
    def __init__(self, wm_model, classifier_transforms, device):
        self.wm_model = wm_model
        self.wm_model.eval()
        self.classifier_transforms = classifier_transforms
        
        self.device = device
        
    def predict_image(self, pil_image):
        pil_image = pil_image.convert("RGB")
        input_img = self.classifier_transforms(pil_image).float().unsqueeze(0)
        outputs = self.wm_model(input_img.to(self.device))
        
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence_scores = probs.cpu().reshape(-1).tolist()
        
        # Get prediction (0 = clean, 1 = watermarked)
        result = torch.max(outputs, 1)[1].cpu().reshape(-1).tolist()[0]
        
        return {
            'prediction': result,
            'confidence_clean': confidence_scores[0],
            'confidence_watermarked': confidence_scores[1],
            'raw_outputs': outputs.cpu().reshape(-1).tolist()
        }
        
    def run(self, files, num_workers=8, bs=8, pbar=True):
        eval_dataset = ImageDataset(files, self.classifier_transforms)
        loader = DataLoader(
            eval_dataset,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            batch_size=bs,
            drop_last=False,
            num_workers=num_workers
        )
        if pbar:
            loader = tqdm(loader)
        
        result = []
        for batch in loader:
            with torch.no_grad():
                outputs = self.wm_model(batch.to(self.device))
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence_scores = probs.cpu().tolist()
                
                # Get predictions
                predictions = torch.max(outputs, 1)[1].cpu().tolist()
                raw_outputs = outputs.cpu().tolist()
                
                # Combine results
                for i, pred in enumerate(predictions):
                    result.append({
                        'prediction': pred,
                        'confidence_clean': confidence_scores[i][0],
                        'confidence_watermarked': confidence_scores[i][1],
                        'raw_outputs': raw_outputs[i]
                    })
        
        return result