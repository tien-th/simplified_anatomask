import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from split_pet_data import split_pet_data

def process_image(image: np.ndarray, fix_depth=140):
    """
    Process the image from D x H x W to C x H x W x D
    - Resize the depth dimension to fix_depth using interpolation
    - Ensure fix_depth is divisible by 4 (pad if necessary)
    - Normalize pixel values by dividing by 32767
    - Convert image to (1, H, W, D) format
    
    Args:
        image (np.ndarray): The image with shape (D, H, W)
        fix_depth (int): The desired depth size
    
    Returns:
        torch.Tensor: Processed image with shape (1, H, W, D)
    """
    
    # Convert to torch tensor and normalize to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0

    # Reshape to (1, 1, D, H, W) for interpolation (N, C, D, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # Resize depth dimension using trilinear interpolation (D → fix_depth)
    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)

    # Remove batch dimensions → (1, fix_depth, H, W)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor

class MedicalImageReportDataset(Dataset):
    def __init__(self, vision_ssl_paths, image_text_pairs_path, split='train', transform=None):
        """
        Args:
            vision_ssl_paths (str): List of Path to the vision ssl folder (e.g., "./DAC001").
            image_text_pairs_path (str): List of Path to the image-text pairs folder (e.g., "./DAC001_CTAC3.75mm_H_1001_PETWB3DAC001").
            split (str): One of 'train', 'val', or 'test'.
                - train: use all month folders except THANG 10, THANG 11, THANG 12.
                - val: use only THANG 10.
                - test: use only THANG 11 and THANG 12.
            transform: Optional transform to be applied on a sample (e.g., conversion to torch tensor, normalization, etc.).
        """
        self.vision_ssl_paths = vision_ssl_paths
        self.image_text_pairs_path = image_text_pairs_path
        self.split = split.lower()
        self.transform = transform
        
        # Determine which month folders to include based on the split.
        self.month_folders = split_pet_data(self.vision_ssl_paths, self.image_text_pairs_path, split=self.split)
        
        # Allowed modalities (exclude "whole_body")
        allowed_modalities = ['abdomen_pelvis', 'chest', 'head_neck']
        
        # Build the list of (image_path, report_path) pairs.
        self.samples = []
        for month_folder in self.month_folders:
            images_root = os.path.join(month_folder, 'images')
            if not os.path.isdir(images_root):
                continue
            for modality in allowed_modalities:
                modality_img_folder = os.path.join(images_root, modality)

                if not os.path.isdir(modality_img_folder):
                    continue
                # List all image files ending with .npy
                image_files = sorted([f for f in os.listdir(modality_img_folder) if f.endswith('.npy')])
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    rep_file = base_name + '.txt'
                    img_file_path = os.path.join(modality_img_folder, img_file)
                    if os.path.exists(img_file_path):
                        self.samples.append(img_file_path)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        # Load the image data from a .npy file
        image = np.load(img_path)
        # Optionally apply a transform (if provided) or convert to a torch tensor.
        if self.transform:
            image = self.transform(image)
        else:
            image = process_image(image)
        # Load the report text.
        
        return image

