import os 

def split_pet_data(vision_ssl_paths, image_text_pairs_paths, split='train'):
    month_folders = []

    for vision_path in vision_ssl_paths:
        for month in os.listdir(vision_path):
            month_path = os.path.join(vision_path, month)
            if not os.path.isdir(month_path):
                continue
            if split in ['val', 'test']:
                continue
            else:
                month_folders.append(month_path)
        
    for root in image_text_pairs_paths:
        for month in os.listdir(root):
            month_path = os.path.join(root, month)
            if not os.path.isdir(month_path):
                continue
            if split == 'train':
                if month in ['THANG 10', 'THANG 11', 'THANG 12']:
                    continue
                else:
                    month_folders.append(month_path)
            elif split == 'val':
                if month == 'THANG 10':
                    month_folders.append(month_path)
            elif split == 'test':
                if month in ['THANG 11', 'THANG 12']:
                    month_folders.append(month_path)

    return month_folders
