from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os 
from PIL import Image
import pandas as pd
import shutil

class CustomDataset(Dataset):
    def __init__(self, dataframe=None, img_dir=None, transform=None, class_map=None):
        self.transform = transform
        if dataframe is not None:
            self.dataframe = dataframe
        elif img_dir is not None:
            img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
            labels = [os.path.basename(img_dir)] * len(img_paths)
            self.dataframe = pd.DataFrame({'Class Path': img_paths, 'Class': labels})
        else:
            raise ValueError("Either dataframe or img_dir must be provided")
        self.class_map = class_map  # Allow None for string labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.class_map:
            label = self.class_map[label]  # Convert to numerical label only if class_map provided
        return image, label

def augment_data(dir, class_list, n, crop=256, enable=False):
    tr_dr = os.path.join(dir, 'Training')
    for img_class in class_list:
        aug_dir = os.path.join(dir, 'aug_dir')
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.makedirs(img_dir, exist_ok=True)
        img_list = os.listdir(os.path.join(tr_dr, img_class))
        if not img_list:
            print(f"No images found in {os.path.join(tr_dr, img_class)}")
            continue
        for fname in img_list:
            src = os.path.join(tr_dr, img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)
        transform = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
        if enable:
            img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
            labels = [img_class] * len(img_paths)
            df = pd.DataFrame({'Class Path': img_paths, 'Class': labels})
            dataset = CustomDataset(dataframe=df, transform=transform, class_map=None)  # String labels
        else:
            dataset = CustomDataset(img_dir=img_dir, transform=transform, class_map=None)  # String labels
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        num_aug_images_wanted = n
        num_files = len(os.listdir(img_dir))
        num_batches = max(1, int((num_aug_images_wanted - num_files) / 50))
        save_path = os.path.join(tr_dr, img_class)
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            for j, img in enumerate(batch[0]):
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(save_path, f"aug_{i}_{j}.jpg"))
        shutil.rmtree(aug_dir)

if __name__ == '__main__':
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    augment_data('data', classes, n=1000, crop=256, enable=True)
