import torch
from models.seq_dual_svqvae import SequentialDualSVQVAE
from torchvision import transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from data.stats import data
from data.constants import *
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from types import SimpleNamespace
import os

def denormalize_img(img, mean, std):
    for t,m,s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    img = torch.clamp(img, 0,1)
    return img.permute((1,2,0))

def get_wbc_dataset_with_masks(type):
    mean = data['train']['mean']
    std = data['train']['std']
    
    assert type != 'val', 'no masks in validation set'

    spatial_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180, fill=mean),
        transforms.Resize((512,512), antialias=True)
    ])
    image_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
        
    data_path = data['train']['paths'][type]
    m = data['train']['paths'][type].split('/')
    m[-1] = 'mask'
    mask_path = os.path.join(*m)
    dataset = MaskDataset(data_root=data_path, mask_root=mask_path, 
                         spatial_transform=spatial_transform, image_transform=image_transform)
    return dataset, mean, std

def get_wbc_dataset(type):
    mean = data['train']['mean']
    std = data['train']['std']
    
    if type == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    
    path = data['train']['paths'][type]
    dataset = ImageFolder(root=path, transform=transform)
    return dataset

class MaskDataset(Dataset):
    def __init__(self, data_root, mask_root, spatial_transform, image_transform):
        self.data_root = data_root
        self.mask_root = mask_root
        self.spatial_transform = spatial_transform
        self.image_transform = image_transform
        self.samples = []
        self.targets = []

        # Iterate over each class folder
        classes_folder = sorted([i for i in os.listdir(data_root) if not i.startswith('.')])
        for label_idx, class_folder in enumerate(classes_folder):
            if class_folder.startswith('.'):
                continue
            image_class_dir = os.path.join(data_root, class_folder)
            mask_class_dir = os.path.join(mask_root, class_folder)
            for image_name in os.listdir(image_class_dir):
                if image_name.startswith('.'):
                    continue
                self.samples.append((image_class_dir, mask_class_dir, image_name, label_idx))
                self.targets.append(label_idx)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_class_dir, mask_class_dir, image_name, label = self.samples[idx]
        
        img_path = os.path.join(image_class_dir, image_name)
        mask_path = os.path.join(mask_class_dir, image_name)

        image = Image.open(img_path).convert("RGB")

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("RGB")
        else:
            mask = Image.new("RGB", image.size, color=(255,255,255))  # create white mask
        
        state = torch.get_rng_state()
        image = self.spatial_transform(image)
        torch.set_rng_state(state)
        mask = 1.0-self.spatial_transform(ImageOps.invert(mask))
        
        image = self.image_transform(image)

        return image, mask, label

def apply_mask(images, masks, mean, std):
    noise = torch.randn_like(images)
    for c in range(images.shape[1]):
        noise[:, c] = noise[:, c] * std[c] + mean[c]

    masked_images = masks * images + (1 - masks) * noise
    return torch.clip(masked_images, 0, 1)

if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'  
    print('using device ', device)
    
    # Choose one of the following checkpoint paths or set your own
    # Update to your checkpoint path
    model_checkpoint = 'runs/train-seqdual-withmask-wbc_50-full-XXXXXX/checkpoints/seqdual_best_XX.pt'
    # model_checkpoint = 'runs/train-seqdual-withmask-wbc_10-full-XXXXXX/checkpoints/seqdual_best_XX.pt'
    # model_checkpoint = 'runs/train-seqdual-withmask-wbc_1-full-XXXXXX/checkpoints/seqdual_best_XX.pt'
    
    plt.rcParams['font.size'] = 14
    ds = ''
    if 'wbc_100' in model_checkpoint:
        ds = 'WBC 100'
    elif 'wbc_50' in model_checkpoint:
        ds = 'WBC 50'
    elif 'wbc_10' in model_checkpoint:
        ds = 'WBC 10'
    elif 'wbc_1' in model_checkpoint:
        ds = 'WBC 1'

    pretrain_dir = os.path.join(*model_checkpoint.split('/')[:-1])
    
    model_config = SimpleNamespace()
    with open(os.path.join(pretrain_dir, 'model_config.py'), 'r') as f:
        configs = f.read()
    exec(configs, vars(model_config))
    
    img_size=model_config.img_size
    in_channel=model_config.in_channel
    num_classes=model_config.num_classes
    num_vaes=model_config.num_vaes
    vae_channels=model_config.vae_channels
    res_blocks=model_config.res_blocks
    res_channels=model_config.res_channels
    embedding_dims=model_config.embedding_dims
    codebook_size=model_config.codebook_size
    decays=model_config.decays
    
    # Load checkpoint first to get the training phase
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    training_phase = checkpoint.get('training_phase', 'full')
    
    # Initialize model with the correct training phase
    model = SequentialDualSVQVAE(
        img_size=img_size,
        in_channel=in_channel,
        num_classes=num_classes,
        num_vaes=num_vaes,
        vae_channels=vae_channels,
        res_blocks=res_blocks,
        res_channels=res_channels,
        embedding_dims=embedding_dims,
        codebook_size=codebook_size,
        decays=decays,
        training_phase=training_phase  # Important: use the saved training phase
    )
    
    # Load pretrained weights
    pretrain_weights = checkpoint['model_state_dict']
    model.load_state_dict(pretrain_weights)
    model = model.to(device)
    print(f"Model loaded from {model_checkpoint} with training_phase: {training_phase}")
    
    # Load and plot training losses
    with open(os.path.join(pretrain_dir, 'losses.json'), 'r') as f:
        losses = json.load(f)
    
    for name in losses:
        plt.clf()
        loss = []
        epochs = []
        for idx, l in enumerate(losses[name]):
            if l != 0:
                loss.append(l)
                epochs.append(idx+1)
                
        if len(loss) == 0:
            continue
            
        plt.plot(epochs, loss, '-o', label=name)
        plt.xlim(0, len(losses[name])+1)
        if 'recon' in name or 'latent' in name:
            plt.ylim(0,1)
        if 'acc' in name:
            print(ds, name, 'best with mask = ', max(loss))
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Loss')
        plt.title(f'Sequential DualSVQVAE with Mask {ds}: {name}')
        plt.xlabel('Epoch')
        
        plt.savefig(f"output/seqdual-mask-{ds.replace(' ','-')}-{name}.png") 

    # Get the dataset and test with masks
    # For this test, we'll use the same dataset as training but with masks
    dataset_type = [k for k in ['wbc_1', 'wbc_10', 'wbc_50', 'wbc_100'] if k in model_checkpoint][0]
    dataset, mean, std = get_wbc_dataset_with_masks(dataset_type)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Also get clean validation data
    val_dataset = get_wbc_dataset('val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)

    model.eval()
    y_pred_masked = []
    y_actual_masked = []
    correct_masked = 0 
    total_masked = 0 
    
    # Test with masked images from training set
    print("Testing with masked images...")
    for step, (image_batch, mask_batch, labels_batch) in tqdm(enumerate(test_loader)):
        if step >= 200:  # Limit number of test samples for speed
            break
            
        image_batch = image_batch.to(device)
        mask_batch = mask_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # Create masked images
        overlay_batch = apply_mask(image_batch, mask_batch, mean, std)
        
        with torch.no_grad():
            # Test on masked images
            preds = model.predict(overlay_batch)
            test_prediction = torch.argmax(preds, dim=1)
        
        y_pred_masked.append(test_prediction.item())
        y_actual_masked.append(labels_batch.item())
        
        correct_masked += torch.sum(test_prediction == labels_batch).item()
        total_masked += 1
            
    # Create confusion matrix for masked test
    print(f"Accuracy on masked images: {correct_masked/total_masked:.4f}")
    
    if len(y_actual_masked) > 0:
        cm = confusion_matrix(y_actual_masked, y_pred_masked)
        df_cm = pd.DataFrame(cm, index = [c for c in CLASSES],
                        columns = [c for c in CLASSES])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True, cbar=False)
        plt.title(f'Sequential DualSVQVAE Confusion Matrix for Masked Images, trained on {ds}')
        plt.ylabel('Ground Truth Labels')
        plt.xlabel('Predicted Labels')
        plt.tight_layout()
        plt.savefig(f"output/seqdual-mask-{ds.replace(' ', '-')}-confusion-matrix.png")

    # Now test on validation set (clean images)
    y_pred_clean = []
    y_actual_clean = []
    correct_clean = 0 
    total_clean = 0 
    
    print("Testing on clean validation images...")
    for step, (image_batch, labels_batch) in tqdm(enumerate(val_loader)):
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        with torch.no_grad():
            preds = model.predict(image_batch)
            test_prediction = torch.argmax(preds, dim=1)
        
        y_pred_clean.append(test_prediction.item())
        y_actual_clean.append(labels_batch.item())
        
        correct_clean += torch.sum(test_prediction == labels_batch).item()
        total_clean += 1
    
    print(f"Accuracy on clean validation images: {correct_clean/total_clean:.4f}")
    
    # Compare accuracy on masked vs clean images
    print(f"\nSummary for {ds} model:")
    print(f"Accuracy on masked images: {correct_masked/total_masked:.4f}")
    print(f"Accuracy on clean validation: {correct_clean/total_clean:.4f}")
    
    # Create comparative bar chart
    plt.figure(figsize=(8, 6))
    accuracies = [correct_clean/total_clean, correct_masked/total_masked]
    plt.bar(['Clean Validation', 'With Masks'], accuracies, color=['royalblue', 'orangered'])
    plt.title(f'Sequential DualSVQVAE Accuracy Comparison ({ds})')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"output/seqdual-{ds.replace(' ', '-')}-accuracy-comparison.png")