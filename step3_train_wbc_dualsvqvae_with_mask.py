import torch
from models.dual_svqvae import DualSVQVAE
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import os, shutil, datetime
import logging
from data.stats import data
import numpy as np
import argparse
import json
import time
from types import SimpleNamespace
from PIL import Image
from torch.utils.data import Dataset, DataLoader


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
    dataset = MaskDataset(data_root=data_path, mask_root=mask_path, spatial_transform=spatial_transform, image_transform=image_transform)
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
            transforms.RandomRotation(180, fill=mean),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    
    path = data['train']['paths'][type]

    dataset = ImageFolder(root=path, transform=transform)
    return dataset

def apply_mask(images, masks, mean, std):

    noise = torch.randn_like(images)
    for c in range(images.shape[1]):
        noise[:, c] = noise[:, c] * std[c] + mean[c]

    masked_images = masks * images + (1 - masks) * noise

    return torch.clip(masked_images, 0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training config")

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--alternating_epochs", type=int, required=True)
    args = parser.parse_args()
    
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    save_every = args.save_every
    alternating_epochs = args.alternating_epochs
    model_checkpoint = args.checkpoint
    dataset = args.dataset
    pretrain_dir = os.path.join(*model_checkpoint.split('/')[:-1])
    
    model_config = SimpleNamespace()
    with open(os.path.join(pretrain_dir, 'model_config.py'), 'r') as f:
        configs = f.read()
    exec(configs, vars(model_config))
    
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    directory_name = f"runs/train-dualsvqvae-withmask-{dataset}-" + current_time
    os.makedirs(directory_name, exist_ok=True)
    checkpoints_directory = os.path.join(directory_name, "checkpoints")
    os.makedirs(checkpoints_directory, exist_ok=True)
    current_script_name = os.path.basename(__file__)
    shutil.copy2(current_script_name, directory_name)

    log_file = os.path.join(directory_name, f"run_{current_time}.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])
    logging.info(f"{args}")
    
    num_gpus = 1
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        num_gpus = torch.cuda.device_count()
        logging.info(f"{num_gpus} gpus available")
        logging.info(f'using device {device_name} {device}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f'using device {device}')
    else:
        device = 'cpu'  
        logging.info(f'using device {device}')
    
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
    
    model = DualSVQVAE(
        img_size=img_size,
        in_channel=in_channel,
        num_classes=num_classes,
        num_vaes=num_vaes,
        vae_channels=vae_channels,
        res_blocks=res_blocks,
        res_channels=res_channels,
        embedding_dims=embedding_dims,
        codebook_size=codebook_size,
        decays=decays
    )
    
    logging.info(f'checkpoint: {model_checkpoint}')
    logging.info(f'dataset: {dataset}')
    logging.info(f'input: {in_channel} x {img_size} x {img_size}')
    logging.info(f'num classes: {num_classes}')
    logging.info(f'# vaes: {num_vaes} (SVQVAE1: 3, SVQVAE2: 2)')
    logging.info(f'vae channels: {vae_channels}')
    logging.info(f'vae res blocks: {res_blocks}')
    logging.info(f'vae res channels: {res_channels}')
    logging.info(f'vae embedding dims: {embedding_dims}')
    logging.info(f'codebook sizes: {codebook_size}')
    logging.info(f'decays: {decays}')
    
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    pretrain_weights = checkpoint['model_state_dict']
    model.load_state_dict(pretrain_weights)
    model = model.to(device)
    
    logging.info(f"current memory allocation: {torch.cuda.memory_allocated(device) / (1024 ** 3)}")
    logging.info(f'starting training from checkpoint {model_checkpoint}')
    
    logging.info(f'total number of parameters: { sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logging.info(f'training with batch size {batch_size} for {num_epochs} epochs')
    logging.info(f'saving checkpoint every {save_every} epochs')
    
    training_dataset, mean, std = get_wbc_dataset_with_masks(dataset)
    
    # equal sampling from all classes
    targets_tensor = torch.tensor(training_dataset.targets)
    class_sample_count = torch.tensor([(targets_tensor == t).sum() for t in torch.unique(targets_tensor, sorted=True)])
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in training_dataset.targets])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    testing_dataset = get_wbc_dataset('val')
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    # for finetuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    
    losses = {}
    losses[f'epoch_recon_loss'] = []
    losses[f'epoch_latent_loss'] = []
    losses[f'epoch_cls_loss'] = []
    losses[f'epoch_feature_loss'] = []
    losses[f'epoch_train_acc'] = []
    losses[f'epoch_test_acc'] = []

    show_every_dict = {
        'wbc_1': 1,
        'wbc_10': 10,
        'wbc_50': 50,
        'wbc_100': 50
    }
    
    show_every = show_every_dict[dataset]
    best_test_acc = 0
    latent_loss_weight = 0.25
    for e in range(1, num_epochs+1):
        logging.info( f"{f'starting epoch {e}':-^{50}}" )
        
        model.train()
        
        for k in losses:
            losses[k].append(0.0)
        
        mse_loss = torch.nn.MSELoss()
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        train_total = 0 
        for step, (image_batch, mask_batch, labels_batch) in enumerate(train_loader):
            overlay_batch = apply_mask(image_batch, mask_batch, mean, std)
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            overlay_batch = overlay_batch.to(device)
            
            optimizer.zero_grad()
            
            if e%alternating_epochs == 0:
                # Get encodings from both SVQVAE1 and SVQVAE2
                qt1, qb1, qj1, diff1, _, _ = model.encode(image_batch, 0, 2)  # Last level of SVQVAE1
                qt2, qb2, qj2, diff2, _, _ = model.encode(image_batch, 3, 4)  # Last level of SVQVAE2
                
                # Decode from both encodings
                recon1 = model.decode(qj1, 2, -1)  # Reconstruct from SVQVAE1
                recon2 = model.decode(qj2, 4, -1)  # Reconstruct from SVQVAE2
                
                # Combine losses from both stacks
                latent_loss = (diff1 + diff2).mean()
                recon_loss = 0.5 * (mse_loss(image_batch, recon1) + mse_loss(image_batch, recon2))
                total_loss = latent_loss_weight * latent_loss + recon_loss
                total_loss.backward()
            else:
                recon_loss = torch.tensor(0)
                latent_loss = torch.tensor(0)
                
            # Feature consistency between original and masked images
            # Get encodings from both SVQVAE1 and SVQVAE2 for original image
            _, _, qj1, _, _, _ = model.encode(image_batch, 0, 2)  # Last level of SVQVAE1
            _, _, qj2, _, _, _ = model.encode(image_batch, 3, 4)  # Last level of SVQVAE2
            
            # Get encodings for overlay image
            with torch.no_grad():
                _, _, qj1_overlay, _, _, _ = model.encode(overlay_batch, 0, 2)
                _, _, qj2_overlay, _, _, _ = model.encode(overlay_batch, 3, 4)
            
            # Feature consistency loss
            feature_loss = 0.5 * (mse_loss(qj1, qj1_overlay) + mse_loss(qj2, qj2_overlay))
            feature_loss.backward()
                
            # Classification loss
            preds = model.predict(image_batch)
            prediction_loss = cross_entropy_loss(preds, labels_batch)
            prediction_loss.backward()
     
            optimizer.step()
            
            losses['epoch_recon_loss'][-1] += recon_loss.item()
            losses['epoch_latent_loss'][-1] += latent_loss.item()
            losses['epoch_cls_loss'][-1] += prediction_loss.item()
            losses['epoch_feature_loss'][-1] += feature_loss.item()
            
            train_predictions = torch.argmax(preds, dim=1)
            losses['epoch_train_acc'][-1] += torch.sum(train_predictions == labels_batch).item()
            train_total += labels_batch.shape[0]
                
            if (step+1)%show_every == 0:
                logging.info(f"avg recon loss after step {step+1}: {losses['epoch_recon_loss'][-1]/(step+1)}")
                logging.info(f"avg latent loss after step {step+1}: {losses['epoch_latent_loss'][-1]/(step+1)}")
                logging.info(f"avg cls loss after step {step+1}: {losses['epoch_cls_loss'][-1]/(step+1)}")
                logging.info(f"avg feature loss after step {step+1}: {losses['epoch_feature_loss'][-1]/(step+1)}")
                logging.info(f"training acc after step {step+1}: {losses['epoch_train_acc'][-1]/(train_total)}")    

        losses['epoch_recon_loss'][-1] /= (step+1)
        losses['epoch_latent_loss'][-1] /= (step+1)
        losses['epoch_cls_loss'][-1] /= (step+1)
        losses['epoch_feature_loss'][-1] /= (step+1)
        losses['epoch_train_acc'][-1] /= train_total
        
        logging.info(f"epoch {e} training acc: {losses['epoch_train_acc'][-1]}")   
        
        ### testing metrics
        model.eval()
       
        test_total = 0 
        for step, (image_batch, labels_batch) in enumerate(test_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            preds = model.predict(image_batch)
            test_predictions = torch.argmax(preds, dim=1)
            
            losses['epoch_test_acc'][-1] += torch.sum(test_predictions == labels_batch).item()
            test_total += labels_batch.shape[0]
  
        losses['epoch_test_acc'][-1] /= test_total
        logging.info(f"epoch {e} testing acc: {losses['epoch_test_acc'][-1]}")
        
        if losses['epoch_test_acc'][-1] > best_test_acc:
            best_test_acc = losses['epoch_test_acc'][-1]
            
            if best_test_acc > 0.85:
                torch.save({
                    'model_state_dict':  model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': e
                }, f'{checkpoints_directory}/dualsvqvae_best_{e}.pt')
                logging.info(f'saving checkpoint and losses after episode {e}, best test acc {best_test_acc}')
                
        ### saving checkpoints
        if e%save_every == 0 or e == 1:
            torch.save({
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': e
            }, f'{checkpoints_directory}/dualsvqvae_model_{e}.pt')
            
            with open(f'{checkpoints_directory}/losses.json', 'w') as f:
                json.dump(losses, f)
                
            logging.info(f'saving checkpoint and losses after episode {e}')