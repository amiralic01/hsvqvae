import torch
from models.seq_dual_svqvae import SequentialDualSVQVAE
from torch.nn import functional as F
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from types import SimpleNamespace

def denormalize_img(img, mean, std):
    for t,m,s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    img = torch.clamp(img, 0,1)
    return img.permute((1,2,0))

if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'  
    print('using device ', device)
    
    # !!!Change this to your checkpoint path!!!
    # ex. 'runs/pretrain-seqdual-recon_all-b24-e100-100-s10-0422_145623/checkpoints/model_final.pt'
    # model_checkpoint = 'runs/pretrain-seqdual-recon_all-b1-e1-1-s1-0423_001958/checkpoints/model_final.pt'
    model_checkpoint = 'checkpoints/test-seq/model_final.pt'
    # model_checkpoint = 'runs/pretrain-seqdual-recon_all-b24-e100-100-s10-XXXXXX/checkpoints/model_final.pt'
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
    
    # Load model with training phase set to full for visualization
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
        training_phase='full'  # Use full for visualization
    )
    
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    # Load the weights
    pretrain_weights = checkpoint['model_state_dict']
    model.load_state_dict(pretrain_weights)
    model = model.to(device)
    
    # Print checkpoint info
    print(f"Loaded checkpoint from {model_checkpoint}")
    print(f"Training phase: {checkpoint.get('phase', 'unknown')}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load and plot losses
    with open(os.path.join(pretrain_dir, 'losses.json'), 'r') as f:
        losses = json.load(f)
    
    # Separate SVQVAE1 and SVQVAE2 losses for clearer visualization
    svqvae1_losses = {}
    svqvae2_losses = {}
    
    for name in losses:
        if name.endswith(('0', '1', '2')):  # SVQVAE1 losses
            svqvae1_losses[name] = losses[name]
        elif name.endswith(('3', '4')):  # SVQVAE2 losses
            svqvae2_losses[name] = losses[name]
    
    # Plot SVQVAE1 losses
    plt.figure(figsize=(10, 6))
    for name in svqvae1_losses:
        loss = []
        epochs = []
        for idx, l in enumerate(svqvae1_losses[name]):
            if l != 0:
                loss.append(l)
                epochs.append(idx+1)
        plt.plot(epochs, loss, '-o', label=name)
    
    plt.title('Sequential DualSVQVAE: SVQVAE1 Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"output/seqdual-pretrain-svqvae1-losses.png")
    
    # Plot SVQVAE2 losses
    plt.figure(figsize=(10, 6))
    for name in svqvae2_losses:
        loss = []
        epochs = []
        for idx, l in enumerate(svqvae2_losses[name]):
            if l != 0:
                loss.append(l)
                epochs.append(idx+1)
        plt.plot(epochs, loss, '-o', label=name)
    
    plt.title('Sequential DualSVQVAE: SVQVAE2 Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"output/seqdual-pretrain-svqvae2-losses.png")

    # Define test datasets to visualize reconstructions
    datasets = {
        'wbc': {
            'path': 'data/WBC_100/data/Neutrophil',
            'mean': [0.7048, 0.5392, 0.5885],
            'std': [0.1626, 0.1902, 0.0974],
            'label': 'Neutrophil'
        },
        'cam': {
            'path': 'data/CAM16_100cls_10mask/train/data/normal',
            'mean': [0.6931, 0.5478, 0.6757],
            'std': [0.1972, 0.2487, 0.1969],
            'label': 'Normal'
        },
        'prcc': {
            'path': 'data/pRCC_nolabel',
            'mean': [0.6618, 0.5137, 0.6184],
            'std': [0.1878, 0.2276, 0.1912],
            'label': 'pRCC'
        }
    }
    
    # Run model evaluation on each dataset
    model.eval()
    
    for data_name, data_info in datasets.items():
        # Extract dataset info
        path = data_info['path']
        mean = data_info['mean']
        std = data_info['std']
        label = data_info['label']
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomResizedCrop((512,512), scale=(1.0, 1.0), antialias=True),
            transforms.Resize((512,512), antialias=True)
        ])
        
        pretrain_dataset = PretrainingDataset(img_dir=path, transform=transform)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)
        
        print(f"Processing {data_name} dataset...")

        for step, image_batch in tqdm(enumerate(pretrain_loader)):
            if step > 5:  # Limit number of images processed
                break
                
            image_batch = image_batch.to(device)
            
            with torch.no_grad():  # No gradients needed for visualization
                # SVQVAE1 encodings (levels 0, 1, 2)
                qt0, qb0, qj0, diff0, idt0, idb0 = model.encode(image_batch, 0, 0)
                qt1, qb1, qj1, diff1, idt1, idb1 = model.encode(image_batch, 0, 1)
                qt2, qb2, qj2, diff2, idt2, idb2 = model.encode(image_batch, 0, 2)

                # SVQVAE2 encodings (levels 3, 4)
                qt3, qb3, qj3, diff3, idt3, idb3 = model.encode(image_batch, 3, 3)
                qt4, qb4, qj4, diff4, idt4, idb4 = model.encode(image_batch, 3, 4)
                
                # Reconstructions from different levels
                rimg2 = model.decode(qj2, 2, -1)  # SVQVAE1 level 2
                rimg1 = model.decode(qj1, 1, -1)  # SVQVAE1 level 1
                rimg0 = model.decode(qj0, 0, -1)  # SVQVAE1 level 0
                
                # For SVQVAE2, we can only reconstruct to SVQVAE2 input level
                rimg3 = model.decode(qj3, 3, 3)  # SVQVAE2 level 0
                rimg4 = model.decode(qj4, 4, 3)  # SVQVAE2 level 1
                
                # Create inputs for SVQVAE2 to better evaluate its reconstruction
                level1_encoding = model._get_svqvae1_level1_encoding(image_batch)
                _, _, level3_encoding, _, _, _ = model.svqvae1.encode(image_batch, 0, 2)
                level3_upsampled = F.interpolate(
                    level3_encoding, 
                    size=level1_encoding.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)

            # Extract images for display
            img = image_batch.detach().cpu().squeeze()
            rimg0 = rimg0.detach().cpu().squeeze()
            rimg1 = rimg1.detach().cpu().squeeze()
            rimg2 = rimg2.detach().cpu().squeeze()
            rimg3 = rimg3.detach().cpu().squeeze()
            rimg4 = rimg4.detach().cpu().squeeze()
            
            # Plot results with two figures to show the sequential training better
            
            # Figure 1: SVQVAE1 reconstructions (trained first)
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Sequential DualSVQVAE - SVQVAE1 reconstructions of {label}", fontsize=16)
            
            plt.subplot(141)
            plt.title('Original')
            plt.imshow(denormalize_img(img, mean, std))
            
            plt.subplot(142)
            plt.title('Recon from SVQVAE1 level 0')
            plt.imshow(denormalize_img(rimg0, mean, std))

            plt.subplot(143)
            plt.title('Recon from SVQVAE1 level 1')
            plt.imshow(denormalize_img(rimg1, mean, std))
            
            plt.subplot(144)
            plt.title('Recon from SVQVAE1 level 2')
            plt.imshow(denormalize_img(rimg2, mean, std))
            
            plt.tight_layout()
            plt.savefig(f'output/seqdual-pretrain-{data_name}-{step}-svqvae1.png')
            
            # Figure 2: SVQVAE2 reconstructions (trained second)
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Sequential DualSVQVAE - SVQVAE2 reconstructions of {label}", fontsize=16)
            
            # Extract SVQVAE2 input for display
            svqvae2_input_img = svqvae2_input.detach().cpu().squeeze()
            
            plt.subplot(131)
            plt.title('SVQVAE2 Input (Concatenated)')
            # This is a multi-channel feature map, not an image, so just display first 3 channels if available
            if svqvae2_input_img.shape[0] >= 3:
                display_img = svqvae2_input_img[:3].permute(1, 2, 0)
                # Normalize for visualization
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-6)
                plt.imshow(display_img)
            else:
                plt.imshow(svqvae2_input_img[0], cmap='viridis')
                plt.colorbar()
            
            plt.subplot(132)
            plt.title('Recon from SVQVAE2 level 0')
            # Same for reconstructed SVQVAE2 outputs
            if rimg3.shape[1] >= 3:
                display_img = rimg3.detach().cpu().squeeze()[:3].permute(1, 2, 0)
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-6)
                plt.imshow(display_img)
            else:
                plt.imshow(rimg3.detach().cpu().squeeze()[0], cmap='viridis')
                plt.colorbar()
            
            plt.subplot(133)
            plt.title('Recon from SVQVAE2 level 1')
            if rimg4.shape[1] >= 3:
                display_img = rimg4.detach().cpu().squeeze()[:3].permute(1, 2, 0)
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-6)
                plt.imshow(display_img)
            else:
                plt.imshow(rimg4.detach().cpu().squeeze()[0], cmap='viridis')
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f'output/seqdual-pretrain-{data_name}-{step}-svqvae2.png')
            
            # Close figures to avoid memory issues
            plt.close('all')