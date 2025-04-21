import torch
from models.dual_svqvae import DualSVQVAE
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
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'  
    print('using device ', device)
    
    model_checkpoint = 'checkpoints/pretrain-dualsvqvae-recon_all-b24-e100-s10/checkpoints/model_30.pt'
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    pretrain_weights = checkpoint['model_state_dict']
    model.load_state_dict(pretrain_weights)

    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

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
        plt.plot(epochs, loss, '-o', label=name)
        plt.xlim(0, len(losses[name])+1)
        plt.title(f'Pretrain DualSVQVAE: {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"output/dual-pretrain-{name}.png") 

    data = 'wbc'
    label = 'Neurophil'
    if data == 'wbc':
        mean= [0.7048, 0.5392, 0.5885]
        std= [0.1626, 0.1902, 0.0974]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomResizedCrop((512,512), scale=(1.0, 1.0), antialias=True),
            transforms.Resize((512,512), antialias=True)
        ])
        pretrain_dataset = PretrainingDataset(img_dir=f'data/WBC_100/data/{label}', transform=transform)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)
    elif data == 'cam':
        mean= [0.6931, 0.5478, 0.6757]
        std= [0.1972, 0.2487, 0.1969]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomResizedCrop((512,512), scale=(1.0, 1.0), antialias=True),
            transforms.Resize((512,512), antialias=True)
        ])
        pretrain_dataset = PretrainingDataset(img_dir='data/CAM16_100cls_10mask/train/data/normal', transform=transform)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)
    elif data == 'prcc':  
        mean= [0.6618, 0.5137, 0.6184]
        std= [0.1878, 0.2276, 0.1912]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomResizedCrop((512,512), scale=(1.0, 1.0), antialias=True),
            transforms.Resize((512,512), antialias=True)
        ])
        pretrain_dataset = PretrainingDataset(img_dir='data/pRCC_nolabel', transform=transform)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)

    model.eval()
    for step, image_batch in tqdm(enumerate(pretrain_loader)):

        image_batch = image_batch.to(device)
        
        # SVQVAE1 encodings (levels 0, 1, 2)
        qt0, qb0, qj0, diff0, idt0, idb0 = model.encode(image_batch, 0, 0)
        qt1, qb1, qj1, diff1, idt1, idb1 = model.encode(image_batch, 0, 1)
        qt2, qb2, qj2, diff2, idt2, idb2 = model.encode(image_batch, 0, 2)

        # SVQVAE2 encodings (levels 3, 4)
        qt3, qb3, qj3, diff3, idt3, idb3 = model.encode(image_batch, 3, 3)
        qt4, qb4, qj4, diff4, idt4, idb4 = model.encode(image_batch, 3, 4)
        
        # Reconstructions from different levels
        rimg4 = model.decode(qj4, 4, -1)  # SVQVAE2 level 2 (level 4 overall)
        rimg3 = model.decode(qj3, 3, -1)  # SVQVAE2 level 1 (level 3 overall)
        rimg2 = model.decode(qj2, 2, -1)  # SVQVAE1 level 3
        rimg1 = model.decode(qj1, 1, -1)  # SVQVAE1 level 2
        rimg0 = model.decode(qj0, 0, -1)  # SVQVAE1 level 1

        # Extract images for display
        img = image_batch.detach().cpu().squeeze()
        rimg4 = rimg4.detach().cpu().squeeze()
        rimg3 = rimg3.detach().cpu().squeeze()
        rimg2 = rimg2.detach().cpu().squeeze()
        rimg1 = rimg1.detach().cpu().squeeze()
        rimg0 = rimg0.detach().cpu().squeeze()
        
        # Plot results
        plt.figure(figsize=(15, 10))
        plt.title(f"DualSVQVAE reconstructions of {label}")
        
        plt.subplot(231)
        plt.title('Original')
        plt.imshow(denormalize_img(img, mean, std))
        
        plt.subplot(232)
        plt.title('Recon from SVQVAE1 level 0')
        plt.imshow(denormalize_img(rimg0, mean, std))
        
        plt.subplot(233)
        plt.title('Recon from SVQVAE1 level 1')
        plt.imshow(denormalize_img(rimg1, mean, std))
        
        plt.subplot(234)
        plt.title('Recon from SVQVAE1 level 2')
        plt.imshow(denormalize_img(rimg2, mean, std))
        
        plt.subplot(235)
        plt.title('Recon from SVQVAE2 level 0')
        plt.imshow(denormalize_img(rimg3, mean, std))
        
        plt.subplot(236)
        plt.title('Recon from SVQVAE2 level 1')
        plt.imshow(denormalize_img(rimg4, mean, std))
        
        plt.tight_layout()
        plt.savefig(f'output/dual-pretrain-{data}-{label}-{step}.png')
        
        if step > 6:
            break