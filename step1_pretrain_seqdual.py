import torch
from torch.nn import functional as F
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import os, shutil, datetime
import logging
from data.stats import data
import numpy as np
import argparse
from models.seq_dual_svqvae import SequentialDualSVQVAE
import json

def get_pretrain_dataset(name):
    info = data['pretrain'][name]
    mean = info['mean']
    std = info['std']
    
    if name=='cam16':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    elif name=='prcc':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.RandomCrop((512,512))
        ])
    
    datasets = []
    for folder in info['paths']:
        dataset = PretrainingDataset(img_dir=folder, transform=transform)
        datasets.append(dataset)
        
    dataset = ConcatDataset(datasets)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sequential DualSVQVAE training config")

    parser.add_argument("--batch_size", type=int, help="batch_size", required=True)
    parser.add_argument("--svqvae1_epochs", type=int, required=True, help="Epochs to train SVQVAE1")
    parser.add_argument("--svqvae2_epochs", type=int, required=True, help="Epochs to train SVQVAE2")
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--level_stagger_epochs", type=int, required=True)
    parser.add_argument("--method", type=str, choices=["recon_level", "recon_all"], required=True)
    parser.add_argument("--description", type=str, required=True)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    svqvae1_epochs = args.svqvae1_epochs
    svqvae2_epochs = args.svqvae2_epochs
    save_every = args.save_every
    level_stagger_epochs = args.level_stagger_epochs
    method = args.method
    
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    directory_name = f"runs/pretrain-seqdual-{method}-b{batch_size}-e{svqvae1_epochs}-{svqvae2_epochs}-s{level_stagger_epochs}-" + current_time
    os.makedirs(directory_name, exist_ok=True)
    checkpoints_directory = os.path.join(directory_name, "checkpoints")
    os.makedirs(checkpoints_directory, exist_ok=True)
    current_script_name = os.path.basename(__file__)
    shutil.copy2(current_script_name, directory_name)
    model_file1 = 'models/seq_dual_svqvae.py'
    model_file2 = 'models/vqvae.py'
    model_file3 = 'models/svqvae.py'
    shutil.copy2(model_file1, directory_name)
    shutil.copy2(model_file2, directory_name)
    shutil.copy2(model_file3, directory_name)
    log_file = os.path.join(directory_name, f"run_{current_time}.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])

    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        logging.info(f'using device {device_name} {device}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f'using device{device}')
    else:
        device = 'cpu'  
        logging.info(f'using device {device}')
    
    # Model configuration
    img_size=512
    in_channel=3
    num_classes=5
    num_vaes=5  # DualSVQVAE uses 5 levels (SVQVAE1: 3, SVQVAE2: 2)
    vae_channels=[128, 128, 128, 128, 128]
    res_blocks=[4, 4, 4, 4, 4]
    res_channels=[64, 64, 64, 64, 64]
    embedding_dims=[3, 3, 3, 3, 3]
    codebook_size=[512, 512, 512, 512, 512]
    decays=[0.99, 0.99, 0.99, 0.99, 0.99]
    
    with open(f'{checkpoints_directory}/model_config.py', 'w') as f:
        f.write(f'img_size={img_size}\n')
        f.write(f'in_channel={in_channel}\n')
        f.write(f'num_classes={num_classes}\n')
        f.write(f'num_vaes={num_vaes}\n')
        f.write(f'vae_channels={vae_channels}\n')
        f.write(f'res_blocks={res_blocks}\n')
        f.write(f'res_channels={res_channels}\n')
        f.write(f'embedding_dims={embedding_dims}\n')
        f.write(f'codebook_size={codebook_size}\n')
        f.write(f'decays={decays}\n')
    
    # Initialize datasets
    cam16_ds = get_pretrain_dataset('cam16')
    prcc_ds = get_pretrain_dataset('prcc')
    
    merged_ds = ConcatDataset([cam16_ds, prcc_ds])
    weights = [1.0]*len(cam16_ds) + [4.0]*len(prcc_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(merged_ds), replacement=True)
    
    pretrain_loader = DataLoader(merged_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    
    losses = {}
    for l in range(3):  # SVQVAE1 levels
        losses[f'epoch_recon_loss_vae{l}'] = []
        losses[f'epoch_latent_loss_vae{l}'] = []
    for l in range(3, 5):  # SVQVAE2 levels
        losses[f'epoch_recon_loss_vae{l}'] = []
        losses[f'epoch_latent_loss_vae{l}'] = []
    
    latent_loss_weight = 0.25
    mse_loss = torch.nn.MSELoss()
    
    ###################################
    # Phase 1: Train SVQVAE1
    ###################################
    logging.info(args.description)
    logging.info("Phase 1: Training SVQVAE1")
    logging.info(f'method: {method}')
    logging.info(f'input: {in_channel} x {img_size} x {img_size}')
    logging.info(f'num classes: {num_classes}')
    logging.info(f'# vaes: {num_vaes} (SVQVAE1: 3, SVQVAE2: 2)')
    logging.info(f'level staggering epochs: {level_stagger_epochs}')
    
    # Initialize model in SVQVAE1 training phase
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
        training_phase='svqvae1'
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=svqvae1_epochs)
    
    for e in range(1, svqvae1_epochs+1):
        for k in losses:
            if k.startswith('epoch_recon_loss_vae') or k.startswith('epoch_latent_loss_vae'):
                if int(k[-1]) < 3:  # SVQVAE1 levels only
                    losses[k].append(0.0)
                else:
                    losses[k].append(0)
        
        if level_stagger_epochs == 0:
            training_level = 2  # Train all levels of SVQVAE1
        else:
            training_level = min((e-1)//level_stagger_epochs, 2)
        
        logging.info(f"{f'starting epoch {e}, training SVQVAE1 to level {training_level}':-^{100}}")
        epoch_loss = 0
        
        for step, image_batch in enumerate(pretrain_loader):
            image_batch = image_batch.to(device)
            
            optimizer.zero_grad()
            
            for level in range(training_level+1):
                
                if method == 'recon_level':
                    if level == 0:
                        input = image_batch
                    else:
                        _, _, input, _, _, _ = model.encode(image_batch, 0, level-1)
                    
                    qt, qb, qj, diff, idt, idb = model.encode(input, level, level)
                    recon = model.decode(qj, level, level-1)
                    
                elif method == 'recon_all':
                    input = image_batch
                    qt, qb, qj, diff, idt, idb = model.encode(input, 0, level)
                    recon = model.decode(qj, level, -1)
                
                latent_loss = diff.mean()
                recon_loss = mse_loss(input, recon)
                
                total_loss = latent_loss_weight*latent_loss + recon_loss
                total_loss.backward()
                
                losses[f'epoch_recon_loss_vae{level}'][-1] += recon_loss.item()
                losses[f'epoch_latent_loss_vae{level}'][-1] += latent_loss.item()
            
            optimizer.step()
            
            if (step+1)%10 == 0:
                logging.info(f"{f'step {step+1}':-^{50}}")
                for k, v in losses.items():
                    if v[-1] != 0:
                        logging.info(f'avg {k}: {v[-1]/(step+1)}')
        
        scheduler.step()
        
        for k in losses:
            if losses[k][-1] != 0:
                losses[k][-1] /= (step+1)
        
        if e%save_every == 0 or e == 1 or e == svqvae1_epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': e,
                'phase': 'svqvae1',
            }, f'{checkpoints_directory}/model_phase1_{e}.pt')
            
            with open(f'{checkpoints_directory}/losses.json', 'w') as f:
                json.dump(losses, f)
                
            logging.info(f'saving checkpoint and losses after epoch {e}')
    
    # Save final SVQVAE1 model
    svqvae1_final_path = f'{checkpoints_directory}/model_svqvae1_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': svqvae1_epochs,
        'phase': 'svqvae1',
    }, svqvae1_final_path)
    logging.info(f'SVQVAE1 training complete. Final model saved to {svqvae1_final_path}')
    
    ###################################
    # Phase 2: Train SVQVAE2
    ###################################
    logging.info("Phase 2: Training SVQVAE2")
    
    # Set model to SVQVAE2 training phase
    model.set_training_phase('svqvae2')
    
    # New optimizer targeting only SVQVAE2 parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=svqvae2_epochs)
    
    for e in range(1, svqvae2_epochs+1):
        for k in losses:
            if k.startswith('epoch_recon_loss_vae') or k.startswith('epoch_latent_loss_vae'):
                if int(k[-1]) >= 3:  # SVQVAE2 levels only
                    losses[k].append(0.0)
                else:
                    losses[k].append(0)
        
        if level_stagger_epochs == 0:
            training_level = 1  # Train all levels of SVQVAE2
        else:
            training_level = min((e-1)//level_stagger_epochs, 1)
        
        vae2_levels = [3, 4]  # Index 3 and 4 for the two SVQVAE2 levels
        logging.info(f"{f'starting epoch {e}, training SVQVAE2 to level {training_level} (vae {vae2_levels[training_level]})':-^{100}}")
        epoch_loss = 0
        
        for step, image_batch in enumerate(pretrain_loader):
            image_batch = image_batch.to(device)
            
            optimizer.zero_grad()
            
            for idx in range(training_level+1):
                level = vae2_levels[idx]
                
                if method == 'recon_level':
                    if level == 3:  # First level of SVQVAE2
                        # Get encodings from SVQVAE1
                        level1_encoding = model._get_svqvae1_level1_encoding(image_batch)
                        _, _, level3_encoding, _, _, _ = model.svqvae1.encode(image_batch, 0, 2)
                        
                        # Prepare input for SVQVAE2
                        level3_upsampled = F.interpolate(
                            level3_encoding, 
                            size=level1_encoding.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        input = torch.cat([level3_upsampled, level1_encoding], dim=1)
                        
                        # Process through SVQVAE2 level 0
                        qt, qb, qj, diff, idt, idb = model.svqvae2[0].encode(input)
                        recon = model.svqvae2[0].decode(qj)
                        
                    else:  # Second level of SVQVAE2
                        # Get output from first level of SVQVAE2
                        level1_encoding = model._get_svqvae1_level1_encoding(image_batch)
                        _, _, level3_encoding, _, _, _ = model.svqvae1.encode(image_batch, 0, 2)
                        level3_upsampled = F.interpolate(
                            level3_encoding, 
                            size=level1_encoding.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
                        _, _, input, _, _, _ = model.svqvae2[0].encode(svqvae2_input)
                        
                        # Process through SVQVAE2 level 1
                        qt, qb, qj, diff, idt, idb = model.svqvae2[1].encode(input)
                        recon = model.svqvae2[1].decode(qj)
                        
                elif method == 'recon_all':
                    if level == 3:  # First level of SVQVAE2
                        # Get encodings from SVQVAE1
                        level1_encoding = model._get_svqvae1_level1_encoding(image_batch)
                        _, _, level3_encoding, _, _, _ = model.svqvae1.encode(image_batch, 0, 2)
                        
                        # Prepare input for SVQVAE2
                        level3_upsampled = F.interpolate(
                            level3_encoding, 
                            size=level1_encoding.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
                        
                        # Process and reconstruct original input
                        qt, qb, qj, diff, idt, idb = model.svqvae2[0].encode(svqvae2_input)
                        recon = model.svqvae2[0].decode(qj)
                        input = svqvae2_input
                        
                    else:  # Second level of SVQVAE2
                        # Get full path encoding
                        level1_encoding = model._get_svqvae1_level1_encoding(image_batch)
                        _, _, level3_encoding, _, _, _ = model.svqvae1.encode(image_batch, 0, 2)
                        level3_upsampled = F.interpolate(
                            level3_encoding, 
                            size=level1_encoding.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
                        _, _, level0_output, _, _, _ = model.svqvae2[0].encode(svqvae2_input)
                        
                        # Process through level 1
                        qt, qb, qj, diff, idt, idb = model.svqvae2[1].encode(level0_output)
                        
                        # For recon_all, we try to reconstruct the original SVQVAE2 input
                        recon = model.svqvae2[0].decode(model.svqvae2[1].decode(qj))
                        input = svqvae2_input
                
                latent_loss = diff.mean()
                recon_loss = mse_loss(input, recon)
                
                total_loss = latent_loss_weight*latent_loss + recon_loss
                total_loss.backward()
                
                losses[f'epoch_recon_loss_vae{level}'][-1] += recon_loss.item()
                losses[f'epoch_latent_loss_vae{level}'][-1] += latent_loss.item()
            
            optimizer.step()
            
            if (step+1)%10 == 0:
                logging.info(f"{f'step {step+1}':-^{50}}")
                for k, v in losses.items():
                    if v[-1] != 0:
                        logging.info(f'avg {k}: {v[-1]/(step+1)}')
        
        scheduler.step()
        
        for k in losses:
            if losses[k][-1] != 0:
                losses[k][-1] /= (step+1)
        
        if e%save_every == 0 or e == 1 or e == svqvae2_epochs:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': e,
                'phase': 'svqvae2',
            }, f'{checkpoints_directory}/model_phase2_{e}.pt')
            
            with open(f'{checkpoints_directory}/losses.json', 'w') as f:
                json.dump(losses, f)
                
            logging.info(f'saving checkpoint and losses after epoch {e}')
    
    # Save final complete model
    final_path = f'{checkpoints_directory}/model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': svqvae2_epochs,
        'phase': 'complete',
    }, final_path)
    logging.info(f'Sequential DualSVQVAE training complete. Final model saved to {final_path}')