import torch
from torch import nn
from torch.nn import functional as F
from models.svqvae import SVQVAE
from models.vqvae import VQVAE

class SequentialDualSVQVAE(nn.Module):
    def __init__(
        self,
        img_size,
        in_channel=3,
        num_classes=5,
        num_vaes=3,  # Total number of VAEs (SVQVAE1: 3, SVQVAE2: 2)
        vae_channels=[128, 128, 128, 128, 128],  # Combined channels
        res_blocks=[2, 2, 2, 2, 2],  # Combined res_blocks
        res_channels=[32, 32, 32, 32, 32],  # Combined res_channels
        embedding_dims=[1, 1, 1, 1, 1],  # Combined embedding_dims
        codebook_size=[512, 512, 512, 512, 512],  # Combined codebook_size
        decays=[0.99, 0.99, 0.99, 0.99, 0.99],  # Combined decays
        training_phase='full'  # Options: 'svqvae1', 'svqvae2', 'classifier', 'full'
    ):
        super().__init__()
        
        # Validation to ensure we have enough parameters for both SVQVAEs
        assert num_vaes >= 3, "num_vaes must be at least 3 for SequentialDualSVQVAE (SVQVAE1: 3, SVQVAE2: 2)"
        assert len(vae_channels) >= 5, "Need at least 5 elements in parameter arrays"
        assert img_size/(4**3) > 1, "Image too small for compression"
        
        # Define the number of levels for each SVQVAE
        self.svqvae1_levels = 3
        self.svqvae2_levels = 2
        self.num_vaes = self.svqvae1_levels + self.svqvae2_levels
        self.training_phase = training_phase
        
        # Split parameters for SVQVAE1 and SVQVAE2
        vae1_channels = vae_channels[:3]
        vae1_res_blocks = res_blocks[:3]
        vae1_res_channels = res_channels[:3]
        vae1_embedding_dims = embedding_dims[:3]
        vae1_codebook_size = codebook_size[:3]
        vae1_decays = decays[:3]
        
        vae2_channels = vae_channels[3:5]
        vae2_res_blocks = res_blocks[3:5]
        vae2_res_channels = res_channels[3:5]
        vae2_embedding_dims = embedding_dims[3:5]
        vae2_codebook_size = codebook_size[3:5]
        vae2_decays = decays[3:5]
        
        # Initialize SVQVAE1 (3 levels)
        self.svqvae1 = SVQVAE(
            img_size=img_size,
            in_channel=in_channel,
            num_classes=num_classes,
            num_vaes=self.svqvae1_levels,
            vae_channels=vae1_channels,
            res_blocks=vae1_res_blocks,
            res_channels=vae1_res_channels,
            embedding_dims=vae1_embedding_dims,
            codebook_size=vae1_codebook_size,
            decays=vae1_decays
        )
        
        # Calculate the input channel size for SVQVAE2's first level
        # It combines SVQVAE1's level 3 and level 1 outputs
        level3_channels = vae1_embedding_dims[2] * 2
        level1_channels = vae1_embedding_dims[0] * 2
        svqvae2_input_channels = level3_channels + level1_channels
        
        # Initialize SVQVAE2 (2 levels)
        self.svqvae2 = nn.ModuleList([])
        
        # First VQVAE of SVQVAE2 - takes concatenated inputs
        self.svqvae2.append(
            VQVAE(
                in_channel=svqvae2_input_channels,
                channel=vae2_channels[0],
                n_res_block=vae2_res_blocks[0],
                n_res_channel=vae2_res_channels[0],
                embed_dim=vae2_embedding_dims[0],
                n_embed=vae2_codebook_size[0],
                decay=vae2_decays[0]
            )
        )
        
        # Second VQVAE of SVQVAE2
        self.svqvae2.append(
            VQVAE(
                in_channel=vae2_embedding_dims[0] * 2,  # First level's output channels
                channel=vae2_channels[1],
                n_res_block=vae2_res_blocks[1],
                n_res_channel=vae2_res_channels[1],
                embed_dim=vae2_embedding_dims[1],
                n_embed=vae2_codebook_size[1],
                decay=vae2_decays[1]
            )
        )
        
        # Calculate the combined encoding size for the classifier
        svqvae1_encoding = (img_size//(4**3))**2 * vae1_embedding_dims[2] * 2
        svqvae2_encoding = (img_size//(4**3))**2 * vae2_embedding_dims[1] * 2
        combined_encoding = svqvae1_encoding + svqvae2_encoding
        self.smallest_encoding = combined_encoding
        
        # Modified classifier for combined encodings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.smallest_encoding, self.smallest_encoding*2),
            nn.GELU(),
            nn.Linear(self.smallest_encoding*2, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1)
        )
        
        # Set model parameters based on training phase
        self.set_training_phase(training_phase)
    
    def set_training_phase(self, phase):
        """Set which parts of the model should be trainable"""
        self.training_phase = phase
        
        # Default: freeze everything
        for param in self.svqvae1.parameters():
            param.requires_grad = False
        for module in self.svqvae2:
            for param in module.parameters():
                param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        # Unfreeze based on phase
        if phase == 'svqvae1' or phase == 'full':
            for param in self.svqvae1.parameters():
                param.requires_grad = True
                
        if phase == 'svqvae2' or phase == 'full':
            for module in self.svqvae2:
                for param in module.parameters():
                    param.requires_grad = True
                    
        if phase == 'classifier' or phase == 'full':
            for param in self.classifier.parameters():
                param.requires_grad = True
    
    def forward(self, input, level=None):
        """Forward pass handling both SVQVAEs based on the level"""
        # In classification mode, return predictions
        if level is None:
            return self.predict(input)
        
        # Handle SVQVAE1 (levels 0, 1, 2)
        if level < self.svqvae1_levels:
            return self.svqvae1(input, level)
        
        # Handle SVQVAE2 (levels 3, 4 mapped to 0, 1)
        else:
            svqvae2_level = level - self.svqvae1_levels
            
            if svqvae2_level == 0:
                # Get the required encodings from SVQVAE1
                level1_encoding = self._get_svqvae1_level1_encoding(input)
                _, _, level3_encoding, _, _, _ = self.svqvae1.encode(input, 0, 2)
                
                # Prepare input for SVQVAE2 level 0
                level3_upsampled = F.interpolate(
                    level3_encoding, 
                    size=level1_encoding.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
                
                # Encode and decode through SVQVAE2 level 0
                quant_t, quant_b, joint_quant, diff, _, _ = self.svqvae2[0].encode(svqvae2_input)
                recon_latent = self.svqvae2[0].decode(joint_quant)
                
                return svqvae2_input, recon_latent, diff
                
            elif svqvae2_level == 1:
                # Get level 0 encoding of SVQVAE2
                level0_encoding = self._get_svqvae2_level0_encoding(input)
                
                # Encode and decode through SVQVAE2 level 1
                quant_t, quant_b, joint_quant, diff, _, _ = self.svqvae2[1].encode(level0_encoding)
                recon_latent = self.svqvae2[1].decode(joint_quant)
                
                return level0_encoding, recon_latent, diff
    
    def encode(self, input, from_level, to_level):
        """Encode through specified levels"""
        
        # Handle special case for SVQVAE2 which needs inputs from SVQVAE1
        if from_level >= self.svqvae1_levels:
            # Get SVQVAE1 encodings needed for SVQVAE2
            level1_encoding = self._get_svqvae1_level1_encoding(input)
            _, _, level3_encoding, _, _, _ = self.svqvae1.encode(input, 0, 2)
            
            # Prepare input for SVQVAE2
            level3_upsampled = F.interpolate(
                level3_encoding, 
                size=level1_encoding.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
            
            # Encode through SVQVAE2 levels
            joint_quant = svqvae2_input
            svqvae2_from = from_level - self.svqvae1_levels
            svqvae2_to = to_level - self.svqvae1_levels
            
            for i in range(svqvae2_from, svqvae2_to + 1):
                quant_t, quant_b, joint_quant, diff, id_t, id_b = self.svqvae2[i].encode(joint_quant)
            
            return quant_t, quant_b, joint_quant, diff, id_t, id_b
        
        # Handle standard encoding through SVQVAE1
        elif to_level < self.svqvae1_levels:
            return self.svqvae1.encode(input, from_level, to_level)
        
        # Handle crossing from SVQVAE1 to SVQVAE2
        else:
            # First encode through SVQVAE1
            _, _, svqvae1_encoding, _, _, _ = self.svqvae1.encode(input, from_level, self.svqvae1_levels - 1)
            
            # Prepare input for SVQVAE2
            level1_encoding = self._get_svqvae1_level1_encoding(input)
            level3_upsampled = F.interpolate(
                svqvae1_encoding, 
                size=level1_encoding.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
            
            # Encode through SVQVAE2
            svqvae2_to = to_level - self.svqvae1_levels
            joint_quant = svqvae2_input
            for i in range(0, svqvae2_to + 1):
                quant_t, quant_b, joint_quant, diff, id_t, id_b = self.svqvae2[i].encode(joint_quant)
            
            return quant_t, quant_b, joint_quant, diff, id_t, id_b
    
    def decode(self, joint_quant, from_level, to_level):
        """Decode from specified levels"""
        
        # If decoding within SVQVAE2
        if from_level >= self.svqvae1_levels and to_level >= self.svqvae1_levels:
            svqvae2_from = from_level - self.svqvae1_levels
            svqvae2_to = to_level - self.svqvae1_levels
            
            for i in range(svqvae2_from, svqvae2_to, -1):
                joint_quant = self.svqvae2[i].decode(joint_quant)
            
            return joint_quant
        
        # If decoding within SVQVAE1
        elif from_level < self.svqvae1_levels and to_level < self.svqvae1_levels:
            return self.svqvae1.decode(joint_quant, from_level, to_level)
        
        # If decoding from SVQVAE2 to SVQVAE1 (complex case)
        else:
            # First decode within SVQVAE2
            svqvae2_from = from_level - self.svqvae1_levels
            for i in range(svqvae2_from, 0, -1):
                joint_quant = self.svqvae2[i].decode(joint_quant)
            
            # SVQVAE2 level 0 output
            # We don't try to cross stacks in sequential training approach
            return joint_quant
            
    def _get_svqvae1_level1_encoding(self, input):
        """Helper to get level 1 encoding from SVQVAE1"""
        with torch.set_grad_enabled(self.training_phase in ['svqvae1', 'full']):
            _, _, joint_quant, _, _, _ = self.svqvae1.models[0].encode(input)
        return joint_quant
    
    def _get_svqvae2_level0_encoding(self, input):
        """Helper to get level 0 encoding from SVQVAE2"""
        with torch.set_grad_enabled(self.training_phase != 'classifier'):
            level1_encoding = self._get_svqvae1_level1_encoding(input)
            _, _, level3_encoding, _, _, _ = self.svqvae1.encode(input, 0, 2)
            
            level3_upsampled = F.interpolate(
                level3_encoding, 
                size=level1_encoding.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
            
            _, _, joint_quant, _, _, _ = self.svqvae2[0].encode(svqvae2_input)
        return joint_quant
    
    def predict(self, input):
        """Generate class predictions using both SVQVAE encodings"""
        # Freeze appropriate gradients based on training phase
        with torch.set_grad_enabled(self.training_phase in ['svqvae1', 'full']):
            # Get SVQVAE1 final encoding
            _, _, svqvae1_encoding, _, _, _ = self.svqvae1.encode(input, 0, self.svqvae1_levels - 1)
        
        with torch.set_grad_enabled(self.training_phase in ['svqvae2', 'full']):
            # Get SVQVAE2 final encoding
            level1_encoding = self._get_svqvae1_level1_encoding(input)
            level3_upsampled = F.interpolate(
                svqvae1_encoding, 
                size=level1_encoding.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            svqvae2_input = torch.cat([level3_upsampled, level1_encoding], dim=1)
            
            _, _, svqvae2_level0, _, _, _ = self.svqvae2[0].encode(svqvae2_input)
            _, _, svqvae2_encoding, _, _, _ = self.svqvae2[1].encode(svqvae2_level0)
        
        # Combine encodings for classification
        combined_encoding = torch.cat([
            svqvae1_encoding.flatten(1),
            svqvae2_encoding.flatten(1)
        ], dim=1)
        
        # Run through classifier
        probs = self.classifier(combined_encoding)
        
        return probs