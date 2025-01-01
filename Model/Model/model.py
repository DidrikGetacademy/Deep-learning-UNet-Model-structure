import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Model.Logging.Logger import setup_logger


train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')

class AttentionBlock(nn.Module):
    #Applies an attention mechanism to combine gating and encoder features.


    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):

        #x: Skip connection features
        #g: Gating features

        train_logger.debug(f"[AttentionBlock] Input shapes -> x: {x.shape}, g: {g.shape}")

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        #Just log shape and min/max
        out = x * psi
        train_logger.debug(
            f"[AttentionBlock] Output shape: {out.shape}, "
            f"psi min={psi.min().item():.4f}, psi max={psi.max().item():.4f}"
        )
        return out


class MultiScaleDecoderBlock(nn.Module):
    #Decoding block that up-samples and merges with skip connection.

    def __init__(self, in_channels, out_channels):
        super(MultiScaleDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
    
        #x: Deeper feature map to be up-sampled
        #skip: Corresponding skip-connection feature map
    
        train_logger.debug(
            f"[MultiScaleDecoderBlock] Input shapes -> x: {x.shape}, skip: {skip.shape}"
        )

        x = self.up(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
            Model_logger.debug(
                f"[MultiScaleDecoderBlock] Resized upsampled x to: {x.shape}"
            )

        # Concatenate upsampled and skip
        x = torch.cat((x, skip), dim=1)
        train_logger.debug(f"[MultiScaleDecoderBlock] After concat: {x.shape}")

        out = self.conv(x)
        train_logger.debug(f"[MultiScaleDecoderBlock] Output shape: {out.shape}")
        return out


class UNet(nn.Module):
    #A UNet model with attention and multi-scale decoder blocks.


    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        prev_channels = in_channels

        # Encoder Blocks
        for feature in features:
            self.encoder.append(self.conv_block(prev_channels, feature))
            prev_channels = feature
            train_logger.debug(f"[UNet] Added encoder block with out_channels: {feature}")

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)
        train_logger.debug(f"[UNet] Bottleneck channels: {features[-1] * 2}")

        # Decoder (with MultiScaleDecoderBlock + AttentionBlock)
        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))
        for idx, feature in enumerate(reversed_features):
            # For the first decoder block, the in_channels = last encoder channel * 2
            input_channels = features[-1] * 2 if idx == 0 else reversed_features[idx - 1]
            self.decoder.append(MultiScaleDecoderBlock(input_channels, feature))
            self.decoder.append(AttentionBlock(feature, feature, feature // 2))
            train_logger.debug(
                f"[UNet] Added decoder & attention block -> in:{input_channels}, out:{feature}"
            )

        # Final 1x1 Conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        train_logger.info(f"[UNet] Final conv initialized: in={features[0]}, out={out_channels}")

    def conv_block(self, in_channels, out_channels):
        #Double convolution block (with InstanceNorm and optional Dropout).
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  # Modify if needed
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Ensure float32
        x = x.to(dtype=torch.float32)
        train_logger.debug(f"[UNet forward] Input shape: {x.shape}")

        skip_connections = []

        # Encoder forward pass
        for idx, enc in enumerate(self.encoder):
            x = enc(x)
            skip_connections.append(x)
            train_logger.debug(f"[UNet forward] Encoder[{idx}] output: {x.shape}")
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)
        train_logger.debug(f"[UNet forward] Bottleneck output: {x.shape}")

        # Reverse skip connections (for matching with decoder blocks)
        skip_connections = skip_connections[::-1]

        # Decoder + Attention
        for idx in range(0, len(self.decoder), 2):
            dec_block = self.decoder[idx]
            att_block = self.decoder[idx + 1]

            # MultiScaleDecoderBlock
            x = dec_block(x, skip_connections[idx // 2])
            # AttentionBlock
            x = att_block(skip_connections[idx // 2], x)
            train_logger.debug(f"[UNet forward] Decoder stage {idx//2}, output: {x.shape}")

        out = self.final_conv(x)
        train_logger.debug(f"[UNet forward] Final output shape: {out.shape}")
        return out
    