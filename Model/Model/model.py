import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.relU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,  x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi
        


class MultiScaleDecoderBlock(nn.module):
    def __init__(self, in_channels, out_channels):
        




class UNet(nn.Module): 
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]): 
        super(UNet, self).__init__()

        
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

      
        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))
        for idx, feature in enumerate(reversed_features):
            input_channels = features[-1] * 2 if idx == 0 else reversed_features[idx - 1]
            self.decoder.append(
                nn.ConvTranspose2d(input_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.conv_block(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Add dropout here
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

       
        x = self.bottleneck(x)

      
        skip_connections = skip_connections[::-1]

        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x) 
            
            skip_connection = skip_connections[idx // 2]
            
            attention = AttentionBlock(skip_connection.size(1), x.size(1), x.size(1) // 2)
            x = attention(skip_connection, x)
            x = torch.cat((x, skip_connection), dim=1)
        
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)


            x = torch.cat((skip_connection, x), dim=1) 
            x = self.decoder[idx + 1](x)  

        return self.final_conv(x)
