import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module): 
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]): 
        super(UNet, self).__init__()

        # encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        # decoder
        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))
        for idx, feature in enumerate(reversed_features):
            input_channels = features[-1] * 2 if idx == 0 else reversed_features[idx - 1]
            self.decoder.append(
                nn.ConvTranspose2d(input_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.conv_block(feature * 2, feature))

        # final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # bottleneck
        x = self.bottleneck(x)

        # reverse skip connections
        skip_connections = skip_connections[::-1]

        # decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsampling
            skip_connection = skip_connections[idx // 2]

            # Ensure the dimensions match before concatenation
            if not torch.equal(torch.tensor(x.shape[2:]), torch.tensor(skip_connection.shape[2:])):
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip_connection, x), dim=1)  # concatenate skip connection
            x = self.decoder[idx + 1](x)  # convolutional block

        return self.final_conv(x)
