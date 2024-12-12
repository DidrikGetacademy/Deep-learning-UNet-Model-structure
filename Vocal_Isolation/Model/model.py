import torch
import torch.nn as nn
import torch.nn.functional as F
#Defines a class UNET that inheritate from nn.modules that is standard for all neural networks in pytorch
#Initializing method for U-net model. Amount of incoming channels and out going, the features defines amount of features maps in each layer of the encoder and decoder
# calls the initialize method to nn.module
class UNet(nn.Module): 
    def __init__(self,in_channels=1,out_channels=1,features=[64,128,256,512]): 
        super(UNet,self).__init__() 
        
        
        #initalize a empty list for encoder-layers
        self.encoder = nn.ModuleList() 
        
        
        
        #foreach element in features-list exsample 64,128,256,512
        #updates in_channels for next layer
        for feature in features:
            #adds the block to encoder
            self.encoder.append(self.conv_block(in_channels,feature)) #calls conv_block method to make a konvolusjonsblokk
            in_channels = feature 
            
            
            
        #BottleNeck
        #Gets the last element in the features list, doubles it and saves the bottleneck block.
        self.bottleneck = self.conv_block(features[-1],features[-1] * 2)
        
        
        
        #loops threw the features list in reverse.
        self.decoder = nn.ModuleList()##initalize a empty list for Decoder-layers
        for feature in reversed(features):
            self.decoder.append(
                #ConvTranspose2d: Up-sampling
                nn.ConvTranspose2d(features[-1]*2,feature,kernel_size=2,stride=2) #ConvTranspose2d parameters: amount of upsampling witch is doubleneck that is the double amount from bottleneck, feature: amount of outgoing connections for up-sampling, Kernel & stride: parameters that decide the up-sampling size and steps.
            )
            self.decoder.append(self.conv_block(features[-1]*2,feature)) #The self.decoder adds a konvolusjonblokk after upsampling
            features[-1] = feature   #features[-1] = feature: updates features[-1] for the next layer.
            
        
        #The final convolution: Reduces the number of output channels.
        #self.final_conv: an konvolusjon with kernel size 1 that redces amount of channels to out_channels
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)
        
        
        
        
    def conv_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1), #Retrieve basic functions from the data.
            nn.BatchNorm2d(out_channels),#Stabilize traing by normalizing output too have a average at 0 and standard devation at 1
            nn.ReLU(inplace=True), #Sets all negative values too 0 so the model can learn more complex relation in data.
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)      
            )        
    
    
    
    
    def forward(self,x):
        skip_connections = [] #Stores the outputs from encoder block for skip connections.
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x,kernel_size=2,stride=2) #DownSampling
        
        x = self.bottleneck(x) # pass through the bottleneck 
        
        skip_connections = skip_connections[::-1] #Reveres the skip connection.
        
        for idx in range(0,len(self.decoder),2):
            x = self.decoder[idx](x) #Upsampling
            skip_connection = skip_connections[idx//2]
            
            
            #Handle mismatched dimensions.
            if x.shape != skip_connection.shape:
                x = F.interpolate(x,size=skip_connection.shape[2:])               
                
            x = torch.cat((skip_connection,x),dim=1) #Concatenate skip connections
            x = self.decoder[idx+1](x) #Pass though convolutional block
            
        return self.final_conv(x) #Output layer
                
        
        
        




