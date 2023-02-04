import torch.nn as nn

#%% Pre-setting
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#%% Network blocks
class featureextractor(nn.Module):
    def __init__(self, input_dim = 1):
        super(featureextractor, self).__init__()
        self.input_dim = input_dim

        self.conv = nn.Sequential(
            #C1 L2*66
            nn.Conv2d(self.input_dim, 96, (1, 5), stride = (1, 1), padding = (0, 2)),
            nn.LeakyReLU(0.1),
            #C96 L2*66        
            nn.Conv2d(96, 256, (1, 3), stride = (1, 1), padding = (0, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            #C256 L2*66
            nn.Conv2d(256, 384, (1, 3), stride = (1, 1), padding = (0, 1)),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1),
            #C384 L2*66         
            nn.Conv2d(384, 384, (1, 3), stride = (1, 1), padding = (0, 1)),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1),
            #C384 L2*66         
            nn.Conv2d(384, 256, (1, 3), stride = (1, 1), padding = (0, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            #C256 L2*66        
        )

    def forward(self, input):
        feature = self.conv(input)
        feature = feature.view(-1, 256*66*2)
        return feature

class regressor(nn.Module):
    def __init__(self, output_dim = 1):
        super(regressor, self).__init__()
        self.output_dim = output_dim
        
        self.reg1 = nn.Sequential(
            nn.Linear(256*66*2, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1)
        )
        
        self.reg2 = nn.Sequential(
            nn.Linear(64, self.output_dim)
        )

    def forward(self, input):
        r1 = self.reg1(input)
        r = self.reg2(r1)
        r = r.view(-1)
        return r

class regularizer(nn.Module):
    def __init__(self, output_dim = 1):
        super(regularizer, self).__init__()
        self.output_dim = output_dim
        
        self.REG1 = nn.Sequential(
            nn.Linear(256*66*2, 64),
            nn.LeakyReLU(0.1)
        )
        self.REG2 = nn.Sequential(
            nn.Linear(64, self.output_dim)
        )
        
    def forward(self, input):
        reg1 = self.REG1(input)
        reg = self.REG2(reg1)
        reg = reg.view(-1)
        return reg