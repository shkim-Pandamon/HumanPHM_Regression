import torch, time, pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

#%% Pre-setting
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#%% Network building
class featureextractor(nn.Module):
    def __init__(self, input_dim = 1):
        super(featureextractor, self).__init__()
        self.ngpu=1
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
        self.ngpu=1
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
        self.ngpu=1
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

class Regressor(object):
    def __init__(self, device = "cuda:0"):
        self.device = torch.device(device)
        self.feature = None
        self.gradient = None

        # networks init
        self.FE = featureextractor().to(self.device)
        self.SV = regressor().to(self.device)
        self.FE.apply(weights_init)
        self.SV.apply(weights_init)
        self.MSE_loss = nn.MSELoss()
    
    def test(self, ts_in, ts_sv, model_dir, epoch):
        self.FE.load_state_dict(torch.load('{}_FE_{}.pkl'.format(model_dir, epoch)))
        self.SV.load_state_dict(torch.load('{}_SV_{}.pkl'.format(model_dir, epoch)))
        self.FE.eval()
        self.SV.eval()
        ts_in = np.transpose(ts_in, [0, 2, 1, 3, 4])
        ts_sv = np.transpose(ts_sv, [0, 2, 1])
        x_test = np.reshape(ts_in, [ts_in.shape[0] * ts_in.shape[1], ts_in.shape[2], 1, ts_in.shape[3], ts_in.shape[4]])
        y_test = np.reshape(ts_sv, [ts_sv.shape[0] * ts_sv.shape[1], ts_sv.shape[2]])

        r_test = np.zeros((y_test.shape[0], y_test.shape[1]))
        R_loss_test = 0
        for jj in range(x_test.shape[0]):
            x_test_batch = x_test[jj, :, :, :, :]
            y_test_batch = y_test[jj]
            x_test_batch = torch.from_numpy(x_test_batch)
            y_test_batch = torch.from_numpy(y_test_batch)
            x_test_batch = x_test_batch.to(device=self.device, dtype=torch.float)
            y_test_batch = y_test_batch.to(device=self.device, dtype=torch.float)
            fe_test_batch = self.FE(x_test_batch)
            r_test_batch = self.SV(fe_test_batch)
            R_loss_batch = self.MSE_loss(y_test_batch, r_test_batch)
            R_loss_test = R_loss_test + R_loss_batch.item()            
            y_test_batch = y_test_batch.detach()
            y_test_batch = y_test_batch.cpu().numpy()
            r_test_batch = r_test_batch.detach()
            r_test_batch = r_test_batch.cpu().numpy()
            x_test_batch = x_test_batch.detach()
            x_test_batch = x_test_batch.cpu().numpy()
            r_test[jj, :] = r_test_batch
        return x_test, y_test, r_test, R_loss_test

    def feature_vector(self, da_in, tepoch):
        self.FE.load_state_dict(torch.load(self.load_dir + '_FE' + str(tepoch) + '.pkl'))
        self.SV.load_state_dict(torch.load(self.load_dir + '_SV' + str(tepoch) + '.pkl'))
        self.FE.eval()
        self.SV.eval()
        fvec = np.zeros((da_in.shape[0], 64))
        da_in = torch.from_numpy(da_in)
        da_in = da_in.to(device=self.device, dtype=torch.float)
        datas = Variable(da_in)
        for i in range(datas.size(0)):
            feature = datas[i].unsqueeze(0)
            feature = self.FE(feature)
            for name, module in self.SV.named_children():
                if name == 'reg1':
                    feature = feature.view(feature.size(0), -1)
                if not name =='reg2': 
                    feature = module(feature)
            feature = feature.cpu()
            fvec[i] = feature.detach().numpy()
        return fvec
