import torch, time, pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.autograd import Variable
from lib.neural_networks import *

#%% Model Building
class Regressor(object):
    def __init__(self, epoch, batch_size, model_dir, lrFE, lrSV, lrREG, lossREG, beta1, beta2, device = "cuda"):
        self.device = torch.device(device)
        # parameters
        self.epoch = epoch
        self.batch_size = batch_size
        self.save_dir = model_dir
        self.lossREG = lossREG
        #for GradCAM
        self.load_dir = model_dir
        self.feature = None
        self.gradient = None
        #end

        # networks init
        self.FE = featureextractor().to(self.device)
        self.SV = regressor().to(self.device)
        self.REG_L = regularizer().to(self.device)
        self.REG_PWV = regularizer().to(self.device)
        
        self.FE.apply(weights_init)
        self.SV.apply(weights_init)
        self.REG_L.apply(weights_init)
        self.REG_PWV.apply(weights_init)

        self.FE_optimizer = optim.Adam(self.FE.parameters(), lr=lrFE, betas=(beta1, beta2))
        self.SV_optimizer = optim.Adam(self.SV.parameters(), lr=lrSV, betas=(beta1, beta2))
        self.REG_L_optimizer = optim.Adam(self.REG_L.parameters(), lr=lrREG, betas=(beta1, beta2))
        self.REG_PWV_optimizer = optim.Adam(self.REG_PWV.parameters(), lr=lrREG, betas=(beta1, beta2))

        self.MSE_loss = nn.MSELoss()

    def train(self, tr_in, tr_lb, ts_in, ts_sv):
        self.train_hist = {}
        self.train_hist['SV_loss_train'] = []
        self.train_hist['SV_loss_test'] = []
        self.train_hist['REG_loss_train'] = []

        ts_in = np.transpose(ts_in, [0, 2, 1, 3, 4])
        ts_sv = np.transpose(ts_sv, [0, 2, 1])
        self.x_train = np.reshape(tr_in, [tr_in.shape[0] * tr_in.shape[1], 1, tr_in.shape[2], tr_in.shape[3]])
        self.x_test = np.reshape(ts_in, [ts_in.shape[0] * ts_in.shape[1], ts_in.shape[2], 1, ts_in.shape[3], ts_in.shape[4]])
        self.y_train = np.reshape(tr_lb, [tr_lb.shape[0] * tr_lb.shape[1], tr_lb.shape[2]])
        self.y_test = np.reshape(ts_sv, [ts_sv.shape[0] * ts_sv.shape[1], ts_sv.shape[2]])
        x_test = self.x_test
        y_test = self.y_test

        print('Training START!')
        start_time = time.clock()
        for epoch in range(self.epoch):
            #data on cuda
            rndidx = torch.randint(self.x_train.shape[0], (int(self.batch_size), ))
            x_ = self.x_train[rndidx]
            l_ = self.y_train[rndidx, 0]
            sv_ = self.y_train[rndidx, 4]

            x_ = torch.from_numpy(x_)
            l_ = torch.from_numpy(l_)
            sv_ = torch.from_numpy(sv_)

            x_ = x_.to(device=self.device, dtype=torch.float)
            l_ = l_.to(device=self.device, dtype=torch.float)
            sv_ = sv_.to(device=self.device, dtype=torch.float)
            
            # update Regressor network
            FE_ = self.FE(x_)
            SV_ = self.SV(FE_)
            L_ = self.REG_L(FE_)
            SV_loss = self.MSE_loss(sv_, SV_)
            L_loss = self.MSE_loss(l_, L_)
            N_L_loss = self.lossREG * (1 - torch.tanh(L_loss))
            N_loss = (N_L_loss + N_L_loss)
            self.train_hist['SV_loss_train'].append(np.sqrt(SV_loss.item()))
            self.train_hist['REG_loss_train'].append(np.sqrt((L_loss.item() + L_loss.item())/2))
            
            self.FE_optimizer.zero_grad()
            self.SV_optimizer.zero_grad()
            SV_N_loss = SV_loss + N_loss
            SV_N_loss.backward(retain_graph=True)
            self.SV_optimizer.step()
            self.FE_optimizer.step()
            
            self.REG_L_optimizer.zero_grad()
            L_loss.backward(retain_graph=True)
            self.REG_L_optimizer.step()
                        
            if ((epoch + 1) % 10000) == 0:
                SV_loss_test = 0
                for jj in range(x_test.shape[0]):
                    x_test_batch = x_test[jj, :, :, :, :]
                    y_test_batch = y_test[jj]
                    x_test_batch = torch.from_numpy(x_test_batch)
                    y_test_batch = torch.from_numpy(y_test_batch)
                    x_test_batch = x_test_batch.to(device=self.device, dtype=torch.float)
                    y_test_batch = y_test_batch.to(device=self.device, dtype=torch.float)
                    fe_test_batch = self.FE(x_test_batch)
                    sv_test_batch = self.SV(fe_test_batch)
                    SV_loss_batch = self.MSE_loss(y_test_batch, sv_test_batch)
                    SV_loss_test = SV_loss_test + SV_loss_batch.item()
                SV_loss_test = SV_loss_test/int(x_test.shape[0])
                self.train_hist['SV_loss_test'].append(np.sqrt(SV_loss_test))
                end_time = time.clock()
                time_iter50=end_time-start_time
                start_time = time.clock()
                print("Epoch: [%2d] SV_loss_train: %.8f REG_loss_train: %.8f SV_loss_test: %.8f Time: [%2d]" %
                    ((epoch + 1), np.sqrt(SV_loss.item()), np.sqrt(L_loss.item()) + np.sqrt(L_loss.item()), np.sqrt(SV_loss_test), time_iter50))
        self.save(epoch+1)
                
        print("Training finish!... save training results")
        with open(self.save_dir + '_history.pkl', 'wb') as f:
            pickle.dump(self.train_hist, f)
        self.loss_plot(self.train_hist)
    
    def test(self, ts_in, ts_sv, tr_sv, tepoch):
        self.FE.load_state_dict(torch.load(self.load_dir + '_FE' + str(tepoch) + '.pkl'))
        self.SV.load_state_dict(torch.load(self.load_dir + '_SV' + str(tepoch) + '.pkl'))
        self.FE.eval()
        self.SV.eval()
        ts_in = np.transpose(ts_in, [0, 2, 1, 3, 4])
        ts_sv = np.transpose(ts_sv, [0, 2, 1])
        self.x_test = np.reshape(ts_in, [ts_in.shape[0] * ts_in.shape[1], ts_in.shape[2], 1, ts_in.shape[3], ts_in.shape[4]])
        self.y_test = np.reshape(ts_sv, [ts_sv.shape[0] * ts_sv.shape[1], ts_sv.shape[2]])
        x_test = self.x_test
        y_test = self.y_test
        ylim = np.max(self.y_test)

        r_test = np.zeros((y_test.shape[0], y_test.shape[1]))
        R_loss_test = 0
        r_mean = np.zeros(x_test.shape[1])
        r_std = np.zeros(x_test.shape[1])
        fig = plt.figure(figsize = (8, 5))
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
        r2score = r2_score(y_test.flatten(), r_test.flatten())
        r_mean = np.mean(r_test, axis = 0)
        r_std = np.std(r_test, axis = 0)
        R_loss_test = np.sqrt(R_loss_test/int(x_test.shape[0]))
        r_xaxis = np.arange(x_test.shape[1])
        r_xaxis_poly = np.concatenate((r_xaxis, np.flip(r_xaxis)), axis = 0)
        
        x_sv = np.unique(tr_sv)
        y_sv = np.ones(x_sv.shape[0])
        
        plt.plot(r_xaxis, r_mean * 100, 'r', linewidth = 3)
        r_poly1 = np.concatenate((r_mean + 1.96 * r_std, np.flip(r_mean - 1.96 * r_std)), axis = 0)
        plt.fill(r_xaxis_poly, r_poly1 * 100, alpha = 0.4, color = 'r')
        plt.plot([-100, 100], [-100, 100], 'k--', linewidth = 3)
        plt.legend(('Mean of estimation', 'Ideal estimation', '95% Confidence interval'), fontsize = 15, loc = 'lower right')
        plt.plot(x_sv * 100, y_sv, 'k*')
        plt.xlim([0, ylim * 100])
        plt.ylim([0, ylim * 100 + 10])
        plt.title('MSE: ' + str(np.round_(R_loss_test, 3)) + ',  R2: ' + str(np.round_(r2score, 3)))
        plt.xlabel('Real Severity (Inclusion [%])', fontsize = 20)
        plt.ylabel('Estimated Severity (DL)', fontsize = 20)
        plt.tight_layout()
        R_loss_test = R_loss_test/int(x_test.shape[-1])
        path = self.load_dir + 'Result' + '.png'
        fig.savefig(path)
        plt.close(fig)
        return y_test, r_test
        
    def ABItest(self, tr_in, tr_sv, ts_in, ts_sv):
        ts_in = np.transpose(ts_in, [0, 2, 1, 3, 4])
        ts_sv = np.transpose(ts_sv, [0, 2, 1])
        abi_in = np.reshape(ts_in, [ts_in.shape[0] * ts_in.shape[1], ts_in.shape[2], 1, ts_in.shape[3], ts_in.shape[4]])
        abi_sv = np.reshape(ts_sv, [ts_sv.shape[0] * ts_sv.shape[1], ts_sv.shape[2]])
        abi_ankle = abi_in[:, :, :, 0, :]
        abi_arm = abi_in[:, :, :, 1, :]
        ylim = np.max(abi_sv)
        max_ankle = np.max(abi_ankle, axis = 3).squeeze()
        max_arm = np.max(abi_arm, axis = 3).squeeze()
        abi = max_ankle/max_arm
        ## calibration
        yy = np.reshape(tr_sv, [-1])
        abi_ankle_tr = tr_in[:,:,0,:]
        abi_arm_tr = tr_in[:,:,1,:]
        max_ankle_tr = np.max(abi_ankle_tr, axis = 2).squeeze()
        max_arm_tr = np.max(abi_arm_tr, axis = 2).squeeze()
        abi_tr = max_ankle_tr/max_arm_tr
        xx1_tr = np.reshape(abi_tr, [-1, 1])
        xx2_tr = np.power(xx1_tr, 2)
        xx_tr = np.concatenate((xx1_tr, xx2_tr), axis = 1)
        reg = LinearRegression().fit(xx_tr, yy)
        [a1, a2] = reg.coef_
        abi_cali = reg.intercept_ + a1 * abi + a2 * np.power(abi, 2)
        ## Raw ABI
        x_sv = np.unique(tr_sv)
        y_sv = np.ones(x_sv.shape[0])
        fig = plt.figure(figsize = (8, 5))
        r_mean = np.mean(abi, axis = 0)
        r_std = np.std(abi, axis = 0)
        r_xaxis = np.arange(abi_in.shape[1])
        r_xaxis_poly = np.concatenate((r_xaxis, np.flip(r_xaxis)), axis = 0)
        plt.plot(r_xaxis, r_mean, 'r')
        r_poly1 = np.concatenate((r_mean + 1.96 * r_std, np.flip(r_mean - 1.96 * r_std)), axis = 0)
        plt.fill(r_xaxis_poly, r_poly1, alpha = 0.4, color = 'r')
        plt.plot(x_sv * 100, y_sv, 'k*')
        plt.legend(('Mean of ABI', '95% Confidence interval'), fontsize = 15)
        plt.xlim([0, ylim * 100])
        # plt.ylim([0, 20])
        plt.xlabel('Real Severity (Inclusion [%])', fontsize = 20)
        plt.ylabel('ABI', fontsize = 20)
        plt.tight_layout()
        path = self.load_dir + 'ABI' + '.png'
        fig.savefig(path)
        plt.close(fig)

        ## Calibrated ABI
        fig = plt.figure(figsize = (8, 5))
        r_mean = np.mean(abi_cali * 100, axis = 0)
        r_std = np.std(abi_cali * 100, axis = 0)
        r_xaxis = np.arange(abi_in.shape[1])
        r_xaxis_poly = np.concatenate((r_xaxis, np.flip(r_xaxis)), axis = 0)
        plt.plot(r_xaxis, r_mean, 'r', linewidth = 3)
        r_poly1 = np.concatenate((r_mean + 1.96 * r_std, np.flip(r_mean - 1.96 * r_std)), axis = 0)
        plt.fill(r_xaxis_poly, r_poly1, alpha = 0.4, color = 'r')
        plt.plot([-100, 100], [-100, 100], 'k--', linewidth = 3)
        plt.plot(x_sv * 100, y_sv, 'k*')
        plt.legend(('Mean of estimation', 'Ideal estimation', '95% Confidence interval'), fontsize = 15, loc = 'lower right')
        plt.xlim([0, ylim * 100])
        plt.ylim([0, 90])
        plt.xlabel('Real Severity (Inclusion [%])', fontsize = 20)
        plt.ylabel('Estimated Severity (ABI)', fontsize = 20)
        plt.tight_layout()
        path = self.load_dir + 'ABI_Cali' + '.png'
        fig.savefig(path)
        plt.close(fig)
        return abi_sv, abi_cali, abi
        
    def training_result(self):
        with open(self.load_dir + '_history.pkl', 'rb') as f:
            train_hist = pickle.load(f) # 단 한줄씩 읽어옴
        self.loss_plot(train_hist)
        return train_hist
    
    def loss_plot(self, hist):        
        fig0 = plt.figure()
        x0 = range(len(hist['SV_loss_train']))
        y0 = hist['SV_loss_train']
        plt.plot(x0, y0, label='SV_loss_train')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        path = self.load_dir + 'training_loss.png'
        fig0.savefig(path)       
                    
        fig1 = plt.figure()
        x1 = range(len(hist['SV_loss_test']))
        y1 = hist['SV_loss_test']
        plt.plot(x1, y1, label='SV_loss_test')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        path = self.load_dir + 'test_loss.png'
        fig1.savefig(path)

    def save(self, epoch):
        torch.save(self.FE.state_dict(), self.save_dir + '_FE_' + str(epoch) + '.pkl')
        torch.save(self.SV.state_dict(), self.save_dir + '_SV_' + str(epoch) + '.pkl')

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