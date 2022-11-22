import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch

class ABITest(object):
    def __init__(self):
        self.MSE_loss = nn.MSELoss()
        
    def test(self, tr_in, tr_sv, ts_in, ts_sv):
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
        sv = torch.from_numpy(abi_sv.flatten())
        cali = torch.from_numpy(abi_cali.flatten())
        R_loss_abi = np.sqrt(self.MSE_loss(sv, cali).item())
        return abi_sv, abi_cali, abi, R_loss_abi