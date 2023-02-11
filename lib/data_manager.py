import numpy as np
import h5py

class DataManager(object):
    """Managing data
    load, normalize, shuffle, and split data
    """
    def __init__(self, path):
        self.path = path

    def loads_data(self):
        """load data
        """
        self.data_input_train = np.load(self.path + 'data_train_input.npy')
        self.data_label_train = np.load(self.path + 'data_train_label.npy')
        self.data_input_test = np.load(self.path + 'data_test_input.npy')
        self.data_label_test = np.load(self.path + 'data_test_label.npy')    

    def normalize_data(self):
        """normalize data
        
        for input
        normalize with min and max of train data
        for label
        normalize to N(0,1) for individuality properties
        normalize to [0,1] for disease severitty
        """

        # normalize input
        self.train_input_min = np.min(self.data_input_train)
        self.train_input_max = np.max(self.data_input_train)
        self.data_input_train = (self.data_input_train - self.train_input_min) / (self.train_input_max - self.train_input_min)
        self.data_input_test = (self.data_input_test - self.train_input_min) / (self.train_input_max - self.train_input_min)

        # normalize label
        self.data_label_train[:, :, :5] = (self.data_label_train[:, :, :5] - np.min(np.min(self.data_label_train[:, :, :5], 0), 0))/ \
            (np.max(np.max(self.data_label_train[:, :, :5], 0), 0) - np.min(np.min(self.data_label_train[:, :, :5], 0), 0))
        self.data_label_train[:, :, 5:] = self.data_label_train[:, :, 5:] /10
        self.data_severity_train = self.data_label_train[:, :, 5]

        self.data_label_test[:, :, :5] = (self.data_label_test[:, :, :5] - np.min(np.min(self.data_label_test[:, :, :5], 0), 0))/ \
            (np.max(np.max(self.data_label_test[:, :, :5], 0), 0) - np.min(np.min(self.data_label_test[:, :, :5], 0), 0))
        self.data_label_test[:, :, 5:] = self.data_label_test[:, :, 5:] /100
        self.data_severity_test = self.data_label_test[:, :, 5]
    
    def save_mat2npy(self, dir_path = "."):
        """convernt mat file to numpy file.

        dir_path: savc
        """
        L_info_train = [0, 1, 2] #length
        L_info_test = [0, 1, 2]
        D_info_train = [0, 1, 2] #Diameter
        D_info_test = [0, 1, 2]
        T_info_train = [0, 1, 2] #Thickness
        T_info_test = [0, 1, 2]
        E_info_train = [0, 1, 2] #Youngs Modulus
        E_info_test = [0, 1, 2]
        R_info_train = [0, 1] #Resistance
        R_info_test = [2]
        svlv_train = np.arange(9)
        svlv_test = np.arange(81)
        arteries = [16, 54]

        f = h5py.File('{}/raw_data_train_small.mat'.format(dir_path))
        tr_in = np.transpose(f['tr_data'][()])
        tr_in = tr_in[L_info_train, :, :, :, :, :, :, :]
        tr_in = tr_in[:, D_info_train, :, :, :, :, :, :]
        tr_in = tr_in[:, :, T_info_train, :, :, :, :, :]
        tr_in = tr_in[:, :, :, E_info_train, :, :, :, :]
        tr_in = tr_in[:, :, :, :, R_info_train, :, :, :]
        tr_in = np.reshape(tr_in, [tr_in.shape[0], tr_in.shape[1], tr_in.shape[2], tr_in.shape[3], tr_in.shape[4], 10, 100, tr_in.shape[6], tr_in.shape[7]])
        tr_in = np.transpose(tr_in, [0, 1, 2, 3, 4, 6, 5, 7, 8])
        tr_in = np.reshape(tr_in, [-1, 1, tr_in.shape[6], tr_in.shape[7], tr_in.shape[8]])
        tr_in = np.transpose(tr_in, [0, 2, 1, 3, 4])
        tr_in = tr_in[:, :, :, arteries, :]
        tr_in = tr_in[:, svlv_train, :, : ,:]
        np.save('{}/data_train_input.npy'.format(dir_path), tr_in)

        f = h5py.File('{}/raw_label_train_small.mat'.format(dir_path))
        tr_lb = np.transpose(f['tr_label'][()])
        tr_lb = tr_lb[L_info_train, :, :, :, :, :]
        tr_lb = tr_lb[:, D_info_train, :, :, :, :]
        tr_lb = tr_lb[:, :, T_info_train, :, :, :]
        tr_lb = tr_lb[:, :, :, E_info_train, :, :]
        tr_lb = tr_lb[:, :, :, :, R_info_train, :]
        tr_lb = np.reshape(tr_lb, [tr_lb.shape[0], tr_lb.shape[1], tr_lb.shape[2], tr_lb.shape[3], tr_lb.shape[4], 10, 100, tr_lb.shape[6]])
        tr_lb = np.transpose(tr_lb, [0, 1, 2, 3, 4, 6, 5, 7])
        tr_lb = np.reshape(tr_lb, [-1, tr_lb.shape[6], tr_lb.shape[7]])
        tr_lb = tr_lb[:, svlv_train, :]
        tr_lb[:, :, 5] = tr_lb[:, :, 5]/10
        np.save('{}/data_train_label.npy'.format(dir_path), tr_lb)
        print('train DONE')

        f = h5py.File('{}/raw_data_test_small.mat'.format(dir_path))
        ts_in = np.transpose(f['ts_data'][()])
        ts_in = ts_in[L_info_test, :, :, :, :, :, :, :]
        ts_in = ts_in[:, D_info_test, :, :, :, :, :, :]
        ts_in = ts_in[:, :, T_info_test, :, :, :, :, :]
        ts_in = ts_in[:, :, :, E_info_test, :, :, :, :]
        ts_in = ts_in[:, :, :, :, R_info_test, :, :, :]
        ts_in = np.reshape(ts_in, [ts_in.shape[0], ts_in.shape[1], ts_in.shape[2], ts_in.shape[3], ts_in.shape[4], 91, 5, ts_in.shape[6], ts_in.shape[7]])
        ts_in = np.transpose(ts_in, [0, 1, 2, 3, 4, 6, 5, 7, 8])
        ts_in = np.reshape(ts_in, [-1, 1, ts_in.shape[6], ts_in.shape[7], ts_in.shape[8]])
        ts_in = np.transpose(ts_in, [0, 2, 1, 3, 4])
        ts_in = ts_in[:, :, :, arteries, :]
        ts_in = ts_in[:, svlv_test, :, : ,:]
        np.save('{}/data_test_input.npy'.format(dir_path), ts_in)

        f = h5py.File('{}/raw_label_test_small.mat'.format(dir_path))
        ts_lb = np.transpose(f['ts_label'][()])
        ts_lb = ts_lb[L_info_test, :, :, :, :, :]
        ts_lb = ts_lb[:, D_info_test, :, :, :, :]
        ts_lb = ts_lb[:, :, T_info_test, :, :, :]
        ts_lb = ts_lb[:, :, :, E_info_test, :, :]
        ts_lb = ts_lb[:, :, :, :, R_info_test, :]
        ts_lb = np.reshape(ts_lb, [ts_lb.shape[0], ts_lb.shape[1], ts_lb.shape[2], ts_lb.shape[3], ts_lb.shape[4], 91, 5, ts_lb.shape[6]])
        ts_lb = np.transpose(ts_lb, [0, 1, 2, 3, 4, 6, 5, 7])
        ts_lb = np.reshape(ts_lb, [-1, ts_lb.shape[6], ts_lb.shape[7]])
        ts_lb = ts_lb[:, svlv_test, :]
        ts_lb[:, :, 5] = ts_lb[:, :, 5]/100
        np.save('{}/data_test_label.npy'.format(dir_path), ts_lb)
        print('test DONE')