import numpy as np

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
        self.data_input_test = np.reshape(self.data_input_test, [self.data_input_test.shape[0], 81, 10, self.data_input_test.shape[2], self.data_input_test.shape[3]])
        self.data_label_test = np.reshape(self.data_label_test, [self.data_label_test.shape[0], 81, 10])    

    def normalize_data(self):
        """normalize data
        
        for input
        normalize with min and max of train data
        
        for label
        normalize to N(0,1) for individuality properties
        normalize to [0,1] for disease severitty
        """
        self.data_label_train[:, :, :4] = (self.data_label_train[:, :, :4] - np.min(np.min(self.data_label_train[:, :, :4], 0), 0))/ \
            (np.max(np.max(self.data_label_train[:, :, :4], 0), 0) - np.min(np.min(self.data_label_train[:, :, :4], 0), 0))
        self.data_label_train[:, :, 4:] = self.data_label_train[:, :, 4:] /100
        self.data_label_test = self.data_label_test/100
        self.data_severity_train = self.data_label_train[:, :, 4]
        self.data_severity_test = self.data_label_test.copy

        self.train_input_min = np.min(self.data_input_train)
        self.train_input_max = np.max(self.data_input_train)
        self.data_input_train = (self.data_input_train - self.train_input_min) / (self.train_input_max - self.train_input_min)
        self.data_input_test = (self.data_input_test - self.train_input_min) / (self.train_input_max - self.train_input_min)