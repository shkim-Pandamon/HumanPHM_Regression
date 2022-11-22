import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

class Performance(object):
    def __init__(self):
        pass

    def measures(self, y_real, y_pred, roc_path, performance_path):
        thresholds = np.array([20, 30, 40, 50, 60, 70])
        auc_dl = np.zeros((thresholds.shape[0]))
        acc_dl = np.zeros((thresholds.shape[0]))
        sens_dl = np.zeros((thresholds.shape[0]))
        spec_dl = np.zeros((thresholds.shape[0]))

        y_real_g = y_real
        y_pred_g = y_pred

        fig = plt.figure(figsize = (8, 5))               
        plt.title('Receiver Operationg Characteristic', fontsize = 20)
        plt.xlabel('1 - Specificity', fontsize = 20)
        plt.ylabel('Sensitivity', fontsize = 20)    
        alphas = [0.2, 0.5, 1.0]
        alpha_n = 0
        for kk in range(thresholds.shape[0]):
            threshold = thresholds[kk]
            y_real = y_real_g
            y_pred = y_pred_g
            y_real[:, :threshold] = 0
            y_real[:, threshold:] = 1
            y_real_0 = y_real[:, :threshold]
            y_real_1 = y_real[:, threshold:]
            y_pred_0 = y_pred[:, :threshold]
            y_pred_1 = y_pred[:, threshold:]
            
            #ROC and ACU
            y_real = np.reshape(y_real, -1)
            y_pred = np.reshape(y_pred, -1)

            false_positive_rate_DL, true_positive_rate_DL, thresholds_DL = roc_curve(y_real, y_pred)
            roc_auc_DL = auc(false_positive_rate_DL, true_positive_rate_DL)
            if threshold in [20, 50, 70]:
                plt.plot(false_positive_rate_DL, true_positive_rate_DL, 'b', alpha = alphas[alpha_n], label='Model DL (AUC = %0.2f) - '%roc_auc_DL + 'threshold = ' + str(threshold))
                alpha_n = alpha_n + 1
            auc_dl[kk] = roc_auc_DL
            
            #Accuracy
            y_real_0 = np.reshape(y_real_0, -1)
            y_real_1 = np.reshape(y_real_1, -1)
            y_pred_0 = np.reshape(y_pred_0, -1)
            y_pred_1 = np.reshape(y_pred_1, -1)
            rndidx_0 = np.random.randint(y_real_0.shape[0], size = 1000)
            rndidx_1 = np.random.randint(y_real_1.shape[0], size = 1000)
            y_real_0 = y_real_0[rndidx_0]
            y_real_1 = y_real_1[rndidx_1]
            y_pred_0 = y_pred_0[rndidx_0]
            y_pred_1 = y_pred_1[rndidx_1]
            y_real = np.concatenate((y_real_0, y_real_1))
            y_pred = np.concatenate((y_pred_0, y_pred_1))
            for ii in range(y_pred.shape[0]):
                if y_pred[ii] > threshold/100:
                    y_pred[ii] = 1
                else:
                    y_pred[ii] = 0
            cm_dl = confusion_matrix(y_real, y_pred)
            sens_dl[kk] = (cm_dl[0, 0])/(cm_dl[0, 0] + cm_dl[1, 0])
            spec_dl[kk] = (cm_dl[1, 1])/(cm_dl[0, 1] + cm_dl[1, 1])
            acc_dl[kk] = (cm_dl[0, 0] + cm_dl[1, 1])/(cm_dl[0, 0] + cm_dl[0, 1] + cm_dl[1, 0] + cm_dl[1, 1])
        plt.plot([0,1],[1,1], 'y--')
        plt.plot([0,1],[0,1], 'r--')    
        plt.legend(loc = 'lower right', fontsize = 10)
        plt.show()
        fig.savefig(roc_path)
        plt.close(fig)

        np.save(performance_path + '_auc_dl.npy', auc_dl)
        np.save(performance_path + '_acc_dl.npy', acc_dl)
        np.save(performance_path + '_sens_dl.npy', sens_dl)
        np.save(performance_path + '_spec_dl.npy', spec_dl)

        print('sensitivity')
        print(sens_dl)
        print('specificity')
        print(spec_dl)
        print('acc')
        print(acc_dl)
        print('auc')
        print(auc_dl)
