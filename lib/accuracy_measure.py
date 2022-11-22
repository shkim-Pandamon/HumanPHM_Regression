import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class AccuracyMeasure(object):
    def __init__(self):
        pass

    def measures(self, x_test, y_test, r_test, R_loss_test, x_ref, save_path):
        ylim = np.max(y_test)
        r2score = r2_score(y_test.flatten(), r_test.flatten())
        r_mean = np.mean(r_test, axis = 0)
        r_std = np.std(r_test, axis = 0)
        R_loss_test = np.sqrt(R_loss_test/int(x_test.shape[0]))
        r_xaxis = np.arange(x_test.shape[1])
        r_xaxis_poly = np.concatenate((r_xaxis, np.flip(r_xaxis)), axis = 0)
        x_sv = np.unique(x_ref)
        y_sv = np.ones(x_sv.shape[0])
        fig = plt.figure(figsize = (8, 5))
        
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
        fig.savefig(save_path)
        plt.close(fig)