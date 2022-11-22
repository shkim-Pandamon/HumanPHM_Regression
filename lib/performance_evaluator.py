import numpy as np
from sklearn.metrics import r2_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

class PerformanceEvaluator(object):
    def __init__(self, test_label, test_prediction, test_regession_loss):
        self.test_label = test_label
        self.test_prediction = test_prediction
        self.test_regession_loss = test_regession_loss

    def measures_accuracy(self, save_path):
        ylim = np.max(self.test_label)
        r2score = r2_score(self.test_label.flatten(), self.test_prediction.flatten())
        r_mean = np.mean(self.test_prediction, axis = 0)
        r_std = np.std(self.test_prediction, axis = 0)
        R_loss_test = np.sqrt(R_loss_test/int(self.test_label.shape[0]))
        r_xaxis = np.arange(self.test_label.shape[1])
        r_xaxis_poly = np.concatenate((r_xaxis, np.flip(r_xaxis)), axis = 0)
        r_poly1 = np.concatenate((r_mean + 1.96 * r_std, np.flip(r_mean - 1.96 * r_std)), axis = 0)
        
        fig = plt.figure(figsize = (8, 5))
        plt.plot(r_xaxis, r_mean * 100, 'r', linewidth = 3)
        plt.fill(r_xaxis_poly, r_poly1 * 100, alpha = 0.4, color = 'r')
        plt.plot([-100, 100], [-100, 100], 'k--', linewidth = 3)
        plt.legend(('Mean of estimation', 'Ideal estimation', '95% Confidence interval'), fontsize = 15, loc = 'lower right')
        plt.xlim([0, ylim * 100])
        plt.ylim([0, ylim * 100 + 10])
        plt.title('MSE: ' + str(np.round_(R_loss_test, 3)) + ',  R2: ' + str(np.round_(r2score, 3)))
        plt.xlabel('Real Severity (Inclusion [%])', fontsize = 20)
        plt.ylabel('Estimated Severity (DL)', fontsize = 20)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        

    def draws_BA_plot(self, save_path, sd_limit=1.96):
        randint = np.random.randint(0, self.test_label.shape[0], 16)
        fig = plt.figure(figsize = (8, 5))
        m1 = np.reshape(self.test_label[randint], -1) * 100
        m2 = np.reshape(self.test_prediction[randint], -1) * 100
        diffs = m1 - m2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, axis=0)

        scatter_kwds=None
        mean_line_kwds=None
        limit_lines_kwds=None

        ax = plt.gca()

        scatter_kwds = scatter_kwds or {}
        if 's' not in scatter_kwds:
            scatter_kwds['s'] = 20
        mean_line_kwds = mean_line_kwds or {}
        limit_lines_kwds = limit_lines_kwds or {}
        for kwds in [mean_line_kwds, limit_lines_kwds]:
            if 'color' not in kwds:
                kwds['color'] = 'gray'
            if 'linewidth' not in kwds:
                kwds['linewidth'] = 2
        if 'linestyle' not in mean_line_kwds:
            kwds['linestyle'] = '--'
        if 'linestyle' not in limit_lines_kwds:
            kwds['linestyle'] = ':'

        ax.scatter(m1, diffs, **scatter_kwds)
        ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

        # Annotate mean line with mean difference.
        ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                    xy=(0.99, 0.5),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=14,
                    xycoords='axes fraction')

        if sd_limit > 0:
            half_ylim = (1.5 * sd_limit) * std_diff
            ax.set_ylim(mean_diff - half_ylim,
                        mean_diff + half_ylim)

            limit_of_agreement = sd_limit * std_diff
            lower = mean_diff - limit_of_agreement
            upper = mean_diff + limit_of_agreement
            for j, lim in enumerate([lower, upper]):
                ax.axhline(lim, **limit_lines_kwds)
            ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                        xy=(0.99, 0.40),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        fontsize=14,
                        xycoords='axes fraction')
            ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                        xy=(0.99, 0.56),
                        horizontalalignment='right',
                        fontsize=14,
                        xycoords='axes fraction')

        elif sd_limit == 0:
            half_ylim = 3 * std_diff
            ax.set_ylim(mean_diff - half_ylim,
                        mean_diff + half_ylim)
        ax.set_ylim([-55, 55])
        ax.set_ylabel('Difference [%]', fontsize=20)
        ax.set_xlabel('True PAD Severity [%]', fontsize=20)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
    def evaluates_performance(self, result_dir, model_name, regressor_type):
        thresholds = np.array([20, 30, 40, 50, 60, 70])
        auc_dl = np.zeros((thresholds.shape[0]))
        acc_dl = np.zeros((thresholds.shape[0]))
        sens_dl = np.zeros((thresholds.shape[0]))
        spec_dl = np.zeros((thresholds.shape[0]))

        fig = plt.figure(figsize = (8, 5))               
        plt.title('Receiver Operationg Characteristic', fontsize = 20)
        plt.xlabel('1 - Specificity', fontsize = 20)
        plt.ylabel('Sensitivity', fontsize = 20)    
        alphas = [0.2, 0.5, 1.0]
        alpha_n = 0
        for kk in range(thresholds.shape[0]):
            threshold = thresholds[kk]
            y_real = self.test_label.copy()
            y_pred = self.test_prediction.copy()
            y_real[:, :threshold] = 0
            y_real[:, threshold:] = 1
            y_real_0 = y_real[:, :threshold]
            y_real_1 = y_real[:, threshold:]
            y_pred_0 = y_pred[:, :threshold]
            y_pred_1 = y_pred[:, threshold:]
            
            #ROC and ACU
            y_real = np.reshape(y_real, -1)
            y_pred = np.reshape(y_pred, -1)

            false_positive_rate_DL, true_positive_rate_DL, _ = roc_curve(y_real, y_pred)
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
        fig.savefig("{}/{}_roc.png".format(result_dir, model_name))
        plt.close(fig)

        np.save("{}/{}_auc_dl.npy".format(result_dir, model_name, regressor_type), auc_dl)
        np.save("{}/{}_accuracy_dl.npy".format(result_dir, model_name, regressor_type), acc_dl)
        np.save("{}/{}_sensitivity_dl.npy".format(result_dir, model_name, regressor_type), sens_dl)
        np.save("{}/{}_specificity_dl.npy".format(result_dir, model_name, regressor_type), spec_dl)
