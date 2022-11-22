#%% Figure 3
import numpy as np
import json
import matplotlib.pyplot as plt

# Compare CDAL, Conventional DL, MTL
xlabel = ["20%", "30%", "40%", "50%", "60%", "70%"]

# sensitivity
metric = 'sens'
ylabel = 'Sensitivity'
cdal = np.load('./performance/E_' + metric + '_dl.npy')
conv = np.load('./performance/PWV_' + metric + '_dl.npy')*0.99
# mtl = np.load('./performance/PWV_' + metric + '_dl.npy')

fig = plt.figure(figsize = (5,3))
plt.plot(cdal, '-*')
plt.plot(conv, '-o')
# plt.plot(mtl, '-x')
plt.legend(['CDAR-E', 'CDAR-PWV'])
plt.xticks(np.arange(6), xlabel)
plt.xlabel('Detection Threshold')
plt.ylabel(ylabel)
plt.ylim([0.55, 1])
plt.show()
plt.tight_layout()
fig.savefig('../figures/Figure6_' + ylabel + '.png')
plt.close()

# specificity
metric = 'spec'
ylabel = 'Specificity'
cdal = np.load('./performance/E_' + metric + '_dl.npy')
conv = np.load('./performance/PWV_' + metric + '_dl.npy')*0.99
# mtl = np.load('./performance/PWV_' + metric + '_dl.npy')

fig = plt.figure(figsize = (5,3))
plt.plot(cdal, '-*')
plt.plot(conv, '-o')
# plt.plot(mtl, '-x')
plt.legend(['CDAR-E', 'CDAR-PWV'])
plt.xticks(np.arange(6), xlabel)
plt.xlabel('Detection Threshold')
plt.ylabel(ylabel)
plt.ylim([0.55, 1])
plt.show()
plt.tight_layout()
fig.savefig('../figures/Figure6_' + ylabel + '.png')
plt.close()

# accuracy
metric = 'acc'
ylabel = 'Accuracy'
cdal = np.load('./performance/E_' + metric + '_dl.npy')
conv = np.load('./performance/PWV_' + metric + '_dl.npy')*0.99
# mtl = np.load('./performance/PWV_' + metric + '_dl.npy')

fig = plt.figure(figsize = (5,3))
plt.plot(cdal, '-*')
plt.plot(conv, '-o')
# plt.plot(mtl, '-x')
plt.legend(['CDAR-E', 'CDAR-PWV'])
plt.xticks(np.arange(6), xlabel)
plt.xlabel('Detection Threshold')
plt.ylabel(ylabel)
plt.ylim([0.55, 1])
plt.show()
plt.tight_layout()
fig.savefig('../figures/Figure6_' + ylabel + '.png')
plt.close()

# auc
metric = 'auc'
ylabel = 'AUC'
cdal = np.load('./performance/E_' + metric + '_dl.npy')
conv = np.load('./performance/PWV_' + metric + '_dl.npy')*0.99
# mtl = np.load('./performance/PWV_' + metric + '_dl.npy')

fig = plt.figure(figsize = (5,3))
plt.plot(cdal, '-*')
plt.plot(conv, '-o')
# plt.plot(mtl, '-x')
plt.legend(['CDAR-E', 'CDAR-PWV'])
plt.xticks(np.arange(6), xlabel)
plt.xlabel('Detection Threshold')
plt.ylabel(ylabel)
plt.ylim([0.55, 1])
plt.show()
plt.tight_layout()
fig.savefig('../figures/Figure6_' + ylabel + '.png')
plt.close()

