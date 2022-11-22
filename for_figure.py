#%%
import numpy as np
import matplotlib.pyplot as plt

error = np.arange(5000)/2500
loss_task = np.ones(5000) * 1.5
loss_reg = np.square(error)

fig = plt.figure()
plt.plot(error, loss_reg, label = "$E_{Domain Regressor}$")
plt.plot(error, loss_task, 'k--', label = "$E_{Label Predictor}$")
plt.plot(error, loss_task-0.5*loss_reg, label = "$E_{Feature Extractor}$")
plt.grid()
plt.xlim(0, 2)
plt.ylim(-1,4)
plt.legend()
plt.xlabel('Regression Error ' +r'$(=|p_i-G_\eta(G_f(x_i))|)$')
plt.ylabel('Loss Function')
plt.title('Elementary Loss Function')
plt.savefig('fig2_elementary.png')
# %%
import numpy as np
import matplotlib.pyplot as plt

error = np.arange(5000)/2500
loss_task = np.ones(5000) * 1.5
loss_reg = -np.log(1-np.tanh(np.square(error)))

fig = plt.figure()
plt.plot(error, loss_reg, label = "$E_{Domain Regressor}$")
plt.plot(error, loss_task, 'k--', label = "$E_{Label Predictor}$")
plt.plot(error, loss_task-0.5*np.log(np.tanh(np.square(error))), label = "$E_{Feature Extractor}$")
plt.grid()
plt.xlim(0, 2)
plt.ylim(-1,4)
plt.legend()
plt.xlabel('Regression Error ' +r'$(=|p_i-G_\eta(G_f(x_i))|)$')
plt.ylabel('Loss Function')
plt.title('Stable Loss Function')
plt.savefig('fig2_stable.png')


# %%
