import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import mlab as mlab
import numpy as np
from preprocess import import_data

sub_index = 1
cla_index = 1
ori_data, _, ori_label = import_data(sub_index, datatype='T')
class_index = np.argwhere(ori_label == cla_index+1)
ori_data = ori_data[class_index, :, :]
ori_label = ori_label[class_index]
ori_data = np.squeeze(ori_data)

gen_data = []
gen_data0 = np.load('./S' + str(sub_index) + '_class0.npy')
gen_data0 = np.squeeze(gen_data0[0:250, :, :, :])
gen_data1 = np.load('./S' + str(sub_index) + '_class1.npy')
gen_data1 = np.squeeze(gen_data1[0:750, :, :, :])
gen_data2 = np.load('./S' + str(sub_index) + '_class2.npy')
gen_data2 = np.squeeze(gen_data2[0:250, :, :, :])
gen_data3 = np.load('./S' + str(sub_index) + '_class3.npy')
gen_data3 = np.squeeze(gen_data3[0:250, :, :, :])
gen_data.append(gen_data0)
gen_data.append(gen_data1)
gen_data.append(gen_data2)
gen_data.append(gen_data3)

gen_data = np.concatenate(gen_data)

gg_data = locals()['gen_data' + str(cla_index)]
ori_mean = np.mean(ori_data, axis=0)
gg_mean = np.mean(gg_data, axis=0)
ori_std = np.std(ori_data, axis=0)
gg_std = np.std(gg_data, axis=0)

x = np.linspace(0, 4, 1000)
plt.figure(1)
plt.subplot(311)
plt.plot(x, ori_mean[7, :].transpose(), 'darkorange')  # channel C3 Cz C4: 8 10 12
plt.plot(x, gg_mean[7, :].transpose(), 'steelblue')
plt.xlim([0, 4])
plt.ylim([-1, 1])
plt.ylabel('C3')
plt.xticks([])
# plt.plot(x, ori_std[7, :].transpose() + ori_mean[7, :].transpose(), 'darkorange', linestyle='-', alpha=0.8)
# plt.plot(x, gg_std[7, :].transpose() + gg_mean[7, :].transpose(), 'steelblue', linestyle='-', alpha=0.8)

plt.subplot(312)
plt.plot(x, ori_mean[9, :].transpose(), 'darkorange')  # channel C3 Cz C4: 8 10 12
# plt.subplot(212)
plt.plot(x, gg_mean[9, :].transpose(), 'steelblue')
plt.xlim([0, 4])
plt.ylim([-1, 1])
plt.ylabel('Cz')
plt.xticks([])


# plt.plot(x, ori_std[9, :].transpose() + ori_mean[9, :].transpose(), 'darkorange', linestyle='-', alpha=0.8)
# plt.plot(x, gg_std[9, :].transpose() + gg_mean[9, :].transpose(), 'steelblue', linestyle='-', alpha=0.8)

plt.subplot(313)
plt.plot(x, ori_mean[11, :].transpose(), 'darkorange')  # channel C3 Cz C4: 8 10 12
plt.plot(x, gg_mean[11, :].transpose(), 'steelblue')
plt.xlim([0, 4])
plt.ylim([-1, 1])

plt.ylabel('C4')
plt.xlabel('Time (s)')
plt.savefig('./Pictures/Fig5.png', dpi=600)
# plt.plot(x, ori_std[11, :].transpose() + ori_mean[11, :].transpose(), 'darkorange', linestyle='-', alpha=0.8)
# plt.plot(x, gg_std[11, :].transpose() + gg_mean[11, :].transpose(), 'steelblue', linestyle='-', alpha=0.8)

# plt.fill_between(x, ori_mean[9, :].transpose(), ori_std[9, :].transpose(), 'darkorange', alpha=0.5)
# plt.fill_between(x, gg_mean[9, :].transpose(), gg_std[9, :].transpose(), 'steelblue', alpha=0.5)



# plt.show()

oo = np.mean(ori_mean, axis=0)
gg = np.mean(gg_mean, axis=0)
# plt.figure(2)
spectrum0, freqs0, t0 = mlab.specgram(oo, Fs=250)
spectrum1, freqs1, t1 = mlab.specgram(gg, Fs=250)
min_val = 10 * np.log10(max(spectrum0.min(), spectrum1.min()))
max_val = 10 * np.log10(min(spectrum0.max(), spectrum1.max()))
# gs0 = gs.GridSpec(2,2, width_ratios=[10,0.1])

fig, ax = plt.subplots(2, 1)
ax = ax.flatten()
spectrum0, freqs0, t0, im0 = ax[0].specgram(oo, Fs=250, vmin=min_val, vmax=max_val)
spectrum1, freqs1, t1, im1 = ax[1].specgram(gg, Fs=250, vmin=min_val, vmax=max_val)
ax[0].set_xticks([])
ax[0].set_ylim(0, 40)
# ax[0].set_xlim(0, 4)
ax[0].set_ylabel('real')
ax[1].set_ylim(0, 40)
# ax[1].set_xlim(0, 4)
ax[1].set_ylabel('fake')
plt.xlabel('Time (s)')
fig.colorbar(im0, ax=[ax[0], ax[1]])

'''
plt.subplot(211)
plt.specgram(ori_mean[9, :].transpose(), Fs=250)
# plt.specgram(oo, Fs=250)
plt.ylim(0, 40)
plt.ylabel('real')
plt.colorbar()
plt.subplot(212)
plt.specgram(gg_mean[9, :].transpose(), Fs=250)
# plt.specgram(gg, Fs=250)
plt.ylim(0, 40)
plt.ylabel('generated')
plt.colorbar()
'''
plt.savefig('/home/syh/Pictures/Fig6.png', dpi=1200)
plt.show()





dd = ori_mean
# data = gg_mean
dd = np.mean(dd, axis=1)
dd = np.expand_dims(dd, axis=1)

gg = gg_mean
gg = np.mean(gg, axis=1)
gg = np.expand_dims(gg, axis=1)

biosemi_montage = mne.channels.make_standard_montage('biosemi64')
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')
evoked1 = mne.EvokedArray(dd, info)
evoked1.set_montage(biosemi_montage)
evoked2 = mne.EvokedArray(gg, info)
evoked2.set_montage(biosemi_montage)
plt.figure(1)
plt.subplot(121)
mne.viz.plot_topomap(evoked1.data[:, 0], evoked1.info, show=False)
plt.subplot(122)
mne.viz.plot_topomap(evoked2.data[:, 0], evoked2.info, show=False)


print('the end')

