import mne
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from preprocess import import_data

sub_index = 1
cla_index = 1

ori_data, _, ori_label = import_data(sub_index, datatype='T')
class_index = np.argwhere(ori_label == cla_index+1)
ori_data = ori_data[class_index, :, :]
ori_label = ori_label[class_index]
ori_data = np.squeeze(ori_data)
ori_cov = []
gen_cov = []
for cov_i in range(len(ori_data)):
    one = ori_data[cov_i, :, :]
    oneone = np.dot(one, one.transpose())
    ori_cov.append(oneone / np.trace(oneone))
ori_cov_mean = np.mean(ori_cov, axis=0)
gen_data = []
for cla_index in range(4):

    gg = np.load('./S' + str(sub_index) +
                    '_class' + str(cla_index) + '.npy')
    gen_data.append(gg[0:750, :, :, :])
gen_data = np.load('./S1_class' + str(cla_index) + '.npy')

gen_data = np.concatenate(gen_data)
gen_data = gen_data[0:750, :, :]
for cov_j in range(len(gen_data)):
    two = gen_data[cov_j, :, :]
    twotwo = np.dot(two, two.transpose())
    gen_cov.append(twotwo / np.trace(twotwo))
gen_cov_mean = np.mean(gen_cov, axis=0)



# 这里是创建一个数据
# column = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
#            '16', '17', '18', '19', '20', '21', '22']
# row = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
#           '16', '17', '18', '19', '20', '21', '22']
column = ['1', '', '', '', '', '', '', '', '', '', '11', '', '', '', '',
          '', '', '', '', '', '', '22']
row = ['1', '', '', '', '', '', '', '', '', '', '11', '', '', '', '',
          '', '', '', '', '', '', '22']

po = ori_cov_mean
pg = gen_cov_mean

po = (po - np.min(po)) / (np.max(po) - np.min(po))
pg = (pg - np.min(pg)) / (np.max(pg) - np.min(pg))

minmin = np.min([np.min(po), np.min(pg)])
maxmax = np.max([np.max(po), np.max(pg)])

fig, ax = plt.subplots(1, 2)
ax = ax.flatten()
im0 = ax[0].imshow(po, cmap='RdBu')
# fig.colorbar(im0)
im1 = ax[1].imshow(pg, cmap='RdBu')

ax[0].set_xticks(np.arange(len(row)))
ax[0].set_yticks(np.arange(len(column)))
ax[1].set_xticks(np.arange(len(row)))
ax[1].set_yticks(np.arange(len(column)))

ax[0].set_xticklabels(row, fontsize=8)
ax[0].set_yticklabels(column, fontsize=8)
ax[1].set_xticklabels(row, fontsize=8)
ax[1].set_yticklabels(column, fontsize=8)

ax[0].set_xlabel('real', size=12)
ax[1].set_xlabel('fake', size=12)

# plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")



# fig.tight_layout()
# levels = np.arange(0.015, 0.05)

fig.colorbar(im1, ax=[ax[0], ax[1]], fraction=0.021, pad=0.06)
# fig.colorbar.set_ticks([0, 1])
plt.savefig('./Pictures/test.png', dpi=3600)
plt.show()



print('the end')

