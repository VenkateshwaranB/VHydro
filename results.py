import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import osgeo
import re
import itertools

from matplotlib.patches import Rectangle
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns

import os

from matplotlib.colors import LinearSegmentedColormap
from itertools import groupby,count


# the base_directory property tells us where it was downloaded to:
cls_name = 5
node5 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_5 = pd.read_csv(node5)

cls_name = 6
node6 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_6 = pd.read_csv(node6)

cls_name = 7
node7 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_7 = pd.read_csv(node7)

cls_name = 8
node8 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_8 = pd.read_csv(node8)

cls_name = 9
node9 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_9 = pd.read_csv(node9)

cls_name = 10
node10 = os.path.join("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/", "mckee-"+str(well_name)+" BF node for "+str(cls_name)+".csv")
node_10 = pd.read_csv(node10)




cls_name = 5
history1_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")

cls_name = 6
history1_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")

cls_name = 7
history1_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_7 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")

cls_name = 8
history1_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_8 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")

cls_name = 9
history1_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_9 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")

cls_name = 10
history1_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+".xlsx")
results1_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")

history2_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new2.xlsx")
results2_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new2.xlsx")

history3_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new3.xlsx")
results3_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new3.xlsx")

history4_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/History_"+str(cls_name)+"new4.xlsx")
results4_10 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/"+str(cls_name)+"/results/Results_"+str(cls_name)+" new4.xlsx")


print("True labels cluster 5")
print("-------------------------")
print(results1_5['True'].value_counts().to_frame())
print("#########################")
print("#########################")
print("4 cluster 5")
print("-------------------------")
print(results4_5['Predicted'].value_counts().to_frame())
print("#########################")
print("2 cluster 6")
print("-------------------------")
print(results2_6['Predicted'].value_counts().to_frame())
print("#########################")
print("2 cluster 7")
print("-------------------------")
print(results2_7['Predicted'].value_counts().to_frame())
print("#########################")
print("4 cluster 8")
print("-------------------------")
print(results4_8['Predicted'].value_counts().to_frame())
print("#########################")
print("2 cluster 9")
print("-------------------------")
print(results2_9['Predicted'].value_counts().to_frame())
print("#########################")
print("2 cluster 10")
print("-------------------------")
print(results1_10['Predicted'].value_counts().to_frame())


x1_5 = np.arange(len(history1_5['loss']))
x2_5 = np.arange(len(history2_5['loss']))
x3_5 = np.arange(len(history3_5['loss']))
x4_5 = np.arange(len(history4_5['loss']))

x1_6 = np.arange(len(history1_6['loss']))
x2_6 = np.arange(len(history2_6['loss']))
x3_6 = np.arange(len(history3_6['loss']))
x4_6 = np.arange(len(history4_6['loss']))

x1_7 = np.arange(len(history1_7['loss']))
x2_7 = np.arange(len(history2_7['loss']))
x3_7 = np.arange(len(history3_7['loss']))
x4_7 = np.arange(len(history4_7['loss']))

x1_8 = np.arange(len(history1_8['loss']))
x2_8 = np.arange(len(history2_8['loss']))
x3_8 = np.arange(len(history3_8['loss']))
x4_8 = np.arange(len(history4_8['loss']))

x1_9 = np.arange(len(history1_9['loss']))
x2_9 = np.arange(len(history2_9['loss']))
x3_9 = np.arange(len(history3_9['loss']))
x4_9 = np.arange(len(history4_9['loss']))

x1_10 = np.arange(len(history1_10['loss']))
x2_10 = np.arange(len(history2_10['loss']))
x3_10 = np.arange(len(history3_10['loss']))
x4_10 = np.arange(len(history4_10['loss']))


x_5 = ''
x_6 = ''
x_7 = ''
x_8 = ''
x_9 = ''
x_10 = ''

history_5 = ''
history_6 = ''
history_7 = ''
history_8 = ''
history_9 = ''
history_10 = ''

# mckee-16a :  5 - 3, 6 - 3, 7 - 3, 8 - 3, 9 - 2, 10 - 1
if well_name == "16a":
  x_5 = x1_5
  x_6 = x1_6
  x_7 = x3_7
  x_8 = x2_8
  x_9 = x3_9
  x_10 = x1_10

  history_5 = history1_5
  history_6 = history1_6
  history_7 = history3_7
  history_8 = history2_8
  history_9 = history3_9
  history_10 = history1_10
elif well_name == "4":
  x_5 = x3_5
  x_6 = x3_6
  x_7 = x4_7
  x_8 = x2_8
  x_9 = x2_9
  x_10 = x1_10

  history_5 = history3_5
  history_6 = history3_6
  history_7 = history4_7
  history_8 = history2_8
  history_9 = history2_9
  history_10 = history1_10
elif well_name == "5a":
  x_5 = x4_5
  x_6 = x2_6
  x_7 = x2_7
  x_8 = x4_8
  x_9 = x2_9
  x_10 = x2_10

  history_5 = history4_5
  history_6 = history2_6
  history_7 = history2_7
  history_8 = history4_8
  history_9 = history2_9
  history_10 = history2_10


  fig, axes = plt.subplots(2, 3, figsize=(30, 18))
axes[0, 0].plot(x_5, history_5['loss'], 'b-', label='train', linewidth=sx)
axes[0, 0].plot(x_5, history_5['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 0].fill_between(x_5, history_5['loss'], history_5['val_loss'], color='b', alpha=0.2)
axes[0, 0].legend(fontsize=20, title='Cluster 5', title_fontsize=20)
axes[0, 0].set_title("Model Loss", fontsize=20)
axes[0, 0].set_xlabel("Epoch", fontsize=20)
axes[0, 0].set_ylabel('Loss', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

axes[0, 1].plot(x_6, history_6['loss'], 'b-', label='train', linewidth=sx)
axes[0, 1].plot(x_6, history_6['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 1].fill_between(x_6, history_6['loss'], history_6['val_loss'], color='b', alpha=0.2)
axes[0, 1].legend(title='Cluster 6', fontsize=20, title_fontsize=20)
axes[0, 1].set_title("Model Loss", fontsize=20)
axes[0, 1].set_xlabel("Epoch", fontsize=20)
axes[0, 1].set_ylabel('Loss', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

axes[0, 2].plot(x_7, history_7['loss'], 'b-', label='train', linewidth=sx)
axes[0, 2].plot(x_7, history_7['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 2].fill_between(x_7, history_7['loss'], history_7['val_loss'], color='b', alpha=0.2)
axes[0, 2].legend(title='Cluster 7', fontsize=20, title_fontsize=20)
axes[0, 2].set_title("Model Loss", fontsize=20)
axes[0, 2].set_xlabel("Epoch", fontsize=20)
axes[0, 2].set_ylabel('Loss', fontsize=20)
axes[0, 2].tick_params(axis='both', labelsize=15)

axes[1, 0].plot(x_5, history_5['acc'], 'b-', label='train', linewidth=sx)
axes[1, 0].plot(x_5, history_5['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 0].fill_between(x_5, history_5['acc'], history_5['val_acc'], color='b', alpha=0.2)
axes[1, 0].legend(title='Cluster 5', fontsize=20, title_fontsize=20)
axes[1, 0].set_title("Model Accuracy", fontsize=20)
axes[1, 0].set_xlabel("Epoch", fontsize=20)
axes[1, 0].set_ylabel('Accuracy', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

axes[1, 1].plot(x_6, history_6['acc'], 'b-', label='train', linewidth=sx)
axes[1, 1].plot(x_6, history_6['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 1].fill_between(x_6, history_6['acc'], history_6['val_acc'], color='b', alpha=0.2)
axes[1, 1].legend(title='Cluster 6', fontsize=20, title_fontsize=20)
axes[1, 1].set_title("Model Accuracy", fontsize=20)
axes[1, 1].set_xlabel("Epoch", fontsize=20)
axes[1, 1].set_ylabel('Accuracy', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

axes[1, 2].plot(x_7, history_7['acc'], 'b-', label='train', linewidth=sx)
axes[1, 2].plot(x_7, history_7['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 2].fill_between(x_7, history_7['acc'], history_7['val_acc'], color='b', alpha=0.2)
axes[1, 2].legend(title='Cluster 7', fontsize=20, title_fontsize=20)
axes[1, 2].set_title("Model Accuracy", fontsize=20)
axes[1, 2].set_xlabel("Epoch", fontsize=20)
axes[1, 2].set_ylabel('Accuracy', fontsize=20)
axes[1, 2].tick_params(axis='both', labelsize=15)

plt.show()
#fig.savefig("/content/drive/MyDrive/paper final/GCN datasets/Loss and accuracy plot for 5,6,7_new.jpg")
fig.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/Loss and accuracy plot for 5,6,7.jpg")


fig, axes = plt.subplots(2, 3, figsize=(30, 18))
axes[0, 0].plot(x_8, history_8['loss'], 'b-', label='train', linewidth=sx)
axes[0, 0].plot(x_8, history_8['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 0].fill_between(x_8, history_8['loss'], history_8['val_loss'], color='b', alpha=0.2)
axes[0, 0].legend(fontsize=20, title='Cluster 8', title_fontsize=20)
axes[0, 0].set_title("Model Loss", fontsize=20)
axes[0, 0].set_xlabel("Epoch", fontsize=20)
axes[0, 0].set_ylabel('Loss', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

axes[0, 1].plot(x_9, history_9['loss'], 'b-', label='train', linewidth=sx)
axes[0, 1].plot(x_9, history_9['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 1].fill_between(x_9, history_9['loss'], history_9['val_loss'], color='b', alpha=0.2)
axes[0, 1].legend(title='Cluster 9', fontsize=20, title_fontsize=20)
axes[0, 1].set_title("Model Loss", fontsize=20)
axes[0, 1].set_xlabel("Epoch", fontsize=20)
axes[0, 1].set_ylabel('Loss', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

axes[0, 2].plot(x_10, history_10['loss'], 'b-', label='train', linewidth=sx)
axes[0, 2].plot(x_10, history_10['val_loss'], 'r', label='validation', linewidth=sx)
axes[0, 2].fill_between(x_10, history_10['loss'], history_10['val_loss'], color='b', alpha=0.2)
axes[0, 2].legend(title='Cluster 10', fontsize=20, title_fontsize=20)
axes[0, 2].set_title("Model Loss", fontsize=20)
axes[0, 2].set_xlabel("Epoch", fontsize=20)
axes[0, 2].set_ylabel('Loss', fontsize=20)
axes[0, 2].tick_params(axis='both', labelsize=15)

axes[1, 0].plot(x_8, history_8['acc'], 'b-', label='train', linewidth=sx)
axes[1, 0].plot(x_8, history_8['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 0].fill_between(x_8, history_8['acc'], history_8['val_acc'], color='b', alpha=0.2)
axes[1, 0].legend(title='Cluster 8', fontsize=20, title_fontsize=20)
axes[1, 0].set_title("Model Accuracy", fontsize=20)
axes[1, 0].set_xlabel("Epoch", fontsize=20)
axes[1, 0].set_ylabel('Accuracy', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

axes[1, 1].plot(x_9, history_9['acc'], 'b-', label='train', linewidth=sx)
axes[1, 1].plot(x_9, history_9['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 1].fill_between(x_9, history_9['acc'], history_9['val_acc'], color='b', alpha=0.2)
axes[1, 1].legend(title='Cluster 9', fontsize=20, title_fontsize=20)
axes[1, 1].set_title("Model Accuracy", fontsize=20)
axes[1, 1].set_xlabel("Epoch", fontsize=20)
axes[1, 1].set_ylabel('Accuracy', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

axes[1, 2].plot(x_10, history_10['acc'], 'b-', label='train', linewidth=sx)
axes[1, 2].plot(x_10, history_10['val_acc'], 'r', label='validation', linewidth=sx)
axes[1, 2].fill_between(x_10, history_10['acc'], history_10['val_acc'], color='b', alpha=0.2)
axes[1, 2].legend(title='Cluster 10', fontsize=20, title_fontsize=20)
axes[1, 2].set_title("Model Accuracy", fontsize=20)
axes[1, 2].set_xlabel("Epoch", fontsize=20)
axes[1, 2].set_ylabel('Accuracy', fontsize=20)
axes[1, 2].tick_params(axis='both', labelsize=15)

plt.show()
#fig.savefig("/content/drive/MyDrive/paper final/GCN datasets/Loss and accuracy plot for 8,9,10_new.jpg")
fig.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/Loss and accuracy plot for 8,9,10.jpg")


def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value


well1 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/5/mckee-"+str(well_name)+" facies for 5.xlsx")
well2 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/6/mckee-"+str(well_name)+" facies for 6.xlsx")
well3 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/7/mckee-"+str(well_name)+" facies for 7.xlsx")
well4 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/8/mckee-"+str(well_name)+" facies for 8.xlsx")
well5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/9/mckee-"+str(well_name)+" facies for 9.xlsx")
well6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/10/mckee-"+str(well_name)+" facies for 10.xlsx")

f_colors = ['#FF5757', '#B89393', '#FFBD59','#5271FF','#4FB47A']

facies_colors = ['gold', 'orange', 'chocolate','black','slateblue','mediumorchid', 'cornflowerblue', 'deepskyblue','green']

def plot_colortable(colors, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(colors.rgb_to_hsv(colors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

if well_name == "16a":
  botVal = 587
elif well_name == "5a":
  botVal = 4216
elif well_name == "4":
  botVal = 5801

well1 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/5/mckee-"+str(well_name)+" facies for 5.xlsx")
mckee1 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/5/results/McKee-"+str(well_name)+" results.xlsx")


mckee1['True_l5'] = ''
for j in range(len(mckee1)):
    if mckee1[True][j] == 'Very_High':
      mckee1['True_l5'][j] = int(4)
    elif mckee1[True][j] == 'High':
      mckee1['True_l5'][j] = int(3)
    elif mckee1[True][j] == 'Moderate':
      mckee1['True_l5'][j] = int(2)
    elif mckee1[True][j] == 'Low':
      mckee1['True_l5'][j] = int(1)
    elif mckee1[True][j] == 'Very_Low':
      mckee1['True_l5'][j] = int(0)
    else:
      print(j)

mckee1['Pred_l5'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_5'][j] == 'Very_High':
      mckee1['Pred_l5'][j] = 4
    elif mckee1['Predicted_5'][j] == 'High':
      mckee1['Pred_l5'][j] = 3
    elif mckee1['Predicted_5'][j] == 'Moderate':
      mckee1['Pred_l5'][j] = 2
    elif mckee1['Predicted_5'][j] == 'Low':
      mckee1['Pred_l5'][j] = 1
    elif mckee1['Predicted_5'][j] == 'Very_Low':
      mckee1['Pred_l5'][j] = 0
    else:
      print(j)

mckee1['Pred_l6'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_6'][j] == 'Very_High':
      mckee1['Pred_l6'][j] = 4
    elif mckee1['Predicted_6'][j] == 'High':
      mckee1['Pred_l6'][j] = 3
    elif mckee1['Predicted_6'][j] == 'Moderate':
      mckee1['Pred_l6'][j] = 2
    elif mckee1['Predicted_6'][j] == 'Low':
      mckee1['Pred_l6'][j] = 1
    elif mckee1['Predicted_6'][j] == 'Very_Low':
      mckee1['Pred_l6'][j] = 0
    else:
      print(j)

mckee1['Pred_l7'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_7'][j] == 'Very_High':
      mckee1['Pred_l7'][j] = 4
    elif mckee1['Predicted_7'][j] == 'High':
      mckee1['Pred_l7'][j] = 3
    elif mckee1['Predicted_7'][j] == 'Moderate':
      mckee1['Pred_l7'][j] = 2
    elif mckee1['Predicted_7'][j] == 'Low':
      mckee1['Pred_l7'][j] = 1
    elif mckee1['Predicted_7'][j] == 'Very_Low':
      mckee1['Pred_l7'][j] = 0
    else:
      print(j)

mckee1['Pred_l8'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_8'][j] == 'Very_High':
      mckee1['Pred_l8'][j] = 4
    elif mckee1['Predicted_8'][j] == 'High':
      mckee1['Pred_l8'][j] = 3
    elif mckee1['Predicted_8'][j] == 'Moderate':
      mckee1['Pred_l8'][j] = 2
    elif mckee1['Predicted_8'][j] == 'Low':
      mckee1['Pred_l8'][j] = 1
    elif mckee1['Predicted_8'][j] == 'Very_Low':
      mckee1['Pred_l8'][j] = 0
    else:
      print(j)

mckee1['Pred_l9'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_9'][j] == 'Very_High':
      mckee1['Pred_l9'][j] = 4
    elif mckee1['Predicted_9'][j] == 'High':
      mckee1['Pred_l9'][j] = 3
    elif mckee1['Predicted_9'][j] == 'Moderate':
      mckee1['Pred_l9'][j] = 2
    elif mckee1['Predicted_9'][j] == 'Low':
      mckee1['Pred_l9'][j] = 1
    elif mckee1['Predicted_9'][j] == 'Very_Low':
      mckee1['Pred_l9'][j] = 0
    else:
      print(j)

mckee1['Pred_l10'] = ''
for j in range(len(mckee1)):
    if mckee1['Predicted_10'][j] == 'Very_High':
      mckee1['Pred_l10'][j] = 4
    elif mckee1['Predicted_10'][j] == 'High':
      mckee1['Pred_l10'][j] = 3
    elif mckee1['Predicted_10'][j] == 'Moderate':
      mckee1['Pred_l10'][j] = 2
    elif mckee1['Predicted_10'][j] == 'Low':
      mckee1['Pred_l10'][j] = 1
    elif mckee1['Predicted_10'][j] == 'Very_Low':
      mckee1['Pred_l10'][j] = 0
    else:
      print(j)


# Display logs with facies
logs = mckee1.columns[8:]
rows,cols = 1,7
fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6), sharey=True)

plt.suptitle('WELL F02-1', size=15)
for i in range(cols):
  F = np.vstack((mckee1[logs[i]].to_list(),mckee1[logs[i]].to_list())).T
  im = ax[i].imshow(F, aspect='auto', extent=[0,1,max(well1.DEPTH), min(well1.DEPTH)])
  ax[i].set_title('FACIES')
fig.colorbar(im, ticks=range(8), orientation="vertical")

#pip install sctriangulate
from sctriangulate.colors import build_custom_continuous_cmap

new_cmap = build_custom_continuous_cmap([255,0,0],[255,69,0],[0,0,255],[50,205,50],[0,100,0])

if well_name == '16a':
  mckee1['Pred_l5'][586] = 4
  mckee1['Pred_l6'][586] = 4
  mckee1['Pred_l7'][586] = 4
  mckee1['Pred_l8'][586] = 4
  mckee1['Pred_l9'][586] = 4
  mckee1['Pred_l10'][586] = 4
elif well_name =='4':
  mckee1['Pred_l9'][5800] = 4
elif well_name =='5a':
  mckee1['Pred_l5'][4215] = 4
  mckee1['Pred_l6'][4215] = 4
  mckee1['Pred_l7'][4215] = 4
  mckee1['Pred_l10'][4215] = 4


new_cmap = build_custom_continuous_cmap([255,0,0],[255,69,0],[0,0,255],[50,205,50],[0,100,0])


# Display logs with facies
logs = mckee1.columns[8:]
rows,cols = 1,7
fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(24,28), sharey=True)

plt.suptitle('Mckee-4', size=25)
for i in range(cols):
  #cmap = mpl.cm.get_cmap('my_list', colors)
  F = np.vstack((mckee1[logs[i]].to_list(),mckee1[logs[i]].to_list())).T
  im = ax[i].imshow(F, aspect='auto',cmap=new_cmap, extent=[0,1,max(well1.DEPTH), min(well1.DEPTH)])
  if i == 0:
    ax[i].set_title('True', fontsize=25)
  else:
    ax[i].set_title('Predicted '+ str(i + 4), fontsize=25)
  ax[i].tick_params(axis='y', labelsize=25)
  ax[i].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

#fig.colorbar(im, ticks=range(8), orientation="vertical")
#fig.colorbar(im, ticks=range(8), orientation="vertical")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_results_plot.jpg")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_results_plot.png")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_results_plot.pdf")


# this one used for create a facies map for all the cluster
well1 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/5/mckee-"+str(well_name)+" facies for 5.xlsx")
well2 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/6/mckee-"+str(well_name)+" facies for 6.xlsx")
well3 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/7/mckee-"+str(well_name)+" facies for 7.xlsx")
well4 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/8/mckee-"+str(well_name)+" facies for 8.xlsx")
well5 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/9/mckee-"+str(well_name)+" facies for 9.xlsx")
well6 = pd.read_excel("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/10/mckee-"+str(well_name)+" facies for 10.xlsx")

plotm = pd.DataFrame()
plotm['DEPTH'] = well1['DEPTH']
if well_name == '16a':
  plotm['Facies_5'] = well1['Facies_pred']
  plotm['Facies_6'] = well2['Facies_pred']
  plotm['Facies_7'] = well3['Facies_pred']
  plotm['Facies_8'] = well4['Facies_pred']
  plotm['Facies_9'] = well5['Facies_pred']
  plotm['Facies_10'] = well6['Facies_pred']
  plotm['Facies_5'][586] = 9
  plotm['Facies_6'][586] = 9
  plotm['Facies_7'][586] = 9
  plotm['Facies_8'][586] = 9
  plotm['Facies_9'][586] = 9
elif well_name == '5a':
  plotm['Facies_5'] = well1['Facies_pred']
  plotm['Facies_6'] = well2['Facies_pred']
  plotm['Facies_7'] = well3['Facies_pred']
  plotm['Facies_8'] = well4['Facies_pred']
  plotm['Facies_9'] = well5['Facies_pred']
  plotm['Facies_10'] = well6['Facies_pred']
  plotm['Facies_5'][4215] = 9
  plotm['Facies_6'][4215] = 9
  plotm['Facies_7'][4215] = 9
  plotm['Facies_8'][4215] = 9
  plotm['Facies_9'][4215] = 9
elif well_name == '4':
  plotm['Facies_5'] = well1['Facies_pred']
  plotm['Facies_6'] = well2['Facies_pred']
  plotm['Facies_7'] = well3['Facies_pred']
  plotm['Facies_8'] = well4['Facies_pred']
  plotm['Facies_9'] = well5['Facies_pred']
  plotm['Facies_10'] = well6['Facies_pred']
  plotm['Facies_5'][5800] = 9
  plotm['Facies_6'][5800] = 9
  plotm['Facies_7'][5800] = 9
  plotm['Facies_8'][5800] = 9
  plotm['Facies_9'][5800] = 9

# Display logs with facies
logs = plotm.columns[1:]
rows,cols = 1,6
fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(24,28), sharey=True)

plt.suptitle('WELL F02-1', size=15)

cma = 'viridis'
for i in range(cols):
  print(i)
  if i == 0:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap, extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 5', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  elif i == 1:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap,extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 6', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  elif i == 2:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap, extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 7', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  elif i == 3:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap, extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 8', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  elif i == 4:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap, extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 9', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  elif i == 5:
    cmap = mpl.cm.get_cmap(cma)
    F = np.vstack((plotm[logs[i]].to_list(),plotm[logs[i]].to_list())).T
    im = ax[i].imshow(F, aspect='auto', cmap=cmap, extent=[0,1,max(plotm.DEPTH), min(plotm.DEPTH)])
    ax[i].set_title('Cluster 10', fontsize=25)
    ax[i].tick_params(axis='y', labelsize=25)
    ax[i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)

#fig.colorbar(im, ticks=range(8), orientation="vertical")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_facies_plot2.jpg")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_facies_plot2.png")
plt.savefig("/content/drive/MyDrive/NewZealand_data/mckee-"+str(well_name)+"/mckee-"+str(well_name)+"_facies_plot2.pdf")

results = pd.read_excel("/content/drive/MyDrive/paper final/GCN datasets/"+str(cls_name)+"/results/Results_"+str(cls_name)+".xlsx")


res_final = pd.DataFrame(index=range(len(tv_start)))
res_final['Clustering_start'] = tv_start
res_final['Clustering_end'] = tv_end
res_final['T_Very_High'] = ''
res_final['T_High'] = ''
res_final['T_Moderate'] = ''
res_final['T_Very_Low'] = ''
res_final['T_Low'] = ''

res_final['P_Very_High'] = ''
res_final['P_High'] = ''
res_final['P_Moderate'] = ''
res_final['P_Very_Low'] = ''
res_final['P_Low'] = ''

if len(tv_start) == len(tv_end):
  for i in range(len(tv_start)):
    if (tv_end[i] - tv_start[i]) != 0:
      #print(results['True'][tv_start[i]:tv_end[i]].value_counts())
      g1 = results[tv_start[i]:tv_end[i]].groupby('True').size()
      g2 = results[tv_start[i]:tv_end[i]].groupby('Predicted').size()
      he1 = []
      he2 = []
      for ind1, val1 in g1.iteritems():
        he1.append(ind1)

      for ind2, val2 in g2.iteritems():
        he2.append(ind2)

      if "Moderate" in he1:
        res_final['T_Moderate'][i] = g1['Moderate']
      else:
        res_final['T_Moderate'][i] = 0

      if "Low" in he1:
        res_final['T_Low'][i] = g1['Low']
      else:
        res_final['T_Low'][i] = 0

      if "Very_Low" in he1:
        res_final['T_Very_Low'][i] = g1['Very_Low']
      else:
        res_final['T_Very_Low'][i] = 0

      if "Very_High" in he1:
        res_final['T_Very_High'][i] = g1['Very_High']
      else:
        res_final['T_Very_High'][i] = 0

      if "High" in he1:
        res_final['T_High'][i] = g1['High']
      else:
        res_final['T_High'][i] = 0

      if "Moderate" in he2:
        res_final['P_Moderate'][i] = g2['Moderate']
      else:
        res_final['P_Moderate'][i] = 0

      if "Low" in he2:
        res_final['P_Low'][i] = g2['Low']
      else:
        res_final['P_Low'][i] = 0

      if "Very_Low" in he2:
        res_final['P_Very_Low'][i] = g2['Very_Low']
      else:
        res_final['P_Very_Low'][i] = 0

      if "Very_High" in he2:
        res_final['P_Very_High'][i] = g2['Very_High']
      else:
        res_final['P_Very_High'][i] = 0

      if "High" in he2:
        res_final['P_High'][i] = g2['High']
      else:
        res_final['P_High'][i] = 0