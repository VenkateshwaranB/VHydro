import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import osgeo
import re
import itertools

from itertools import groupby,count
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score


from lasio import lasio

############################################################################################################
# Data Import
############################################################################################################

las = lasio.read("/content/drive/MyDrive/paper final/GCN datasets/las files/las file 1.las")  # ask the user to input the las file path
df = las.df()

# KMeans Clustering
features = df[['VSHALE', 'PHI', 'SW', 'GR', 'DENSITY']].copy()   # ask the user to input the features to be used for clustering or select from the listed columns from las files
features = features.dropna()  
features = features.drop_duplicates()
features = features.reset_index(drop=True) 

x_scaled = scale(features)
x_scaled

############################################################################################################
############################################################################################################
#                                               Graph Dataset
############################################################################################################
############################################################################################################


############################################################################################################
# KMeans Clustering
############################################################################################################
wcss = []

cl_num = 10   ## Ask the user to input the number of clusters
for i in range (5,cl_num):
    kmeans= KMeans(i, random_state=10)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
wcss

number_clusters = range(1,cl_num)
plt.figure(figsize=(10,8))
plt.plot(number_clusters, wcss,'*-' )
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('Within-cluster Sum of Squares',fontsize=20)

range_n_clusters = [5,6,7,8,9,10]

for n_clusters  in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x_scaled)

    silhouette_avg = silhouette_score(x_scaled, cluster_labels)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

cls_name = 5 # upon this do create the dataset for each cls_name like 5 to 10


############################################################################################################
# Node Dataset
############################################################################################################

df['Facies_pred'] = ''
kmeans_ = KMeans(cls_name,n_init=100)
kmeans_.fit(x_scaled)
df['Facies_pred'] =kmeans_.fit_predict(x_scaled)

f = df['Facies_pred']

df = ''
cls_name = 8 # this is recursive from starting cls_num to end cls_num like 5 to 10
df = pd.read_excel("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/facies for "+ str(cls_name) +".xlsx")

fr1 = []
fr2 = []
a1 = df['Facies_pred'].to_list()
l1 = [idx for idx,value in enumerate(a1) if value == 0]
hwe1 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
for i in range(len(hwe1)):
  fr1.append(df['DEPTH'][hwe1[i][0]])
  fr2.append(df['DEPTH'][hwe1[i][-1]])

se1 = []
se2 = []
a2 = df['Facies_pred'].to_list()
l1 = [idx for idx,value in enumerate(a2) if value == 1]
hwe2 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
for i in range(len(hwe2)):
  se1.append(df['DEPTH'][hwe2[i][0]])
  se2.append(df['DEPTH'][hwe2[i][-1]])

th1 = []
th2 = []
a3 = df['Facies_pred'].to_list()
l1 = [idx for idx,value in enumerate(a3) if value == 2]
hwe3 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
for i in range(len(hwe3)):
  th1.append(df['DEPTH'][hwe3[i][0]])
  th2.append(df['DEPTH'][hwe3[i][-1]])

fo1 = []
fo2 = []
a4 = df['Facies_pred'].to_list()
l1 = [idx for idx,value in enumerate(a4) if value == 3]
hwe4 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
for i in range(len(hwe4)):
  fo1.append(df['DEPTH'][hwe4[i][0]])
  fo2.append(df['DEPTH'][hwe4[i][-1]])

fi1 = []
fi2 = []
a5 = df['Facies_pred'].to_list()
l1 = [idx for idx,value in enumerate(a5) if value == 4]
hwe5 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
for i in range(len(hwe5)):
  fi1.append(df['DEPTH'][hwe5[i][0]])
  fi2.append(df['DEPTH'][hwe5[i][-1]])

if cls_name > 5:
  si1 = []
  si2 = []
  a6 = df['Facies_pred'].to_list()
  l1 = [idx for idx,value in enumerate(a6) if value == 5]
  hwe6 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
  for i in range(len(hwe6)):
    si1.append(df['DEPTH'][hwe6[i][0]])
    si2.append(df['DEPTH'][hwe6[i][-1]])

if cls_name > 6:
  se1 = []
  se2 = []
  a7 = df['Facies_pred'].to_list()
  l1 = [idx for idx,value in enumerate(a7) if value == 6]
  hwe7 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
  for i in range(len(hwe7)):
    se1.append(df['DEPTH'][hwe7[i][0]])
    se2.append(df['DEPTH'][hwe7[i][-1]])

if cls_name > 7:
  ei1 = []
  ei2 = []
  a8 = df['Facies_pred'].to_list()
  l1 = [idx for idx,value in enumerate(a8) if value == 7]
  hwe8 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
  for i in range(len(hwe8)):
    ei1.append(df['DEPTH'][hwe8[i][0]])
    ei2.append(df['DEPTH'][hwe8[i][-1]])

if cls_name > 8:
  ni1 = []
  ni2 = []
  a9 = df['Facies_pred'].to_list()
  l1 = [idx for idx,value in enumerate(a9) if value == 8]
  hwe9 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
  for i in range(len(hwe9)):
    ni1.append(df['DEPTH'][hwe9[i][0]])
    ni2.append(df['DEPTH'][hwe9[i][-1]])

if cls_name > 9:
  te1 = []
  te2 = []
  a10 = df['Facies_pred'].to_list()
  l1 = [idx for idx,value in enumerate(a10) if value == 9]
  hwe10 = [list(g) for _,g in groupby(l1,key=lambda n,c=count():n-next(c))]
  for i in range(len(hwe10)):
    te1.append(df['DEPTH'][hwe10[i][0]])
    te2.append(df['DEPTH'][hwe10[i][-1]])


if cls_name == 5:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5
  total_dept == len(tt_dept)
elif cls_name == 6:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5) + len(hwe6)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5 + hwe6
  total_dept == len(tt_dept)
elif cls_name == 7:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5) + len(hwe6) + len(hwe7)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5 + hwe6 + hwe7
  total_dept == len(tt_dept)
elif cls_name == 8:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5) + len(hwe6) + len(hwe7) + len(hwe8)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5 + hwe6 + hwe7 + hwe8
  total_dept == len(tt_dept)
elif cls_name == 9:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5) + len(hwe6) + len(hwe7) + len(hwe8) + len(hwe9)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5 + hwe6 + hwe7 + hwe8 + hwe9
  total_dept == len(tt_dept)
elif cls_name == 10:
  total_dept = len(hwe1) + len(hwe2) + len(hwe3) + len(hwe4) + len(hwe5) + len(hwe6) + len(hwe7) + len(hwe8) + len(hwe9) + len(hwe10)
  tt_dept = hwe1 + hwe2 + hwe3 + hwe4 + hwe5 + hwe6 + hwe7 + hwe8 + hwe9 + hwe10
  total_dept == len(tt_dept)

tv = sorted(tt_dept)

len(tv)

tv_start = []
tv_end = []
for i in range(len(tv)):
  tv_start.append(tv[i][0])
  tv_end.append(tv[i][-1])

tv_final = pd.DataFrame({'clustering_start': tv_start,'clustering_end':tv_end})
#tv_final.to_excel("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/clustering range "+ str(cls_name) +".xlsx")

############################################################################################################
# Spliting train/test/validation node dataset
############################################################################################################

def list_index(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))

list_index(tv, 1006)

### Traing(80%) | Testing(10%) | Validation(10%)

# The range of the above three divisions are followed:
# 1.   **Cluster 5**:
# *    0(0 list, 0 value) to 371(4 list, 111 value), 680(45 list, 0 value) to 707(46 list, 15 value), 995(73 list, 0 value) to 2667(210 list, 50 value)
# *   372(5 list, 0 value) to 679(44 list, 55 value)
# *   708(47 list, 0 value) to 994(72 list, 30 value)

# 2.   **Cluster 6**:
# *    0(0 list, 0 value) to 371(4 list, 111 value), 681(45 list, 0 value) to 707(49 list, 8 value), 1003(97 list, 0 value) to 2667(286 list, 48 value)
# *   372(5 list, 0 value) to 680(44 list, 56 value)
# *   708(50 list, 0 value) to 1002(96 list, 10 value)

# 2.   **Cluster 7**:
# *    0(0 list, 0 value) to 371(4 list, 111 value), 680(62 list, 0 value) to 707(65 list, 8 value), 996(104 list, 0 value) to 2667(292 list, 48 value)
# *   372(5 list, 0 value) to 679(61 list, 45 value)
# *   708(66 list, 0 value) to 995(103 list, 28 value)

# 2.   **Cluster 8**:
# *    0(0 list, 0 value) to 370(3 list, 112 value), 681(55 list, 7 value) to 707(56 list, 16 value), 1002(112 list, 0 value) to 2667(323 list, 48 value)
# *   371(4 list, 0 value) to 680(54 list, 7 value)
# *   708(57 list, 0 value) to 1001(111 list, 5 value)

# 2.   **Cluster 9**:
# *    0(0 list, 0 value) to 370(3 list, 112 value), 681(453 list, 0 value) to 707(59 list, 16 value), 1002(71 list, 0 value) to 2667(345 list, 48 value)
# *   371(4 list, 0 value) to 680(52 list, 6 value)
# *   708(60 list, 0 value) to 1001(122 list, 5 value)

# 2.   **Cluster 10**:
# *    0(0 list, 0 value) to 370(5 list, 112 value), 680(51 list, 0 value) to 706(54 list, 14 value), 1006(116 list, 0 value) to 2667(356 list, 48 value)
# *   371(6 list, 0 value) to 679(50 list, 6 value)
# *   707(55 list, 0 value) to 1005(115 list, 3 value)

train1 = 6
train2 = 51
train3 = 55
train4 = 116

test1 = 6
test2 = 51

valid1 = 55
valid2 = 116


### Make a node connection using depth
frows = 0
for i in range(len(tv)):
  tvl = len(tv[i])
  frows = frows + (tvl * (tvl - 1))

nodeF = pd.DataFrame(index=range(frows))
nodeF['full'] = ''
nodeF['full1'] = ''
nodeF['train'] = ''
nodeF['train1'] = ''
nodeF['test'] = ''
nodeF['test1'] = ''
nodeF['valid'] = ''
nodeF['valid1'] = ''


#full connections
fir = []
k1 = 0
for k in range(len(tv)):
  fir = tv[k]
  print(fir)
  print(k)
  for i in range(len(fir)):
    ler = fir[i]
    for j in range(len(fir)):
      if fir[i] != fir[j]:
        nodeF['full'][k1] = df['DEPTH'][ler]
        nodeF['full1'][k1] = df['DEPTH'][fir[j]]
        k1 = k1 + 1

# training dataset connections
train = []
train_tv = []
train_tv = tv[0:train1] + tv[train2:train3] + tv[train4:]

k1 = 0
for k in range(len(train_tv)):
  train = train_tv[k]
  for i in range(len(train)):
    ler = train[i]
    for j in range(len(train)):
      if train[i] != train[j]:
        nodeF['train'][k1] = df['DEPTH'][ler]
        nodeF['train1'][k1] = df['DEPTH'][train[j]]
        k1 = k1 + 1


# testing dataset connections
test = []
test_tv = []
test_tv = tv[test1:test2]

k1 = 0
for k in range(len(test_tv)):
  test = test_tv[k]
  for i in range(len(test)):
    ler = test[i]
    for j in range(len(test)):
      if test[i] != test[j]:
        nodeF['test'][k1] = df['DEPTH'][ler]
        nodeF['test1'][k1] = df['DEPTH'][test[j]]
        k1 = k1 + 1

# validation dataset connections
valid = []
valid_tv = []
valid_tv = tv[valid1:valid2]

k1 = 0
for k in range(len(valid_tv)):
  valid = valid_tv[k]
  for i in range(len(valid)):
    ler = valid[i]
    for j in range(len(valid)):
      if valid[i] != valid[j]:
        nodeF['valid'][k1] = df['DEPTH'][ler]
        nodeF['valid1'][k1] = df['DEPTH'][valid[j]]
        k1 = k1 + 1

fut = ''
trt = ''
tet = ''
vat = ''
if cls_name == 5:
  fut = 136018
  trt = 125204
  tet = 5932
  vat = 4882
elif cls_name == 6:
  fut = 125764
  trt = 116108
  tet = 6014
  vat = 3642
elif cls_name == 7:
  fut = 119672
  trt = 112798
  tet = 4098
  vat = 2776
elif cls_name == 8:
  fut = 119672
  trt = 111622
  tet = 4394
  vat = 2878
elif cls_name == 9:
  fut = 119200
  trt = 111622
  tet = 4394
  vat = 2878
elif cls_name == 10:
  fut = 111834
  trt = 104242
  tet = 4870
  vat = 2722

node_data = nodeF


we2 = pd.DataFrame()
we2['depth'] = node_data['full'][0:fut]
we2['target'] = node_data['full1'][0:fut]

gnt = np.array(we2)
gnt = gnt.astype('str')

for i in range(len(gnt)):
  gnt[i][0] = np.char.replace(gnt[i][0], '.', '')
  gnt[i][1] = np.char.replace(gnt[i][1], '.', '')

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_full_node "+ str(cls_name) +"new.txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(gnt)

we2 = pd.DataFrame()
we2['depth'] = node_data['train'][0:trt]
we2['target'] = node_data['train1'][0:trt]

gnt = np.array(we2)
gnt = gnt.astype('str')

for i in range(len(gnt)):
  gnt[i][0] = np.char.replace(gnt[i][0], '.', '')
  gnt[i][1] = np.char.replace(gnt[i][1], '.', '')

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_train_node "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(gnt)

we2 = pd.DataFrame()
we2['depth'] = node_data['test'][0:tet]
we2['target'] = node_data['test1'][0:tet]

gnt = np.array(we2)
gnt = gnt.astype('str')

for i in range(len(gnt)):
  gnt[i][0] = np.char.replace(gnt[i][0], '.', '')
  gnt[i][1] = np.char.replace(gnt[i][1], '.', '')

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_test_node "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(gnt)

we2 = pd.DataFrame()
we2['depth'] = node_data['valid'][0:vat]
we2['target'] = node_data['valid1'][0:vat]

gnt = np.array(we2)
gnt = gnt.astype('str')

for i in range(len(gnt)):
  gnt[i][0] = np.char.replace(gnt[i][0], '.', '')
  gnt[i][1] = np.char.replace(gnt[i][1], '.', '')

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_valid_node "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(gnt)


############################################################################################################
#                                         Edge Dataset
############################################################################################################

############################################################################################################
# Petrophysical Properties Calculation
############################################################################################################

# Shale Volume
def shale_volume(gamma_ray, gamma_ray_max, gamma_ray_min):
  vshale = (gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min)
  #vshale = vshale / (3 - 2 * vshale)
  vshale = 0.083 * (2 ** (2 * 3.7 * vshale  ) - 1) #for teritary rocks
  return round(vshale, 4)

# Density Porosity
def density_porosity(input_density, matrix_density, fluid_density):
  denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
  return round(denpor, 4)

# water Saturation
def sw_archie(porosity, rt):
  sw = ((1 * (porosity**2)) / (rt * 0.03))**(1/2)
  return sw

# oil Saturation
def ow_archie(sw):
  ow = 1 - sw
  return ow

#permeability
def permeabilty(porosity):
  pe = 0.00004 * np.exp(57.117 * porosity)
  return pe

holeC = df

well = pd.DataFrame()
#Calculate Shale Volume
#well['VSHALE'] = shale_volume(holeC['SGR'], holeC['SGR'],holeC['SGR'])
well['VSHALE'] = shale_volume(holeC['CGR'], holeC['CGR'].max(),holeC['CGR'].min())
#Calculate density porosity
well['PHI'] = density_porosity(holeC['RHOB'], 2.65, 1)
#Calculate PHIE (Effective porosity)
well['PHIECALC'] = well['PHI'] - (well['VSHALE'] * 0.3)
#Calculate water saturation
well['WSAT'] = sw_archie(well['PHIECALC'], np.log(holeC['ILD']))
#Calculate Oil saturation
well['OSAT'] = ow_archie(well['WSAT'])
#Calculate permeability
#well['PERM'] = permeabilty(well['PHI'], well['PHIECALC'], well['WSAT'])
well['PERM'] = permeabilty(well['PHIECALC'])



hwe1 = []
hwe2 = []
hwe3 = []
hwe4 = []
for i in range(len(well['PHIECALC'])):
  if well['PHIECALC'][i] <= 0.1:
    hwe1.append(well['PHIECALC'][i])
  elif well['PHIECALC'][i] <= 0.2 and well['PHIECALC'][i] > 0.1:
    hwe2.append(well['PHIECALC'][i])
  elif well['PHIECALC'][i] <= 0.3 and well['PHIECALC'][i] > 0.2:
    hwe3.append(well['PHIECALC'][i])
  else:
    hwe4.append(well['PHIECALC'][i])


hwe1 = []
hwe2 = []
hwe3 = []
hwe4 = []
hwe5 = []
hwe6 = []
for i in range(len(well['PERM'])):
  if well['PERM'][i] <= 0.01:
    hwe1.append(well['PERM'][i])
  elif well['PERM'][i] <= 1 and well['PERM'][i] > 0.01:
    hwe2.append(well['PERM'][i])
  elif well['PERM'][i] <= 10 and well['PERM'][i] > 1:
    hwe3.append(well['PERM'][i])
  elif well['PERM'][i] <= 100 and well['PERM'][i] > 10:
    hwe4.append(well['PERM'][i])
  elif well['PERM'][i] > 100:
    hwe5.append(well['PERM'][i])
  else:
    print(well['PERM'][i])
    hwe6.append(well['PERM'][i])


hwe1 = []
hwe2 = []
hwe3 = []
for i in range(len(well['VSHALE'])):
  if well['VSHALE'][i] <= 0.5:
    hwe1.append(well['VSHALE'][i])
  elif well['VSHALE'][i] > 0.5:
    hwe2.append(well['VSHALE'][i])
  else:
    hwe3.append(well['VSHALE'][i])

hwe1 = []
hwe2 = []
hwe3 = []
for i in range(len(well['OSAT'])):
  if well['OSAT'][i] >= 0.5:
    hwe1.append(well['OSAT'][i])
  elif well['OSAT'][i] < 0.5:
    hwe2.append(well['OSAT'][i])
  else:
    hwe3.append(well['OSAT'][i])


hwe1 = []
hwe2 = []
hwe3 = []
for i in range(len(well['WSAT'])):
  if well['WSAT'][i] <= 0.5:
    hwe1.append(well['WSAT'][i])
  elif well['WSAT'][i] > 0.5:
    hwe2.append(well['WSAT'][i])
  else:
    hwe3.append(well['WSAT'][i])


fe_vs = well['VSHALE']
a1 = fe_vs.min()
b1 = fe_vs.mean()
c1 = fe_vs.max()
print(a1, b1, c1)

fe_vs = well['PHIECALC']
a1 = fe_vs.min()
b1 = fe_vs.mean()
c1 = fe_vs.max()
print(a1, b1, c1)

fe_vs = well['PERM']
a1 = fe_vs.min()
b1 = fe_vs.mean()
c1 = fe_vs.max()
print(a1, b1, c1)

fe_vs = well['OSAT']
a1 = fe_vs.min()
b1 = fe_vs.mean()
c1 = fe_vs.max()
print(a1, b1, c1)

fe_vs = well['WSAT']
a1 = fe_vs.min()
b1 = fe_vs.mean()
c1 = fe_vs.max()
print(a1, b1, c1)

############################################################################################################
# labeled dataset & adjacency matrix
############################################################################################################

adj = pd.DataFrame()
dom = well.drop(columns=['PHI'])
adj['DEPTH'] = holeC['DEPTH']
adj['PE_5'] = ''
adj['PE_4'] = ''
adj['PE_3'] = ''
adj['PE_2'] = ''
adj['PE_1'] = ''
adj['PO_1'] = ''
adj['PO_2'] = ''
adj['PO_3'] = ''
adj['VS_1'] = ''
adj['VS_2'] = ''
adj['SW_1'] = ''
adj['SW_2'] = ''
adj['OW_1'] = ''
adj['OW_2'] = ''

for i in range(len(adj['DEPTH'])):
  #permeabilty label
  if dom['PERM'][i] <= 0.01:
    adj['PE_1'][i] = 1
  else:
    adj['PE_1'][i] = 0
  if dom['PERM'][i] <= 1 and dom['PERM'][i] > 0.01:
    adj['PE_2'][i] = 1
  else:
    adj['PE_2'][i] = 0
  if dom['PERM'][i] <= 10 and dom['PERM'][i] > 1:
    adj['PE_3'][i] = 1
  else:
    adj['PE_3'][i] = 0
  if dom['PERM'][i] <= 100 and dom['PERM'][i] > 10:
    adj['PE_4'][i] = 1
  else:
    adj['PE_4'][i] = 0
  if dom['PERM'][i] > 100:
    adj['PE_5'][i] = 1
  else:
    adj['PE_5'][i] = 0
  #porosity label
  if dom['PHIECALC'][i] <= 0.1:
    adj['PO_1'][i] = 1
  else:
    adj['PO_1'][i] = 0
  if dom['PHIECALC'][i] <= 0.2 and dom['PHIECALC'][i] > 0.1:
    adj['PO_2'][i] = 1
  else:
    adj['PO_2'][i] = 0
  if dom['PHIECALC'][i] <= 0.3 and dom['PHIECALC'][i] > 0.2:
    adj['PO_3'][i] = 1
  else:
    adj['PO_3'][i] = 0
  #volume shale label
  if dom['VSHALE'][i] <= 0.5:
    adj['VS_2'][i] = 1
  else:
    adj['VS_2'][i] = 0
  if dom['VSHALE'][i] > 0.5:
    adj['VS_1'][i] = 1
  else:
    adj['VS_1'][i] = 0
  #water saturation label
  if dom['WSAT'][i] <= 0.5:
    adj['SW_2'][i] = 1
  else:
    adj['SW_2'][i] = 0
  if dom['WSAT'][i] > 0.5:
    adj['SW_1'][i] = 1
  else:
    adj['SW_1'][i] = 0
  #Oil saturation label
  if dom['OSAT'][i] >= 0.5:
    adj['OW_2'][i] = 1
  else:
    adj['OW_2'][i] = 0
  if dom['OSAT'][i] < 0.5:
    adj['OW_1'][i] = 1
  else:
    adj['OW_1'][i] = 0


### Label generation for depths

adj['Values'] = adj['PE_1'].astype(str) + adj['PE_2'].astype(str) + adj['PE_3'].astype(str) + adj['PE_4'].astype(str) + adj['PE_5'].astype(str) + adj['PO_1'].astype(str) + adj['PO_2'].astype(str) + adj['PO_3'].astype(str) + adj['VS_1'].astype(str) + adj['VS_2'].astype(str) + adj['SW_1'].astype(str) + adj['SW_2'].astype(str) + adj['OW_1'].astype(str) + adj['OW_2'].astype(str)

adj['cate'] = pd.Series(adj['Values'], dtype="category")

get1 = adj['cate'].unique()


adj["labels"] = ''
k = 0
for i in range(len(adj['DEPTH'])):
  if adj['Values'][i] == get1[0]:
    adj["labels"][i] = "Moderate"
  elif adj['Values'][i] == get1[1]:
    adj["labels"][i] = "Moderate"
  elif adj['Values'][i] == get1[2]:
     adj["labels"][i] = "Very_Low"
  elif adj['Values'][i] == get1[3]:
     adj["labels"][i] = "Very_Low"
  elif adj['Values'][i] == get1[4]:
     adj["labels"][i] = "Low"
  elif adj['Values'][i] == get1[5]:
     adj["labels"][i] = "Low"
  elif adj['Values'][i] == get1[6]:
     adj["labels"][i] = "Moderate"
  elif adj['Values'][i] == get1[7]:
     adj["labels"][i] = "High"
  elif adj['Values'][i] == get1[8]:
     adj["labels"][i] = "High"
  else:
    k = k + 1
    print(adj['DEPTH'][i])
    print(i)
    print(k)

print(k)


############################################################################################################
# Spliting train/test/validation edge dataset
############################################################################################################

ad_data = adj.drop(columns=['Values', 'cate'])
ad_data['RESULTS'] = adj['labels']
ad_data = ad_data.drop(columns=['labels'])

ght = np.array(ad_data).astype('str')

for i in range(len(ght)):
  ght[i][0] = np.char.replace(ght[i][0], '.', '')


few = [ad_data['DEPTH'].to_list(), ad_data['PE_1'].to_list(), ad_data['PE_2'].to_list(), ad_data['PE_3'].to_list(), ad_data['PE_4'].to_list(), ad_data['PE_5'].to_list(), ad_data['PO_1'].to_list(), ad_data['PO_2'].to_list(), ad_data['PO_3'].to_list(), ad_data['VS_1'].to_list(), ad_data['VS_2'].to_list(), ad_data['SW_1'].to_list(), ad_data['SW_2'].to_list(), ad_data['OW_1'].to_list(), ad_data['OW_2'].to_list(), ad_data['RESULTS'].to_list()]

few = []

for i in range(len(ad_data)):
  few.append(ad_data.loc[i])

#full edge dataset
import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_full_edge "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(few)

#training edge dataset
edget = []
got1 = []
got1 = tv[0:train1] + tv[train2:train3] + tv[train4:]

for k in range(len(got1)):
  traine = got1[k]
  for i in range(len(traine)):
    edget.append(few[traine[i]])

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_train_edge "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(edget)


#testing edge dataset
edget = []
got1 = []
got1 = tv[test1:test2]

for k in range(len(got1)):
  teste = got1[k]
  for i in range(len(teste)):
    edget.append(few[teste[i]])

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_test_edge "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(edget)

#validation edge dataset
edget = []
got1 = []
got1 = tv[valid1:valid2]

for k in range(len(got1)):
  valide = got1[k]
  for i in range(len(valide)):
    edget.append(few[valide[i]])

import csv
with open("/content/drive/MyDrive/paper final/GCN datasets/"+ str(cls_name) +"/BF_valid_edge "+ str(cls_name) +".txt", 'w') as f:
  writer = csv.writer(f, delimiter='\t')
  writer.writerows(edget)