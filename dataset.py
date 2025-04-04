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



#petrophysical properties
def shale_volume(gamma_ray, gamma_ray_max, gamma_ray_min):
    vshale = (gamma_ray - gamma_ray_min) / (gamma_ray_max - gamma_ray_min)
    #vshale = vshale / (3 - 2 * vshale)
    vshale = 0.083 * (2 ** (2 * 3.7 * vshale  ) - 1) #for teritary rocks
    return round(vshale, 4)

def density_porosity(input_density, matrix_density, fluid_density):
    denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
    return round(denpor, 4)







#Calculate Shale Volume
df['VSHALE'] = shale_volume(df['CGR'], df['CGR'].max(),df['CGR'].min())


# KMeans Clustering
features = df[['VSHALE', 'PHI', 'SW', 'GR', 'DENSITY']].copy()
features = features.dropna()  
features = features.drop_duplicates()
features = features.reset_index(drop=True) 

x_scaled = scale(features)
x_scaled

wcss = []

cl_num = 12
for i in range (1,cl_num):
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

range_n_clusters = [2,3,4,5,6,7,8,9,10,11]

for n_clusters  in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x_scaled)

    silhouette_avg = silhouette_score(x_scaled, cluster_labels)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

cls_name = 5

# Node Dataset

df['Facies_pred'] = ''
kmeans_ = KMeans(cls_name,n_init=100)
kmeans_.fit(x_scaled)
df['Facies_pred'] =kmeans_.fit_predict(x_scaled)

f = df['Facies_pred']

df = ''
cls_name = 8
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





def list_index(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))