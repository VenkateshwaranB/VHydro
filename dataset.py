import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import osgeo
import re
import itertools

from itertools import groupby,count



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