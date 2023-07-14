import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import hdbscan
import folium
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

# Loading CSV File (Output of Text processing step)
input_file = "Results/TextualMetadataProcessing/TP_op1M.csv"
df = pd.read_csv(input_file)
df1 = df.copy()
coord = {}
Lat = np.asarray(df['Latitude'])
Long = np.asarray(df['Longitude'])
coords = df[['Longitude', 'Latitude']].values
rads = np.radians(coords)

# Clustering step
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, metric='haversine', gen_min_span_tree=True, min_cluster_size=20, min_samples=None, p=None).fit(rads)
cluster_labels = clusterer.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(-1, num_clusters)])
cluster_label = clusterer.labels_
x = cluster_label
cluster_points = {}
for i in range(num_clusters + 1):
    cluster_points[i] = []

for i in range(len(x)):
    l = cluster_points[x[i]]
    l.append(coords[i])
    cluster_points[x[i]] = l

plt.figure(figsize=(12, 12))

# Plotting points on map
m = Basemap(projection='merc', resolution='l', epsg=4269, llcrnrlon=-122.567288, llcrnrlat=37.696627, urcrnrlon=-122.329308, urcrnrlat=37.852144)
x, y = m(coords[:, 1], coords[:, 0])
m.scatter(x, y, 5, marker='o', color='b')
m.arcgisimage(service='World_Shaded_Relief', xpixels=5000, verbose=False)
plt.show()

def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    '''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

plt.figure(figsize=(12, 12))

m = Basemap(projection='merc', resolution='l', epsg=4269, llcrnrlon=-122.567288, llcrnrlat=37.696627, urcrnrlon=-122.329308, urcrnrlat=37.852144)

unique_label = np.unique(cluster_labels)
cmaps = get_cmap(num_clusters)

for i, cluster in enumerate(clusters):
    lons_select = cluster[:, 1]
    lats_select = cluster[:, 0]
    x, y = m(lons_select, lats_select)
    m.scatter(x, y, 5, marker='o', color=cmaps(i), zorder=10)

m.arcgisimage(service='World_Shaded_Relief', xpixels=5000, verbose=False)
plt.show()

# Finding Cluster representatives

cluster_exemplar_photoid = {}
cluster_exemplar = {}
for i in range(num_clusters + 1):
    cluster_exemplar[i] = []
    cluster_exemplar_photoid[i] = []

for i in range(num_clusters):
    l = np.rad2deg(clusterer.exemplars_[i])
    cluster_exemplar[i] = l
