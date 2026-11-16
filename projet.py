import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# plt.rcParams["figure.figsize"] = (8, 5)
# RANDOM_STATE = 42

df = pd.read_csv("data/data.csv")
# df.columns = ["id", "title", "location", "revenu", "score"]
print(df.head())
