from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
def clustering(data_cleaned,scaled_data,k):
    Kmeans=KMeans(n_clusters=k,random_state=40)
    Clusters=Kmeans.fit_predict(scaled_data)
    data_cleaned["Cluster"]=Clusters
    profile=data_cleaned.groupby("Cluster").mean(numeric_only=True).round(2)
    return data_cleaned,profile