import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing_module import preprocess_data
from clustering_module_SMA import clustering
from suggestion_module_SMA import finance

st.set_page_config(page_title="#Customer Segmentation App#",page_icon='💼',layout="centered",initial_sidebar_state="expanded")
st.title("Financial Customer Segmentation")
st.markdown("Upload your ***Customer Data***. 🔧Cleaning!! ->📊 Clustering!! -> 💡Suggesting stratigies!!")
upload_file=st.file_uploader("Upload the CSV File",type=["csv"])

if upload_file:
    st.success("File uploaded Successfully")
    data_raw=pd.read_csv(upload_file)

    st.subheader("step 1: Raw Data Preview")
    st.dataframe(data_raw.head())

    st.subheader("step 2: Preprocessing your data")
    data_cleaned,scaled_data=preprocess_data(data_raw)
    st.write("completed preprocessing")

    st.subheader("step 3: Clustering Customers")
    k=st.slider("Select number of clusters",2,10,4)
    data_clustered,cluster_profiles=clustering(data_cleaned,scaled_data,k)
    st.write("customers clustered into segments")

    st.subheader("📊Cluster profile Summmary")
    st.dataframe(cluster_profiles)

    st.subheader("step 4:💡 Financial suggestions based on segment")
    financial_suggestion=finance(cluster_profiles)
    st.dataframe(financial_suggestion)

    st.subheader("step 5: 📊Visualize graphs")

    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(scaled_data)
    data_clustered["PCA 1"]=pca_result[:,0]
    data_clustered["PCA 2"]=pca_result[:,1]
    fig_1,ax_1=plt.subplots()
    sns.scatterplot(data=data_clustered,x='PCA 1',y='PCA 2',hue='Cluster',palette='muted',ax=ax_1)
    ax_1.set_title("PCA Cluster Visualization")
    st.pyplot(fig_1)

    tsne=TSNE(n_components=2,perplexity=40,learning_rate="auto",init='random',random_state=40)
    tsne_result=tsne.fit_transform(scaled_data)
    data_clustered["TSNE 1"]=tsne_result[:,0]
    data_clustered["TSNE 2"]=tsne_result[:,1]
    fig_2,ax_2=plt.subplots()
    sns.scatterplot(data=data_clustered,x='TSNE 1',y='TSNE 2',hue='Cluster',palette='muted',ax=ax_2)
    ax_2.set_title("T-SNE Cluster Visualization")
    st.pyplot(fig_2)

    st.subheader("Other Cluster Insights")

    st.markdown("**Cluster based on Income levels**")
    fig_3,ax_3=plt.subplots()
    sns.barplot(data=data_clustered,x="Cluster",y="gross income",estimator='mean',ci=None,ax=ax_3)
    st.pyplot(fig_3)

    st.markdown("**Customer distribution by cluster**")
    cluster_counts=data_clustered['Cluster'].value_counts()
    fig_4,ax_4=plt.subplots()
    explode= [0.05] * len(cluster_counts) 
    ax_4.pie(cluster_counts,labels=cluster_counts.index,autopct='%1.1f%%',startangle=90,explode=explode)
    ax_4.axis('Equal')
    st.pyplot(fig_4)

    if 'Unit price' in data_clustered.columns:
        st.markdown("* Price Distribution*")
        fig_5,ax_5=plt.subplots()
        sns.boxplot(data=data_clustered,x="Cluster",y="Unit price",ax=ax_5)
        st.pyplot(fig_5)

    st.download_button("Download Clustered data",data_clustered.to_csv(index=False),"Clustered_data.csv")
else:
    st.warning("please upload your file")