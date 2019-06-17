import sacred
ex = sacred.Experiment("EEG_K-Means_Clustering")
from sacred.observers import MongoObserver

from sklearn.decomposition import PCA
import data_reader as read
import pandas as pd
import numpy as np
import scipy.cluster as cluster
from sklearn.cluster import MiniBatchKMeans
import util_funcs

@ex.named_config
def debug_config():
    num_eegs = 20

@ex.config
def config():
    num_pca_comps = 6
    num_k_means = 5
    num_eegs = None
    num_cores = None
    data_split = "dev_test"
    ref = "01_tcp_ar"
    ex.observers.append(MongoObserver.create(client=util_funcs.get_mongo_client()))


@ex.capture
def get_data(num_eegs, data_split, ref):
    """Returns the data to process.

    Parameters
    ----------
    num_eegs : type
        Description of parameter `num_eegs`.
    data_split : type
        Description of parameter `data_split`.
    ref : type
        Description of parameter `ref`.

    Returns
    -------
    tuple
        first elem is data freq coefficients of (num_instances, num_freq_bins),
        second elem is annotation percentages of (num_instances, annotations)

    """
    edf_dataset = read.EdfDataset(data_split, ref, num_files=num_eegs)

    fft_reader = read.EdfFFTDatasetTransformer(
        edf_dataset=edf_dataset,
        precache=True
        )
    data = fft_reader[0:num_eegs]
    cols = [set(datum[0].dropna().columns) for datum in data]
    annotations = np.array([datum[1].mean().values for datum in data])
    common_cols = list(cols[0].intersection(*cols))
    x_data = np.array([datum[0][common_cols].values for datum in data])
    x_data = x_data.reshape(x_data.shape[0], -1)
    return x_data, annotations



@ex.automain
def main(num_pca_comps, num_k_means):
    data, annotations = get_data()
    pca = PCA(num_pca_comps)
    pca.fit(data)
    pca_vects = pca.transform(data)
    kmeans = MiniBatchKMeans(num_k_means)
    cluster_pct = kmeans.fit_transform(cluster.vq.whiten(pca_vects))
    cluster_pct = np.apply_along_axis(lambda x: x/x.sum(), axis=1, arr=cluster_pct)
    percent_y_per_cluster = pd.DataFrame(annotations.T @ cluster_pct, index=util_funcs.get_annotation_types())
    percent_y_per_cluster = (percent_y_per_cluster.T / percent_y_per_cluster.sum(axis=1)).fillna(0)
    print(percent_y_per_cluster)
    return {'pca': pca, 'kmeans': kmeans, 'cluster_pct': cluster_pct, 'percent_y_per_cluster': percent_y_per_cluster}
