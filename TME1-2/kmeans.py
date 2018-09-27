# Code here the `compute_visual_dict` function
def compute_visual_dict(sift, n_clusters=1000, n_init=1, verbose=1):
    # reorder data
    dim_sift = sift[0].shape[-1]
    sift = [s.reshape(-1, dim_sift) for s in sift]
    sift = np.concatenate(sift, axis=0)
    # remove zero vectors
    keep = ~np.all(sift==0, axis=1)
    sift = sift[keep]
    # randomly pick sift
    ids, _ = compute_split(sift.shape[0], pc=0.05)
    sift = sift[ids]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, verbose=verbose, n_jobs=-1).fit(sift)
    cc = kmeans.cluster_centers_
    n = 128
    return np.append(cc, np.zeros(n).reshape(1,n), axis=0)
    # TODO compute kmeans on `sift`, get cluster centers, add zeros vector