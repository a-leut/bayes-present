import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import mixture

COMPONENTS = 2
SAMPLES = 10

def get_clusters(array):
    dpgmm = mixture.DPGMM(n_components=COMPONENTS)
    dpgmm.fit(array)
    return dpgmm.predict(array)

def main():
    with open('clean_data.csv', 'rb') as f:
        x = np.loadtxt(f, delimiter=',', skiprows=1)
    res = pd.DataFrame()
    for i in tqdm(range(SAMPLES)):
        clus_assign = get_clusters(x).astype('int')
        groups = np.unique(clus_assign, return_counts=True)
        print(groups)
        res['t%d' % i] = clus_assign
    res.to_csv('cluster_data_%s_comps_%s_trials.csv' % (COMPONENTS, SAMPLES), index=False)

if __name__ == '__main__':
    main()
