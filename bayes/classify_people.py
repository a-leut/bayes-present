import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import mixture

COMPONENTS = 25
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
        clusters = get_clusters(x).astype('int')
        res['t%d' % i] = clusters
    res.to_csv('cluster_data_%s.csv' % COMPONENTS, index=False)

if __name__ == '__main__':
    main()
