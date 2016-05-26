import numpy as np
from sklearn import mixture

def get_clusters(array):
    dpgmm = mixture.DPGM(n_components=25)
    dpgmm.fit(array)
    return dpgmm.predict(array)

def main():
    with open('clean_data.csv', 'rb') as f:
        x = np.loadtext(f, delimiter=',', skiprows=1)
        clusters = get_clusters()
        print(clusters)

if __name__ == '__main__':
    main()
