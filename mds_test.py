import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import sys

from sammon import sammon
from scipy.spatial import Voronoi, voronoi_plot_2d


def load_graph(path):
    n = None
    graph = {}
    with open(path, 'r') as f:
        it = 0
        for line in f:
            nums = map(int, line.split())
            if n is None:
                n = nums[0]
            else:
                graph[it] = nums
            it += 1
        assert n + 1 == it
    return n, graph


def calc_weights(n, graph):
    ws = np.zeros((n, n))
    for v in xrange(1, n+1):
        for u in xrange(1, n+1):
            ws[u-1,v-1] = len(set(graph[u]).union(graph[v])) - len(set(graph[u]).intersection(graph[v]))
    return ws


def dijksta(n, graph, weights, v1, v2):
    Q = set()
    dists = np.zeros(n)
    prev = np.zeros(n)
    for v in xrange(n):
        dists[v] = np.inf
        prev[v] = np.nan
        Q.add(v+1)
        
    dists[v1-1] = 0
    while Q:
        u = min([(dists[x-1], x) for x in Q])[1]
        Q.remove(u)
        for v in graph[u]:
            d = weights[u-1,v-1]
            alt = dists[u-1] + d
            if alt < dists[v-1]:
                dists[v-1] = alt
                prev[v-1] = u
    return dists[v2-1]


def calc_dists(n, graph, weights):
    dists = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(i+1,n):
            d = dijksta(n, graph, weights, i+1, j+1)
            dists[i,j] = d
    return dists + dists.T


def plot_a_thing(data_trans, graph, figname=None, to_file=True):
    assert not to_file or figname
    vor = Voronoi(data_trans)

    reds = 0
    plt.figure(figsize=(10,10))
    plt.scatter(data_trans[:,0], data_trans[:,1], linewidths=.2, s=50)
    for i in xrange(n):
        plt.text(data_trans[i][0], data_trans[i][1], str(i+1), color="red", fontsize=12)
        for j in graph[i+1]:
            if np.where(np.all(np.array([i,j-1]) == vor.ridge_points, axis=1))[0].shape == (0,) and \
               np.where(np.all(np.array([j-1,i]) == vor.ridge_points, axis=1))[0].shape == (0,):
                plt.plot((data_trans[i][0], data_trans[j-1][0]),
                         (data_trans[i][1], data_trans[j-1][1]), 'r-')
                reds += 1
            else:
                plt.plot((data_trans[i][0], data_trans[j-1][0]),
                         (data_trans[i][1], data_trans[j-1][1]), 'g-')

    plt.xlim(-2,2)
    plt.ylim(-2,2)

    i = 0
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k--')
        i += 1

    center = data_trans.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = data_trans[pointidx[1]] - data_trans[pointidx[0]] # tangent
            t /= np.linalg.norm(t)
            norm = np.array([-t[1], t[0]]) # normal
            midpoint = data_trans[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, norm)) * norm * 100
            plt.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], 'k--')

    plt.text(1, 2.1, 'Bad edges: ' + str(reds / 2), color="red", fontsize=20)
    
    if to_file:
        plt.savefig(figname)


def save_embedding(data_trans, path):
    with open(path + '_emb', 'w') as f:
        for x, y in data_trans:
            f.write(str(x) + ' ' + str(y) + '\n')


if __name__ == '__main__':
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'sammon'
    assert mode in ['sammon', 'mds']
    
    n, graph = load_graph(path)
    weights = calc_weights(n, graph)
    dists = calc_dists(n, graph, weights)
    
    if mode == 'sammon':
        data_trans = sammon(dists, inputdist='distance', init='random', display=0)[0]
    elif mode == 'mds':
        mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', n_init=10, max_iter=100000, eps=.00001)
        data_trans = mds.fit_transform(dists)
    
    data_trans = scale(data_trans)
    
    plot_a_thing(data_trans, graph, figname=path + '_' + mode + '.png')
    save_embedding(data_trans, path + '_' + mode)
