`sammon.py` comes from https://github.com/tompollard/sammon

# Usage

Run `python mds_test.py path_to_graph mode`.

`mode` if optional, should be `mds` or `sammon`. Default is `sammon`.

Example: `python mds_test.py graphs/g1 sammon`.

Graph should be represented by a text file. First line contains one integer `n` (number of vertices), 
`(i+1)`th line contains neighbors of a vertex `i`. Vertices are numered from `1` to `n`. 
Examples are in the `graphs` folder. DO NOT put a new line at the end of the file, it will be interpreted as a vertex with no neighbors.

Script produces a text file with vertex embeddings and a plot, both are saved in `path_to_graph` directory. Dotted lines represent Voronoi borders. 
"Bad edges" is the number of edges in the graph that cross an area they don't belong to.



About graph embeddings:

[1] http://www.stat.yale.edu/~lc436/papers/JCGS-mds.pdf

[2] http://www.graphviz.org/Documentation/GKN04.pdf

[3] https://en.wikipedia.org/wiki/Sammon_mapping

[4] https://en.wikipedia.org/wiki/Multidimensional_scaling

[5] https://www.codeproject.com/Articles/43123/Sammon-Projection
