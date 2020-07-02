import networkx as nx
from node2vec import Node2Vec

def n2v():
    graph = nx.read_edgelist('../data/edge_nonHINE.txt',delimiter=' ',nodetype=int,data=(('weight',float),))

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=64, walk_length=100, num_walks=10, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    # model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format('../data/noneHINE.txt')

if __name__ == '__main__':
    n2v()