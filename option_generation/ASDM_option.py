import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
from options.graph.cover_time import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity


def AverageShortestOptions(G, P, k, delta = 1, subgoal=False):


    X = nx.to_networkx_graph(G)
    if not nx.is_connected(X):
        cs = list(nx.connected_components(X))
        for c_ in cs:
            if len(c_) > 1:
                c = c_
                break
        Xsub = X.subgraph(c)
        A = nx.to_numpy_matrix(Xsub)
        print('connected comp =', c)
    else:
        A = G.copy()

    options = []

    while len(options) < k:

        cities = []

        # pair wise distances in graph
        D = nx.floyd_warshall_numpy(nx.to_networkx_graph(A))

        # weight of each node pair
        W = np.zeros(A.shape)
        for pair in P:
            W[pair[0],pair[1]] += 1

        W = np.ones(A.shape)

        duplicates = []
        #for every node make duplicates
        # each element of form:
        # node that it is duplicate of, associated demand (weight), associated penalty
        for i in range(len(A)):
            for j in range(len(A)):
                if i == j:
                    continue
                duplicates.append([i, W[i,j], max(0, W[i,j] * (D[j,i] - 2 * delta))])
                duplicates.append([i, W[j,i], max(0, W[j,i] * (D[j,i] - 2 * delta))])


        S = [0, 1]
        def cost(a,b):
            def pick_city(a, exclude_set = []):
                best_city = None
                best_city_id = None
                best_city_connection_cost = - np.inf

                for i, city in enumerate(duplicates):
                    # ignore cities that have already been connected
                    if i in exclude_set:
                        continue

                    c = city[2] - D[a,city[0]]*city[1]
                    if c > best_city_connection_cost:
                        best_city_connection_cost = c
                        best_city = city
                        best_city_id = i

                assert(not best_city_id == None)
                return best_city, best_city_id

            city_a, city_a_id = pick_city(a)
            city_b, city_b_id = pick_city(b, [city_a])


            #add two costs for connection
            cost = D[a,city_a[0]]*city_a[1] + D[b,city_b[0]]*city_b[1]

            for i,city in enumerate(duplicates):
                if i in [city_a_id, city_b_id]:
                    continue
                cost += city[2] #add all other penalties

            #print(a,b,city_a,city_b,cost)
            return cost


        while True:

            best_a = S[0]
            best_b = S[1]
            min_cost = cost(best_a, best_b)

            for a in S:
                for b in range(len(A)):
                    if b in S:
                        continue
                    # loop over all a to remove from S and all b to add
                    # because S is size 2 we can simply use a as our element left in
                    # S after the other was removed.
                    c = cost(a, b)
                    if c < min_cost:
                        best_a = a
                        best_b = b
                        min_cost = c

            if S == [best_a, best_b]:
                break
            S = [best_a, best_b]

        option = (S[0], S[1])
        options.append(option)


        if subgoal:
            B = A.copy()
            B[:, option[1]] = 1
            B[option[1], :] = 1
        else:
            B = AddEdge(A, option[0], option[1])
        #update graph
        A = B

    return A, options

if __name__ == "__main__":
    N = 5
    #Gnx = nx.cycle_graph(N)
    Gnx = nx.path_graph(N)
    graph = nx.to_numpy_matrix(Gnx)

    P = np.array([np.random.permutation(np.arange(N))[:2] for i in range(10)])

    print('#'*10)
    for i in range(2):
        t = ComputeCoverTime(graph)
        print('Number of Options',i)
        print('CoverTime     ', t)
        lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
        print('lambda        ', lb)
        print()
        print(graph)
        graph, options = AverageShortestOptions(graph, P, 1)
        print(options)

    # proposedAugGraph, options, _, _ = FiedlerOptions(graph, 8)
    #
    # pGnx = nx.to_networkx_graph(proposedAugGraph)
    #
    # nx.draw_spectral(pGnx)
    # plt.savefig('drawing.pdf')
    #


    # t = ComputeCoverTime(graph)
    # print('CoverTime     ', t)
    # lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
    # print('lambda        ', lb)
    #
    # t3 = ComputeCoverTime(proposedAugGraph)
    # print('CoverTime Aug ', t3)
    # lb3 = nx.algebraic_connectivity(nx.to_networkx_graph(proposedAugGraph))
    # print('lambda        ', lb3)
