import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
from options.graph.cover_time import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity


"""
TODO: Generates too many options?? Why? How?

double check algo because probably wrong (need paper)

"""

def AverageShortestOptions_not_optimized(G, P, k, delta = 1, subgoal=False):

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

        # pair wise distances in graph
        D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
        D = np.zeros(A.shape)
        for source in D_dict:
            for target in range(len(A)):
                D[source[0],target] = source[1][target]

        print(D)
        print()

        # weight of each node pair
        W = np.zeros(A.shape)
        for pair in P:
            W[pair[0],pair[1]] += 1

        duplicates = []
        #for every node make duplicates
        # each element of form:
        # node that it is duplicate of, associated demand (weight), associated penalty
        for i in range(len(A)):
            for j in range(len(A)):
                if i == j:
                    continue
                duplicates.append([i, W[i,j], max(0, W[i,j] * (D[i,j] - 2 * delta))])
                duplicates.append([i, W[j,i], max(0, W[j,i] * (D[j,i] - 2 * delta))])



        def cost(a,b):
            cost = 0
            for city in duplicates:
                # choose the closest facility
                f = a if D[a,city[0]] < D[b,city[0]] else b
                #either connect and pay d * w or deny service and pay penalty
                cost += min(D[f,city[0]]*city[1],city[2])

                #print(city[0],min(D[f,city[0]]*city[1],city[2]))

            return cost

        S = [0, 1]
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
                    print(a,b,c)
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


def AverageShortestOptions_Probably_Wrong(G, Pairs, k, delta = 1):
    A = G.copy()

    options = []

    while len(options) < k:

        # pair wise distances in graph
        D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
        D = np.zeros(A.shape,dtype='int')
        for source in D_dict:
            for target in range(len(A)):
                D[source[0],target] = source[1][target]

        # weight of each node pair
        try:
            W = get_weight_matrix(len(A),Pairs)
        except:
            W = np.ones((len(A),len(A)),dtype='int')

        P_out = np.clip(W*(D-2*delta),0,None)
        P_in = np.clip(W.T*(D-2*delta),0,None)
        P = np.hstack((P_out,P_in))


        C = np.hstack((W,W.T))

        def cost(a,b):
            D_min = np.minimum(D[a],D[b])
            D_min = np.repeat(D_min[:,None].T,len(D_min),axis=0)
            D_min = np.hstack((D_min,D_min))
            M = C * D_min
            return np.sum(np.minimum(P,M))

        S = [0, 1]
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



        A[option[0],option[1]] = 1
        A[option[1],option[0]] = 1

    return A, options

def AverageShortestOptions(G, Pairs, k, delta = 1):
    A = G.copy()

    options = []

    # pair wise distances in graph
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    D = np.zeros(A.shape,dtype='int')
    for source in D_dict:
        for target in range(len(A)):
            D[source[0],target] = source[1][target]

    # weight of each node pair

    try:
        W = get_weight_matrix(len(A),Pairs)
    except:
        W = np.ones((len(A),len(A)),dtype='int')


    P = np.clip(W*(D-2*delta),0,None)



    #F is a list of facilities (indicies)
    def cost(S):
        cost = 0
        distances = np.amin(D[:,S],axis=1)
        for city in range(len(A)):
            W_scaled = W * distances[city]
            cost += np.sum(np.minimum(W_scaled[city], P[city]))
            cost += np.sum(np.minimum(W_scaled[:,city], P[:,city]))
        return cost


    S = list(range(k+1)) #need k+1 nodes for k edge star
    while True:
        min_cost = cost(S)
        best_set = S.copy()

        # loop over all a to remove from S and all b to add
        for a in S:
            for b in range(len(A)):
                if b in S:
                    continue
                S_ = S.copy()
                S_.remove(a)
                S_.append(b)

                c = cost(S_)
                if c < min_cost:
                    min_cost = c
                    best_set = S_

        if set(S) == set(best_set):
            break
        S = best_set

    for i in range(1,len(S)):
        option = (S[0], S[i])
        options.append(option)

        A[option[0],option[1]] = 1
        A[option[1],option[0]] = 1

    return A, options

def BruteOptions(G, P, k, delta = 1, subgoal=False):

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


        # weight of each node pair
        W = np.zeros(A.shape)
        for pair in P:
            W[pair[0],pair[1]] += 1

        def cost(i,j):
            A_ = A.copy()
            A_[i,j] = 1
            A_[j,i] = 1 #add option
            return average_shortest_distance(A_,W)

        option = (0,0)
        avg_shortest_distance = None
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                c = cost(i,j)
                if avg_shortest_distance == None:
                    avg_shortest_distance = c
                    continue

                if c < avg_shortest_distance:
                    option = (i,j)
                    avg_shortest_distance = c


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


def average_shortest_distance(A, W):
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    sum = 0
    for source in D_dict:
        for target in range(len(A)):
            sum += source[1][target] * W[source[0],target]
    return sum/np.sum(W)

def get_weight_matrix(N,P):
    W = np.zeros((N,N),dtype='int')
    for pair in P:
        W[pair[0],pair[1]] += 1
    return W

def get_random_graph(N, p=.5, directed=False):
    A = np.zeros((N,N),dtype='int')
    if directed:
        for i in range(N):
            for j in range(N):
                A[i,j] = 1 if np.random.rand() < p else 0
    else:
        for i in range(N-1):
            for j in range(i+1,N):
                A[i,j] = 1 if np.random.rand() < p else 0
                A[j,i] = A[i,j]
    return A


def compare_to_brute_exp(graph, P, W, num_options, show_graphs=False):
    avg_dist = average_shortest_distance(graph,W)
    ASPDM_list = [avg_dist]
    Brute_list = [avg_dist]

    graph_alg_orig = graph.copy()
    graph_brute = graph.copy()

    for i in range(num_options):
        print(i,"/", num_options,"\t\t",end="\r")
        graph_alg, options_alg = AverageShortestOptions(graph_alg_orig, P, i)
        avg_dist_alg = average_shortest_distance(graph_alg,W)

        graph_brute, options_brute = BruteOptions(graph_brute, P, 1)
        avg_dist_brute = average_shortest_distance(graph_brute,W)

        if show_graphs:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            nx.draw(nx.to_networkx_graph(graph),ax=ax1)
            nx.draw(nx.to_networkx_graph(graph_alg),ax=ax2)
            nx.draw(nx.to_networkx_graph(graph_brute),ax=ax3)
            plt.show()

        if ASPDM_list[-1] == avg_dist_alg and Brute_list[-1] == avg_dist_brute:
            ASPDM_list.append(avg_dist_alg)
            Brute_list.append(avg_dist_brute)
            break
        ASPDM_list.append(avg_dist_alg)
        Brute_list.append(avg_dist_brute)

    return ASPDM_list, Brute_list, graph_alg, graph_brute

def compare_to_brute_multiple_exp(N, num_options, num_trials, show_graphs = False):

    P = np.array([np.random.permutation(np.arange(N))[:2] for i in range(N*N)])
    # P = []
    # [[P.append((i,j)) for i in range(N)] for j in range(N)]
    # P = np.array(P)
    W = get_weight_matrix(N,P)


    A = []
    B = []

    for i in range(num_trials):
        print("trial ",(i+1)," of ",num_trials)
        #Gnx = nx.barabasi_albert_graph(n=N,m=1)
        Gnx = nx.fast_gnp_random_graph(n=N,p=.2)

        graph = nx.to_numpy_matrix(Gnx).astype(dtype='int')

        ASPDM_list, Brute_list, graph_alg, graph_brute = compare_to_brute_exp(graph,P,W,num_options)
        A.append(ASPDM_list)
        B.append(Brute_list)
        if show_graphs:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            nx.draw(nx.to_networkx_graph(graph),ax=ax1)
            nx.draw(nx.to_networkx_graph(graph_alg),ax=ax2)
            nx.draw(nx.to_networkx_graph(graph_brute),ax=ax3)
            plt.show()


    # A = np.array(A)
    # B = np.array(B)
    # X = np.arange(A.shape[1])
    # plt.fill_between(X,np.max(A,axis=0),np.min(A,axis=0),alpha=.5)
    # plt.fill_between(X,np.max(B,axis=0),np.min(B,axis=0),alpha=.5)
    fig = plt.figure()
    [plt.plot(data,color='orange') for data in A]
    [plt.plot(data,color='blue') for data in B]
    plt.xlabel('number of options')
    plt.ylabel('average shortest distance')
    plt.title('ASPDM-orange  Brute-Blue')
    fig.show()
    plt.show()


def base_experiment_compare(graph, P, W):
    print('#'*10)
    for i in range(10):
        t = ComputeCoverTime(graph)
        print('Number of Options',i)
        print('CoverTime     ', t)
        lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
        print('lambda        ', lb)
        print()

        avg_dist = average_shortest_distance(graph,W)

        graph_alg, options_alg = AverageShortestOptions(graph, P, 1)
        lb_alg = nx.algebraic_connectivity(nx.to_networkx_graph(graph_alg))
        avg_dist_alg = average_shortest_distance(graph_alg,W)


        graph_brute, options_brute = BruteOptions(graph, P, 1)
        lb_brute = nx.algebraic_connectivity(nx.to_networkx_graph(graph_brute))
        avg_dist_brute = average_shortest_distance(graph_brute,W)

        print('avg_shortest_dist: ',"alg:", avg_dist_alg,"brute:",avg_dist_brute)
        #print('lambda: ',"alg:", lb_alg,"brute:",lb_brute)
        print(options_alg,options_brute)

        graph = graph_alg


def convert_file_to_graph_and_P(file):
    f = open(file,'r')
    text = f.read()
    lines = text.split('\n')

    N = len(lines[0])
    square_grid_size = N

    grid = np.zeros((N,N))
    grid_init = np.zeros((N,N))
    grid_goal = np.zeros((N,N))

    for i,l in enumerate(lines):
        if i >= N:
            continue
        for j,c in enumerate(l):
            if c == '-':
                continue
            if c == 'w':
                grid[i,j] = 1
            if c == 'a':
                grid_init[i,j] = 1
            if c == 'g':
                grid_goal[i,j] = 1
            if c == 'b':
                grid_init[i,j] = 1
                grid_goal[i,j] = 1


    A = np.zeros((N*N,N*N))
    def dual_link(r_1,c_1,r_2,c_2):
        A[r_1*N+c_1,r_2*N+c_2] = 1
        A[r_2*N+c_2,r_1*N+c_1] = 1

    for i in range(N):
        for j in range(N):
            if not i == 0:
                #check walls
                if not (grid[i,j] or grid[i-1,j]):
                    dual_link(i,j,i-1,j)
            if not j == 0:
                #check walls
                if not (grid[i,j] or grid[i,j-1]):
                    dual_link(i,j,i,j-1)

    P_start = []
    P_end = []
    for i in range(N):
        for j in range(N):
            if grid_init[i,j] == 1:
                P_start.append(i*N+j)
            if grid_goal[i,j] == 1:
                P_end.append(i*N+j)

    P = []
    for s in P_start:
        for e in P_end:
            P.append((s,e))

    return A,P, square_grid_size


def fix_graph_and_P(graph,P):
    i = 0
    while i < len(graph):
        if np.sum(graph[i]) == 0:
            graph = np.delete(graph,i,0)
            graph = np.delete(graph,i,1)
            j = 0
            while j < len(P):
                if P[j][0] == i or P[j][1] == i:
                    del P[j]
                else:
                    if P[j][0] > i:
                        P[j] = (P[j][0] - 1,P[j][1])
                    if P[j][1] > i:
                        P[j] = (P[j][0], P[j][1] - 1)
                    j += 1
        else:
            i += 1
    return graph, P

def test_domain():
    domains = ['4x4grid','4x4grid-NEW','9x9grid-NEW', 'fourroom']
    path = 'options/tasks/'
    for d in domains:
        fname = path + d + '.txt'
        graph, P, size = convert_file_to_graph_and_P(fname)
        graph,P = fix_graph_and_P(graph, P)
        # P = []
        # [[P.append((i,j)) for i in range(len(graph))] for j in range(len(graph))]
        # P = np.array(P)
        new_graph, options = AverageShortestOptions(graph, P, 10)
        for option in options:
            def conv(x):
                return int(x/size), x%size
            print(conv(option[0]),conv(option[1]))

        print("#"*10)

if __name__ == "__main__":
    N = 10
    #Gnx = nx.cycle_graph(N)
    Gnx = nx.path_graph(N)

    #Gnx = nx.random_regular_graph(d=2, n=N)
    #Gnx = nx.barabasi_albert_graph(n=N,m=1)
    graph = nx.to_numpy_matrix(Gnx).astype(dtype='int')

    #P = np.array([np.random.permutation(np.arange(N))[:2] for i in range(N)])
    P = []
    [[P.append((i,j)) for i in range(N)] for j in range(N)]
    P = np.array(P)
    W = get_weight_matrix(N,P)

    #test_domain()

    compare_to_brute_multiple_exp(75, 10 ,3)

    #A, options = AverageShortestOptions(graph,P,3)


    #base_experiment_compare(graph,P,W)
    # ASPDM_list,Brute_list,_,_ = compare_to_brute_exp(graph,P,W,3, True)
    # plt.plot(ASPDM_list,label="ASPDM")
    # plt.plot(Brute_list,label="Brute")
    # plt.legend()
    # plt.show()
    # plt.pause(1000000000000000000)
