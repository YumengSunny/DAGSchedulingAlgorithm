import copy


class Graph:
    def find_all_paths(self, G_, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to end_vertex in graph """
        graph = G_
        path = path + [start_vertex]

        if start_vertex == end_vertex:
            return [path]

        if start_vertex not in graph:
            return []

        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                # solve this in a recursive way
                extended_paths = self.find_all_paths(G_,
                                                     vertex,
                                                     end_vertex,
                                                     path)
                for p in extended_paths:
                    paths.append(p)

        return paths

    def get_subpath_between(self, P, a, b):
        ''' Get a subpath in P where the starting node is a, and the end node is b
        '''
        for idx, v in enumerate(P):
            if v == a:
                idx_a = idx + 1
            if v == b:
                idx_b = idx

        return P[idx_a:idx_b]

    def find_longest_path_dfs(self, G_, start_vertex, end_vertex, weights):
        """ find the longest path with depth first search """

        # find all paths
        paths = self.find_all_paths(G_, start_vertex, end_vertex)

        # search for the critical path
        costs = []
        for path in paths:
            cost = 0
            for v in path:
                cost = cost + weights[v - 1]
            costs.append(cost)

        (m, i) = max((v, i) for i, v in enumerate(costs))

        return m, paths[i]

    def find_associative_nodes(self, G_, candidate_nodes, critical_path):
        """ find associative nodes that could block the critical path """
        associated_nodes = []

        # if there is any route from A -> B, then B is associated with A
        for S in candidate_nodes:
            for E in critical_path:
                if self.find_all_paths(G_, S, E):
                    if S not in associated_nodes:  associated_nodes.append(S)

        return associated_nodes

    def find_predecesor(self, G, node):
        precs = []
        for key in G:
            if node in G[key]:
                precs.append(key)

        return precs

    def find_successor(self, G, node):
        if node in G:
            return G[node]
        else:
            return []

    def find_ancestors(self, G, node, path=[]):
        G_copy = copy.deepcopy(G)
        return self.find_ancestors_iter(G_copy, node, path=[])

    def find_ancestors_iter(self, G, node, path=[]):
        v_source = self.source(G)
        a = []

        # print(node)
        # print(path)

        if node == v_source:
            return []

        predecesor_nodes = self.find_predecesor(G, node)

        if not predecesor_nodes:
            return []

        self.remove_nodes(G, [node])

        for v in predecesor_nodes:
            path.append(v)
            self.find_ancestors_iter(G, v, path)

        # print(path)
        for i in path:
            if i not in a:
                a.append(i)

        a.sort()

        return a

    def find_descendants(self, G, node, path=[]):
        G_copy = copy.deepcopy(G)
        return self.find_descendants_iter(G_copy, node, path=[])

    def find_descendants_iter(self, G, node, path=[]):
        v_sink = self.sink(G)
        a = []

        # print(node)

        if node == v_sink:
            return []

        successor_nodes = self.find_successor(G, node)

        if not successor_nodes:
            return []

        self.remove_nodes(G, [node])

        for v in successor_nodes:
            path.append(v)
            self.find_descendants_iter(G, v, path)

        # print(path)
        for i in path:
            if i not in a:
                a.append(i)

        a.sort()

        return a

    def remove_nodes(self, G, nodes):
        ''' remove nodes (and its related edges) from a graph
        '''
        for key, value in G.copy().items():
            if key in nodes:
                G.pop(key)
            else:
                for v in value:
                    if v in nodes:
                        value.remove(v)

    def sink(self, G):
        k = G.keys()
        return max(k)

    def source(self, G):
        return 1

    if __name__ == "__main__":
        G = {1: [2, 3, 4], 2: [5, 6], 3: [7, 8], 4: [11], 5: [9], 6: [9], 7: [10], 8: [10], 9: [11], 10: [11], 11: []}
        C = [1, 5, 6, 7, 3, 6, 4, 2, 9, 8, 1]

        # print(find_all_paths(G, 1, 11))
        # print(find_longest_path_dfs(G, 1, 11, C))

        # print(find_predecesor(G, 10))
        # print(find_successor(G, 6))

        print(find_descendants(G, 2))
        print(find_ancestors(G, 2))
