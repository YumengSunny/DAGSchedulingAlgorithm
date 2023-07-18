import codecs
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
from operator import itemgetter
import time, datetime

from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import chardet
import copy

from graph import Graph
from bisect import bisect_left

TASKSET_TO_EVALUATE = 1000
A_VERY_LARGE_NUMBER = 1000000

dag_base_folder = "data/data-generic/"
L_ratio = -1


class EOPA_Algorithm:

    e = A_VERY_LARGE_NUMBER  # this has to be global, or be passed by reference

    def __init__(self):
        self.graph = Graph()

    def print_debug(self, *args, **kw):
        # print(args)
        pass

    def load_task(self, task_idx):
        # << load DAG task <<
        dag_task_file = dag_base_folder + "Tau_{:d}.gpickle".format(task_idx)

        # task is saved as NetworkX gpickle format
        # G = nx.read_gpickle(dag_task_file)

        with open(dag_task_file, 'rb') as f:
            G = pickle.load(f)

        # formulate the graph list
        G_dict = {}
        C_dict = {}
        V_array = []
        max_key = 0
        for u, v, weight in G.edges(data='label'):
            if u not in G_dict:
                G_dict[u] = [v]
            else:
                G_dict[u].append(v)

            if v > max_key:
                max_key = v

            if u not in V_array:
                V_array.append(u)
            if v not in V_array:
                V_array.append(v)

            C_dict[u] = weight
        C_dict[max_key] = 1

        G_dict[max_key] = []

        # formulate the c list (c[0] is c for v1!!)
        C_array = []
        for key in sorted(C_dict):
            C_array.append(C_dict[key])

        V_array.sort()
        L, lamda = self.graph.find_longest_path_dfs(G_dict, V_array[0], V_array[-1], C_array)
        W = sum(C_array)

        VN_array = V_array.copy()

        for i in lamda:
            if i in VN_array:
                VN_array.remove(i)

        # scale L (the length of the critical path)
        if L_ratio != -1:
            # print("Old L ratio:", L * 1.0 / W)

            L_old = L
            vol_old = W - L

            L_new = L_ratio * W
            vol_new = (1 - L_ratio) * W

            L_multiplier = L_new / L_old

            L = 0
            for i in lamda:
                C_dict[i] = max(round(C_dict[i] * L_multiplier), 1)
                L = L + C_dict[i]

            vol_multiplier = vol_new / vol_old
            for i in VN_array:
                C_dict[i] = max(round(C_dict[i] * vol_multiplier), 1)

            # formulate the c list (c[0] is c for v1!!)
            C_array = []
            for key in sorted(C_dict):
                C_array.append(C_dict[key])

            # check critical path!!!!
            L_prime, lamda_prime = self.graph.find_longest_path_dfs(G_dict, V_array[0], V_array[-1], C_array)

            if lamda_prime != lamda or L_prime != L:
                raise Exception("Lambda does not hold!")

        # >> end of load DAG task >>
        return G_dict, C_dict, C_array, lamda, VN_array, L, W

    def load_taskset_metadata(self, dag_base_folder):
        number_of_tasks_in_set = 10
        Taskset = {}

        aTau = []
        aT = []
        aC = []

        for task_idx in range(number_of_tasks_in_set):
            # << load DAG task <<
            dag_task_file = dag_base_folder + "/Tau_{:d}.gpickle".format(task_idx)

            # task is saved as NetworkX gpickle format
            G = nx.read_gpickle(dag_task_file)

            Ti = G.graph["T"]
            Wi = G.graph["W"]
            Ui = G.graph["U"]

            ############################################################################
            # assign priorities according to RMPO / DMPO
            idx = bisect_left(aT, Ti)

            aTau.insert(idx, task_idx)
            aT.insert(idx, Ti)
            aC.insert(idx, Wi)

        for i, task_idx in enumerate(aTau):
            Taskset[i] = {}
            Taskset[i]["tau"] = aTau[i]
            Taskset[i]["T"] = aT[i]
            Taskset[i]["C"] = aC[i]

        return Taskset

    def remove_nodes_in_list(self, nodes, nodes_to_remove):
        for i in nodes.copy():
            if i in nodes_to_remove:
                nodes.remove(i)

    def get_nodes_volume(self, nodes, C_list):
        ''' sum of workload
        nodes can be individual nodes or from a path
        '''
        volume = 0

        for i in nodes:
            volume = volume + C_list[i]

        return volume

    def remove_nodes_in_graph(self, G, nodes):
        ''' remove nodes (and its related edges) from a graph
        '''
        for key, value in G.copy().items():
            if key in nodes:
                G.pop(key)
            else:
                for v in value:
                    if v in nodes:
                        value.remove(v)

    ################################################################################
    ################################################################################
    def find_concurrent_nodes(self, G, node):
        ''' find concurrent nodes
        '''
        ancs = self.graph.find_ancestors(G, node, path=[])
        decs = self.graph.find_descendants(G, node, path=[])

        V = list(G.keys())
        V.remove(node)
        self.remove_nodes_in_list(V, ancs)
        self.remove_nodes_in_list(V, decs)

        return V

    def test_parallelism(self, G, node, n):
        ''' test if delta(node) < n
        '''

        # start to iterative delta
        delta = 0
        node_left = node.copy()

        while node_left:
            # this search is limited to node,
            # so all keys and values that do not contain node_left will be removed
            G_new = copy.deepcopy(G)

            for key, value in G_new.copy().items():
                if key not in node_left:
                    del G_new[key]
                else:
                    value_new = value.copy()
                    for j in value:
                        if j not in node_left:
                            value_new.remove(j)

                    G_new[key] = value_new

            delta = delta + 1
            if delta >= n:
                # print_debug("PARALLISM: False")
                return False

            finished = False
            while not finished:
                for ve in node_left.copy():
                    if not self.graph.find_predecesor(G_new, ve):
                        node_left.remove(ve)
                        finished = True
                        break

            suc_ve = self.graph.find_successor(G_new, ve)
            while suc_ve:
                suc_ve_first = suc_ve[0]
                if suc_ve_first in node_left:
                    node_left.remove(suc_ve_first)
                suc_ve = self.graph.find_successor(G_new, suc_ve_first)

        # print_debug("PARALLISM: True")
        return True

    def find_providers_consumers(self, G_dict, lamda, VN_array):
        """ Find providers and consumers
        """
        providers = []
        consumers = []

        new_provider = []
        nc_nodes_left = VN_array.copy()
        for key, i in enumerate(lamda):
            if new_provider == []:
                new_provider = [i]

            if (key + 1) < len(lamda):
                pre_nodes = self.graph.find_predecesor(G_dict, lamda[key + 1])

                # print_debug("Checking: ", i, "Pre: ", pre_nodes)
                if pre_nodes == [i]:
                    new_provider.append(lamda[key + 1])
                else:
                    # print_debug("New provider:", new_provider)
                    providers.append(new_provider)
                    new_provider = []

                    # remove critical nodes
                    self.remove_nodes_in_list(pre_nodes, lamda)

                    # find all consumers
                    new_consumer = []
                    for pre_node in pre_nodes:
                        # add this pre-node first
                        if pre_node in nc_nodes_left:
                            new_consumer.append(pre_node)

                        # find any ancestor of this pre-node
                        ancestors_of_node = self.graph.find_ancestors(G_dict, pre_node, path=[])

                        self.remove_nodes_in_list(ancestors_of_node, lamda)
                        if ancestors_of_node:
                            for anc_v in ancestors_of_node:
                                if anc_v not in new_consumer and anc_v in nc_nodes_left:  new_consumer.append(anc_v)

                            # print_debug(ancestors_of_node)

                    new_consumer.sort()
                    consumers.append(new_consumer)

                    # remove from NC list
                    for i_nc in new_consumer:
                        nc_nodes_left.remove(i_nc)

                    new_consumer = []
            else:
                # the last node needs special care as it has no successors
                # print_debug("New provider:", new_provider)
                providers.append(new_provider)

                # find all consumers (all the left nc nodes)
                nc_nodes_left.sort()
                consumers.append(nc_nodes_left)

        return providers, consumers

    def find_G_theta_i_star(self, G, providers, consumers, i):
        G_theta_i_star = []

        # collect all consumer nodes in the following providers
        number_of_providers = len(providers)
        if i == number_of_providers - 1:
            # skip as this is the last provider
            return []

        theta_i = consumers[i]

        all_later_consumer_nodes = []
        for l in range(i + 1, number_of_providers):
            for k in consumers[l]:
                all_later_consumer_nodes.append(k)

        for theta_ij in theta_i:
            con_ij = self.find_concurrent_nodes(G, theta_ij)

            for k in con_ij:
                if k in all_later_consumer_nodes:
                    if k not in G_theta_i_star:
                        G_theta_i_star.append(k)

        return G_theta_i_star

    def EOPA(self, task_idx, m):
        """ Response time analysis using alpha_beta
        """
        # --------------------------------------------------------------------------
        # I. load the DAG task
        G_dict, C_dict, C_array, lamda, VN_array, L, W = self.load_task(task_idx)

        # --------------------------------------------------------------------------
        # II. providers and consumers
        # iterative all critical nodes
        # after this, all provides and consumers will be collected
        providers, consumers = self.find_providers_consumers(G_dict, lamda, VN_array)
        self.print_debug("Providers:", providers)
        self.print_debug("Consumers:", consumers)

        # --------------------------------------------------------------------------
        # III. calculate the finish times of each provider, and the consumers within
        f_dict = {}  # the set of all finish times
        I_dict = {}  # interference workload
        I_e_dict = {}  # interference workload (for EO)
        R_i_minus_one = 0  # the response time of the previous provider theta^*_(i - 1)

        alpha_arr = []
        beta_arr = []

        # if EOPA:
        Prio = self.Eligiblity_Ordering_PA(G_dict, C_dict)
        self.print_debug("Prioirties", Prio)
        # elif TPDS:
        #   Prio = self.TPDS_Ordering_PA(G_dict, C_dict)
        # reverse the order
        #  for i in Prio:
        #     Prio[i] = A_VERY_LARGE_NUMBER - Prio[i]
        # self.print_debug("Prioirties", Prio)

        # ==========================================================================
        # iteratives all providers (first time, to get all finish times)
        for i, theta_i_star in enumerate(providers):
            self.print_debug("- - - - - - - - - - - - - - - - - - - -")
            self.print_debug("theta(*)", i, ":", theta_i_star)
            self.print_debug("theta", i, ":", consumers[i])

            # get the finish time of all provider nodes (in provider i)
            for _, provi_i in enumerate(theta_i_star):
                # (skipped) sort with topoligical order, skipped because guaranteed by the generator
                # find the max finish time of all its predecences
                f_provider_i_pre_max = 0

                predecsor_of_ij = self.graph.find_predecesor(G_dict, provi_i)
                for pre_i in predecsor_of_ij:
                    if f_dict[pre_i] > f_provider_i_pre_max:
                        f_provider_i_pre_max = f_dict[pre_i]

                f_i = C_dict[provi_i] + f_provider_i_pre_max

                f_dict[provi_i] = f_i
                f_theta_i_star = f_i  # finish time of theta_i_star. every loop refreshes this

            # iteratives all consumers
            # (skipped) topoligical order, skipped because guaranteed by the generator
            # note: consumer can be empty
            theta_i = consumers[i]
            f_v_j_max = 0;
            f_v_j_max_idx = -1
            for _, theta_ij in enumerate(theta_i):
                # the interference term
                con_nc_ij = self.find_concurrent_nodes(G_dict, theta_ij)
                self.remove_nodes_in_list(con_nc_ij, lamda)
                if self.test_parallelism(G_dict, con_nc_ij, m - 1):
                    # sufficient parallism
                    interference = 0
                    I_dict[theta_ij] = []
                    I_e_dict[theta_ij] = []
                else:
                    # not enough cores
                    # start to search interference nodes >>>
                    # concurrent nodes of ij. Can be empty.
                    con_ij = self.find_concurrent_nodes(G_dict, theta_ij)
                    # print_debug("Con nodes:", con_ij)

                    int_ij = con_ij.copy()
                    self.remove_nodes_in_list(int_ij, lamda)

                    ans_ij = self.graph.find_ancestors(G_dict, theta_ij, path=[])
                    self.remove_nodes_in_list(ans_ij, lamda)
                    # print_debug("Ans nodes:", ans_ij)

                    for ij in ans_ij:
                        ans_int = I_dict[ij]
                        for ijij in ans_int:
                            if ijij in int_ij:
                                int_ij.remove(ijij)
                    # print_debug("Int nodes:", int_ij)

                    I_dict[theta_ij] = int_ij

                    # if EOPA or TPDS:
                    # for EOPA, only the (m - 1) longest lower priority interference node is kept
                    int_ij_EO = []
                    int_ij_EO_less_candidates = []
                    int_ij_EO_less_candidates_C = []

                    if int_ij:
                        for int_ij_k in int_ij:
                            if Prio[int_ij_k] > Prio[theta_ij]:
                                # E_k > E_i, add with confidence
                                int_ij_EO.append(int_ij_k)
                            else:
                                # E_k < E_i, put into a list and later will only get longest m - 1
                                int_ij_EO_less_candidates.append(int_ij_k)
                                int_ij_EO_less_candidates_C.append(C_dict[int_ij_k])

                        # sort nodes by C (if it exists), and append (m-1) longest to int_ij_EO
                        if int_ij_EO_less_candidates:
                            list_of_less_EO_nodes_C = int_ij_EO_less_candidates_C
                            indices, _ = zip(
                                *sorted(enumerate(list_of_less_EO_nodes_C), key=itemgetter(1), reverse=True))
                            # indices = [i[0] for i in sorted(enumerate(list_of_less_EO_nodes_C), key=lambda x:x[1])]
                            int_ij_EO_less_candidates_sorted = []

                            for idx_ in range(len(list_of_less_EO_nodes_C)):
                                int_ij_EO_less_candidates_sorted.append(int_ij_EO_less_candidates[indices[idx_]])

                            # adding (m - 1) lower EO nodes
                            for xxx in range(1, m):
                                if len(int_ij_EO_less_candidates) >= xxx:
                                    int_ij_EO.append(int_ij_EO_less_candidates_sorted[xxx - 1])

                        int_ij = int_ij_EO.copy()

                    I_e_dict[theta_ij] = int_ij
                    # >>> end of searching interference nodes

                    int_c_sum = sum(C_dict[ij] for ij in int_ij)
                    interference = math.ceil(1.0 / (m - 1) * int_c_sum)

                # find the max finish time of all its predecences
                f_ij_pre_max = 0

                predecsor_of_ij = self.graph.find_predecesor(G_dict, theta_ij)
                for pre_i in predecsor_of_ij:
                    if f_dict[pre_i] > f_ij_pre_max:
                        f_ij_pre_max = f_dict[pre_i]

                # calculate the finish time
                f_ij = C_dict[theta_ij] + f_ij_pre_max + interference
                f_dict[theta_ij] = f_ij
                self.print_debug("f_theta({}) : {}".format(theta_ij, f_dict[theta_ij]))

        # ==========================================================================
        # iteratives all providers (2nd time)
        # the finish times of provider nodes have to be calculated again to get f_theta_i_star
        # the finish times of consumer nodes have to be calculated again to get lambda_ve
        for i, theta_i_star in enumerate(providers):
            self.print_debug("- - - - - - - - - - - - - - - - - - - -")
            self.print_debug("theta(*)", i, ":", theta_i_star)
            self.print_debug("theta", i, ":", consumers[i])

            # get the finish time of all provider nodes (in provider i)
            for _, provi_i in enumerate(theta_i_star):
                # (skipped) sort with topoligical order, skipped because guaranteed by the generator
                # find the max finish time of all its predecences
                f_provider_i_pre_max = 0

                predecsor_of_ij = self.graph.find_predecesor(G_dict, provi_i)
                for pre_i in predecsor_of_ij:
                    if f_dict[pre_i] > f_provider_i_pre_max:
                        f_provider_i_pre_max = f_dict[pre_i]

                f_i = C_dict[provi_i] + f_provider_i_pre_max

                f_dict[provi_i] = f_i
                f_theta_i_star = f_i  # finish time of theta_i_star. every loop refreshes this

            self.print_debug("finish time of theta(*)", f_theta_i_star)

            # iteratives all consumers
            # (skipped) topoligical order, skipped because guaranteed by the generator
            # note: consumer can be empty
            theta_i = consumers[i]
            f_v_j_max = 0;
            f_v_j_max_idx = -1
            for _, theta_ij in enumerate(theta_i):
                self.print_debug(theta_ij, ":", C_dict[theta_ij])

                # the interference term
                con_nc_ij = self.find_concurrent_nodes(G_dict, theta_ij)
                self.remove_nodes_in_list(con_nc_ij, lamda)
                if self.test_parallelism(G_dict, con_nc_ij, m - 1):
                    # sufficient parallism
                    interference = 0
                    I_dict[theta_ij] = []
                    I_e_dict[theta_ij] = []
                else:
                    # not enough cores
                    # start to search interference nodes >>>
                    # concurrent nodes of ij. Can be empty.
                    con_ij = self.find_concurrent_nodes(G_dict, theta_ij)
                    # print_debug("Con nodes:", con_ij)

                    int_ij = con_ij.copy()
                    self.remove_nodes_in_list(int_ij, lamda)

                    ans_ij = self.graph.find_ancestors(G_dict, theta_ij, path=[])
                    self.remove_nodes_in_list(ans_ij, lamda)
                    # print_debug("Ans nodes:", ans_ij)

                    for ij in ans_ij:
                        ans_int = I_dict[ij]
                        for ijij in ans_int:
                            if ijij in int_ij:
                                int_ij.remove(ijij)
                    # print_debug("Int nodes:", int_ij)

                    I_dict[theta_ij] = int_ij

                    # if EOPA or TPDS:
                    # for EOPA, only the (m - 1) longest lower priority interference node is kept
                    int_ij_EO = []
                    int_ij_EO_less_candidates = []
                    int_ij_EO_less_candidates_C = []

                    if int_ij:
                        for int_ij_k in int_ij:
                            if Prio[int_ij_k] > Prio[theta_ij]:
                                # E_k > E_i, add with confidence
                                int_ij_EO.append(int_ij_k)
                            else:
                                # E_k < E_i, put into a list and later will only get longest m - 1
                                int_ij_EO_less_candidates.append(int_ij_k)
                                int_ij_EO_less_candidates_C.append(C_dict[int_ij_k])

                        # sort nodes by C (if it exists), and append (m-1) longest to int_ij_EO
                        if int_ij_EO_less_candidates:
                            list_of_less_EO_nodes_C = int_ij_EO_less_candidates_C
                            indices, _ = zip(
                                *sorted(enumerate(list_of_less_EO_nodes_C), key=itemgetter(1), reverse=True))
                            # indices = [i[0] for i in sorted(enumerate(list_of_less_EO_nodes_C), key=lambda x:x[1])]
                            int_ij_EO_less_candidates_sorted = []

                            for idx_ in range(len(list_of_less_EO_nodes_C)):
                                int_ij_EO_less_candidates_sorted.append(int_ij_EO_less_candidates[indices[idx_]])

                            # adding (m - 1) lower EO nodes
                            for xxx in range(1, m):
                                if len(int_ij_EO_less_candidates) >= xxx:
                                    int_ij_EO.append(int_ij_EO_less_candidates_sorted[xxx - 1])

                        int_ij = int_ij_EO.copy()

                    I_e_dict[theta_ij] = int_ij
                    # >>> end of searching interference nodes

                    int_c_sum = sum(C_dict[ij] for ij in int_ij)
                    interference = math.ceil(1.0 / (m - 1) * int_c_sum)

                # find the max finish time of all its predecences
                f_ij_pre_max = 0

                predecsor_of_ij = self.graph.find_predecesor(G_dict, theta_ij)
                for pre_i in predecsor_of_ij:
                    if f_dict[pre_i] > f_ij_pre_max:
                        f_ij_pre_max = f_dict[pre_i]

                # calculate the finish time
                f_ij = C_dict[theta_ij] + f_ij_pre_max + interference
                f_dict[theta_ij] = f_ij
                self.print_debug("f_theta({}) : {}".format(theta_ij, f_dict[theta_ij]))

                # find max(f_vj)
                if f_ij > f_v_j_max:
                    f_v_j_max = f_ij
                    f_v_j_max_idx = theta_ij

            # --------------------------------------------------------------------------
            # start to calculate the response time of provider i
            Wi_nc = sum(C_dict[ij] for ij in theta_i)
            Li = sum(C_dict[ij] for ij in theta_i_star)
            Wi = Wi_nc + Li

            # if not EOPA and not TPDS:
            #    # G(theta_i^*) needs to be added for random
            # Wi and Wi_nc will be updated
            #    G_theta_i_star = self.find_G_theta_i_star(G_dict, providers, consumers, i)

            #    Wi_G = sum(C_dict[ij] for ij in G_theta_i_star)
            #    Wi_nc = Wi_nc + Wi_G
            #    Wi = Wi_nc + Li

            # --------------------------------------------------------------------------
            # IV. bound alpha and beta
            # For Case A (has no delay to the critical path):
            if f_theta_i_star >= f_v_j_max:
                self.print_debug("** Case A **")
                alpha_i = Wi_nc
                beta_i = 0
            # For Case B (has delay to the critical path):
            else:
                self.print_debug("** Case B **")
                # search for lamda_ve
                ve = f_v_j_max_idx  # end node & backward search
                lamda_ve = [ve]
                len_lamda_ve = 0

                while True:
                    # find pre of ve
                    pre_of_ve = self.graph.find_predecesor(G_dict, ve)  # pre_of_ve can be empty!

                    # only care about those within this provider
                    for ij in pre_of_ve.copy():
                        if ij not in theta_i:
                            pre_of_ve.remove(ij)

                    if pre_of_ve:
                        # calculate the finish times, and the maximum
                        f_pre_of_ve = []
                        for ij in pre_of_ve:
                            f_pre_of_ve.append(f_dict[ij])

                        max_value = max(f_pre_of_ve)
                        max_index = f_pre_of_ve.index(max_value)

                        if max_value > f_theta_i_star:
                            ve = pre_of_ve[max_index]
                            lamda_ve.append(ve)
                        else:
                            break
                    else:
                        break

                # calculate accmulative intererence
                for ve_i in lamda_ve:
                    if f_dict[ve_i] - C_dict[ve_i] >= f_theta_i_star:
                        len_lamda_ve = len_lamda_ve + C_dict[ve_i]
                    else:
                        len_lamda_ve = len_lamda_ve + max((f_dict[ve_i] - f_theta_i_star), 0)

                self.print_debug("lamda_ve:", lamda_ve, "len:", len_lamda_ve)

                # beta_i
                beta_i = len_lamda_ve

                # alpha (a): find alpha by estimation of finish times
                alpha_hat_class_a = []
                alpha_hat_class_b = []

                for _, theta_ij in enumerate(theta_i):
                    if f_dict[theta_ij] <= f_theta_i_star:
                        if theta_ij not in alpha_hat_class_a:
                            alpha_hat_class_a.append(theta_ij)
                    elif f_dict[theta_ij] < f_theta_i_star + C_dict[theta_ij]:
                        if theta_ij not in alpha_hat_class_b:
                            alpha_hat_class_b.append(theta_ij)
                    else:
                        pass

                # if not EOPA and not TPDS:
                # for random, the alpha_i is different
                #    for _, theta_ij in enumerate(G_theta_i_star):
                #        if f_dict[theta_ij] <= f_theta_i_star:
                #            if theta_ij not in alpha_hat_class_a:
                #                alpha_hat_class_a.append(theta_ij)
                #        elif f_dict[theta_ij] < f_theta_i_star + C_dict[theta_ij]:
                #            if theta_ij not in alpha_hat_class_b:
                #                alpha_hat_class_b.append(theta_ij)
                #        else:
                #            pass

                self.print_debug("A:", alpha_hat_class_a, "B:", alpha_hat_class_b)

                alpha_i_hat = sum(C_dict[ij] for ij in alpha_hat_class_a) + \
                              sum(f_theta_i_star - (f_dict[ij] - C_dict[ij]) for ij in alpha_hat_class_b)

                alpha_i_new = 0

                # alpha_i is the max of the two
                alpha_i = max(alpha_i_hat, alpha_i_new)

            # if not EOPA and not TPDS:
            # RTA-CFP
            # calculate the response time based on alpha_i and beta_i
            #    Ri = Li + beta_i + math.ceil(1.0 / m * (Wi - Li - alpha_i - beta_i))
            # this improved is to bound Ri to be better than classic bound
            # Ri = Li + math.ceil(1.0 / m * (Wi - Li - max(alpha_i - (m-1) * beta_i, 0)))
            # else:
            # RTA-CFP + EOPA
            if beta_i == 0:
                Ri = Li
            else:
                I_e_lambda_ve = []
                I_e_lambda_ve_candidates = []
                I_e_lambda_ve_candidates_I = []

                len_I_lambda_ve = {}

                # I_lambda_ve calculation
                for v_k in lamda_ve:
                    I_v_k = I_dict[v_k]

                    for v_j in I_v_k:
                        if f_dict[v_j] > f_theta_i_star:
                            if Prio[v_j] > Prio[v_k]:
                                # E_k > E_i, add with confidence
                                I_e_lambda_ve.append(v_j)

                                if f_dict[v_j] - C_dict[v_j] >= f_theta_i_star:
                                    I_lambda_ve_j = C_dict[v_j]
                                else:
                                    I_lambda_ve_j = f_dict[v_j] - f_theta_i_star

                                len_I_lambda_ve[v_j] = I_lambda_ve_j
                            else:
                                # E_k < E_i, put into a list and later will only get longest m - 1
                                I_e_lambda_ve_candidates.append(v_j)

                                # get I_lambda_ve_j
                                if f_dict[v_j] - C_dict[v_j] >= f_theta_i_star:
                                    I_lambda_ve_j = C_dict[v_j]
                                else:
                                    I_lambda_ve_j = f_dict[v_j] - f_theta_i_star

                                I_e_lambda_ve_candidates_I.append(I_lambda_ve_j)

                                len_I_lambda_ve[v_j] = I_lambda_ve_j

                # sort nodes by I (if it exists), and append (m-1) longest to int_ij_EO
                if I_e_lambda_ve_candidates:
                    indices, _ = zip(
                        *sorted(enumerate(I_e_lambda_ve_candidates_I), key=itemgetter(1), reverse=True))
                    I_e_lambda_ve_candidates_sorted = []

                    for idx_ in range(len(I_e_lambda_ve_candidates_I)):
                        I_e_lambda_ve_candidates_sorted.append(I_e_lambda_ve_candidates[indices[idx_]])

                    # adding (m) lower EO nodes
                    for xxx in range(1, m + 1):
                        if len(I_e_lambda_ve_candidates) >= xxx:
                            I_e_lambda_ve.append(I_e_lambda_ve_candidates_sorted[xxx - 1])

                # test parallelism of I_lambda_ve
                if self.test_parallelism(G_dict, I_e_lambda_ve, m):
                    I_term_for_EO = 0
                else:
                    # calculate I_ve
                    I_ve = 0

                    for v_j in I_e_lambda_ve:
                        len_I_lambda_ve_j = len_I_lambda_ve[v_j]
                        I_ve = I_ve + len_I_lambda_ve_j

                    I_term_for_EO = math.ceil(1.0 / m * I_ve)

                    # check: I_ve should <= (Wi - Li - alpha_i - beta_i)
                    # print_debug("(DEBUG) Wi - Li - alpha_i - beta_i: {}".format( Wi - Li - alpha_i - beta_i ))
                    # print_debug("(DEBUG) I_ve: {}".format( I_ve ))

                Ri = Li + beta_i + I_term_for_EO

            self.print_debug("R_i:", Ri)

            R_i_minus_one = R_i_minus_one + Ri
            self.print_debug("R_sum: ", R_i_minus_one)

            alpha_arr.append(alpha_i)
            beta_arr.append(beta_i)

        R = R_i_minus_one

        # bound R to the classic bound
        # if not EOPA and not TPDS:
        #     R_classic = rta_np_classic(task_idx, m)
        #     R = min(R, R_classic)

        return R, alpha_arr, beta_arr

    def EO_Compute_Length(self, G, C):
        lf = {}
        lb = {}
        l = {}

        # topological ordering
        # (skipped as this is guaranteed by the generator)
        G_new = copy.deepcopy(G)

        # [debug]
        # print(G_new)

        source_node_id = 1
        sink_node_id = 99
        G_new[source_node_id] = []
        G_new[sink_node_id] = []
        C[source_node_id] = 1
        C[sink_node_id] = 1

        # print(C)

        # find node who has no parents
        child_nodes = []
        for key, value in copy.deepcopy(G_new).items():
            for i_value in value:
                child_nodes.append(i_value)

        for key, value in copy.deepcopy(G_new).items():
            if key not in child_nodes:
                if key is not source_node_id and key is not sink_node_id:
                    G_new[source_node_id].append(key)

        # find nodes who has no child
        for key, value in G_new.items():
            if not value:
                if key is not sink_node_id:
                    G_new[key].append(sink_node_id)
        # print(G_new)
        # [debug]

        theta_i = G_new.keys()

        for theta_ij in theta_i:
            # calculate the length
            C_i = C[theta_ij]

            # forward searching in G_new
            # [debug]
            c_array = [0] * sink_node_id
            for key in sorted(C):
                c_array[key - 1] = C[key]
            # print(c_array)

            lf_i, _ = self.graph.find_longest_path_dfs(G_new, min(theta_i), theta_ij, c_array)

            # backward searching in G_new
            # [debug]
            c_array = [0] * sink_node_id
            for key in sorted(C):
                c_array[key - 1] = C[key]
            # print(c_array)

            lb_i, _ = self.graph.find_longest_path_dfs(G_new, theta_ij, max(theta_i), c_array)

            # calculate l
            l_i = lf_i + lb_i - C_i

            # assign to length
            l[theta_ij] = l_i
            lf[theta_ij] = lf_i
            lb[theta_ij] = lb_i

        return l, lf, lb


    def EO_iter(self, G_dict, C_dict, providers, consumers, Prio):
        #global e = EOPA_Algorithm.e
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        nodes = []
        for theta_star_i in providers:
            for theta_star_ij in theta_star_i:
                nodes.append(theta_star_ij)

        for theta_i in consumers:
            for theta_ij in theta_i:
                nodes.append(theta_ij)

        for iii in nodes:
            if Prio[iii] != -1:
                pass
                # raise Exception("Some prioirities are already assigned!")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        for theta_star_i in providers:
            for i in theta_star_i:
                Prio[i] = EOPA_Algorithm.e

        EOPA_Algorithm.e = EOPA_Algorithm.e - 1

        for i, theta_star_i in enumerate(providers):
            theta_i = consumers[i]
            while theta_i:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                for iii in theta_i:
                    if Prio[iii] != -1:
                        pass
                        # raise Exception("Some prioirities are already assigned!")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # --------------------------------------------------------------------------
                # build up a new (temporal) DAG with only the consumers
                G_new = copy.deepcopy(G_dict)
                for key, value in copy.deepcopy(G_new).items():
                    if key not in theta_i:
                        del G_new[key]
                    else:
                        value_new = value.copy()
                        for j in value:
                            if j not in theta_i:
                                value_new.remove(j)
                        G_new[key] = value_new

                # --------------------------------------------------------------------------
                # find the longest local path in theta_i
                # find l(v_i)
                l, lf, lb = self.EO_Compute_Length(G_new, C_dict)

                lamda_ve = []

                # find ve
                ve = -1
                l_max = -1
                for theta_ij in theta_i:
                    if not self.graph.find_successor(G_new, theta_ij):
                        if l[theta_ij] > l_max:
                            l_max = l[theta_ij]
                            ve = theta_ij

                # found ve, then found lamda_ve
                if ve != -1:
                    lamda_ve.append(ve)
                    pre_ve = self.graph.find_predecesor(G_new, ve)
                    while pre_ve:
                        ve = -1
                        l_max = -1
                        for theta_ij in pre_ve:
                            # if not find_predecesor(G_new, theta_ij):
                            if l[theta_ij] > l_max:
                                l_max = l[theta_ij]
                                ve = theta_ij
                        if ve != -1:
                            lamda_ve.append(ve)
                        pre_ve = self.graph.find_predecesor(G_new, ve)

                # find an existed vj
                found_vj = False
                lamda_ve.sort()
                for vj in lamda_ve:
                    pre_vj = self.graph.find_predecesor(G_new, vj)
                    if len(pre_vj) > 1:
                        found_vj = True

                if found_vj:
                    # update VN_array
                    V_array = list(copy.deepcopy(G_new).keys())
                    V_array.sort()
                    VN_array = V_array.copy()
                    for lamda_ve_i in lamda_ve:
                        VN_array.remove(lamda_ve_i)

                    # find new providers and consumers
                    providers_new, consumers_new = self.find_providers_consumers(G_new, lamda_ve, VN_array)
                    self.EO_iter(G_new, C_dict, providers_new, consumers_new, Prio)
                    break
                else:
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    for vjjj in lamda_ve:
                        if Prio[vjjj] != -1:
                            pass
                            # raise Exception("Priority abnormal!")
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    for vj in lamda_ve:
                        Prio[vj] = EOPA_Algorithm.e
                        EOPA_Algorithm.e = EOPA_Algorithm.e - 1

                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    for vjjj in lamda_ve:
                        if Prio[vjjj] <= 0:
                            pass
                            # raise Exception("Priority abnormal!")
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    self.remove_nodes_in_list(theta_i, lamda_ve)

    def Eligiblity_Ordering_PA(self, G_dict, C_dict):

        Prio = {}

        # --------------------------------------------------------------------------
        # I. load task parameters
        C_exp = []
        for key in sorted(C_dict):
            C_exp.append(C_dict[key])

        V_array = list(copy.deepcopy(G_dict).keys())
        V_array.sort()
        _, lamda = self.graph.find_longest_path_dfs(G_dict, V_array[0], V_array[-1], C_exp)

        VN_array = V_array.copy()

        for i in lamda:
            if i in VN_array:
                VN_array.remove(i)

        # --------------------------------------------------------------------------
        # II. initialize eligbilities to -1
        for i in G_dict:
            Prio[i] = -1

        # --------------------------------------------------------------------------
        # III. providers and consumers
        # iterative all critical nodes
        # after this, all provides and consumers will be collected

        # >> for time measurement
        global time_EO_CPC
        begin_time = time.time()
        # << for time measurement

        providers, consumers = self.find_providers_consumers(G_dict, lamda, VN_array)

        # >> for time measurement
        time_EO_CPC = time.time() - begin_time
        # << for time measurement

        # --------------------------------------------------------------------------
        # IV. Start iteration
        # >> for time measurement
        global time_EO
        begin_time = time.time()
        # << for time measurement

        self.EO_iter(G_dict, C_dict, providers, consumers, Prio)

        # >> for time measurement
        time_EO = time.time() - begin_time
        # << for time measurement

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for i in Prio:
            if Prio[i] <= 1:
                pass
                # raise Exception("Some prioirities are not assigned!")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return Prio

    time_EO = 0
    time_EO_CPC = 0


# test code:
if __name__ == "__main__":
    #G = {1: [2, 3, 4, 7], 2: [6], 3: [5], 4: [5], 7: [8], 6: [8], 5: [8], 8: []}
    #C = {1: 0, 2: 79, 3: 74, 4: 73, 5: 76, 6: 71, 7: 99, 8: 0}
    EOPA_Algorithm = EOPA_Algorithm()
    G_dict, C_dict, C_array, lamda, VN_array, L, W = EOPA_Algorithm.load_task(1)
    # R, alpha_arr, beta_arr = rta_alphabeta_new(9, 4, False, True)
    print(G_dict)
    print(lamda)
    print(L)
    print(W)

    l, lf, lb = EOPA_Algorithm.EOPA(1, 4)
    print(l)
    # draw the gml
    dag_task_file = dag_base_folder + "Tau_{:d}.gml".format(1)
    G= nx.read_gml(dag_task_file)
    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True)
    plt.show()
