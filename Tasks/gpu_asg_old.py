import io
import os
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import Dataset
from torch.autograd import Variable
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib
import networkx as nx
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GpuAssignmentDataset(Dataset):
    def __init__(self, num_samples, max_load=5, max_demand=1,
                 racks_per_cluster=2,
                 machines_per_rack=2,
                 gpus_per_machine=4,
                 max_gpu_request=6,
                 seed=None):
        super(GpuAssignmentDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        input_size = gpus_per_machine * machines_per_rack * racks_per_cluster
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        G = nx.Graph()
        cur = 0
        G.add_node('center')
        demand_mask = [True]
        for i in range(racks_per_cluster):
            node_r = f'r{i}'
            G.add_node(node_r)
            G.add_edge(node_r, 'center', weight=0.5)
            demand_mask.append(True)
            for j in range(machines_per_rack):
                node_m = f'r{i}m{j}'
                G.add_node(node_m)
                G.add_edge(node_m, node_r, weight=0.1)
                demand_mask.append(True)
                for k in range(gpus_per_machine):
                    node_g = f'g{cur}'
                    G.add_node(node_g)
                    G.add_edge(node_g, node_m, weight=0.025)
                    demand_mask.append(False)
                    for l in range(k):
                        node_neighbor = f'g{cur - l - 1}'
                        G.add_edge(node_g, node_neighbor, weight=0.025)
                    cur += 1
        resources = nx.adjacency_matrix(G).todense()
        self.static = np.zeros((num_samples, resources.shape[0], resources.shape[1]), dtype=np.float32)
        self.static[0:] = resources
        self.root = torch.from_numpy(self.static[0][0].reshape(-1,1))

        # self.static = np.expand_dims(self.static, axis=(0,1))


        dynamic_shape = (num_samples, 1, input_size + 1 + racks_per_cluster + machines_per_rack * racks_per_cluster)
        self.gpu_request_seq = torch.randint(1,max_gpu_request, (num_samples, ))
        loads = torch.full(dynamic_shape, 1.)

        demands_gpu = torch.randint(0, max_demand + 1, (num_samples, 1, input_size))
        demands_gpu= demands_gpu/ float(max_load)

        demands = torch.zeros(dynamic_shape)
        # print(demand_mask)
        # print('gpus')
        # print(demands_gpu)
        j = 0
        for i in range(len(demand_mask)):
            if demand_mask[i] == False:
                demands[:,:,i] = demands_gpu[:,:,j]
                j += 1
        # print('after:')
        # print(demands)


        demand_mask = torch.tensor(demand_mask)
        demands[:, 0, 0] = 0 # depot starts with a demand of 0
        for d1 in demands:
            for d2 in d1:
                # d2[demand_mask] = 100
                d2[demand_mask == True] = 0

        self.demand_mask = demand_mask
        self.G = G

        self.nodes_lst = np.array(list(self.G))
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        x = torch.from_numpy(self.static[idx])
        z = self.root
        return (x, self.dynamic[idx], z)

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)
        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.

        if ~(repeat_home).any():
            new_mask[~(repeat_home).nonzero(), 0] = 0

        # for i in range(len(new_mask[0])):
        #     if new_mask[0][i] == self.demand_mask[i]:
        #         print(f'{i} {self.nodes_lst[i]}')
        # print('-'*50)
        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))
        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        # return torch.tensor(tensor.data, device=dynamic.device)
        return tensor.data.clone().detach()

    def reward(self, static, tour_indices):
        """
        tour distance of selected GPUs
        """
        # if len(tour_indices.cpu()[0]) == 1:
        #     nodes = np.array([self.nodes_lst[tour_indices.cpu()]])
        # else:
        #     nodes = self.nodes_lst[tour_indices.cpu()][0]
        nodes = np.array([self.nodes_lst[tour_indices.cpu()]])
        nodes = np.append(nodes, 'center')
        routes, path = self.find_routes(nodes, self.G)
        length = 0
        for pair in path:
            length += self.G[pair[0]][pair[1]]["weight"]
        return torch.tensor(-length, dtype=torch.float32)


    def render(self, logger, dynamic, name, epoch, tour_indices):
        """Plots the found solution."""
        nodes = self.nodes_lst[tour_indices.cpu()][0]
        nodes = np.insert(nodes, 0, 'center')

        routes, path = self.find_routes(nodes, self.G)
        plt.close('all')

        pos = graphviz_layout(self.G, prog="twopi")

        # nx.draw_networkx_nodes(self.G, pos, nodelist=[], node_color="g")

        plt.rcParams["figure.figsize"] = [12, 10]
        plt.rcParams["figure.dpi"] = 60
        plt.rcParams["figure.autolayout"] = False
        nx.draw_networkx(self.G, pos, with_labels=True)
        # for ctr, edgelist in enumerate(path):
        #     nx.draw_networkx_edges(self.G, pos=pos, edgelist=edgelist, edge_color='r', width=5)
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=path, edge_color='r', width=5)
        demands = dynamic.data[:, 1]
        node_lst = demands.ne(0)[0].cpu()
        # print(demands)
        # print(self.nodes_lst)
        # print(self.nodes_lst[~node_lst])
        # print(self.nodes_lst[tour_indices][0])
        # print('-'*50)
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.nodes_lst[~node_lst], node_color="r")
        plt.tight_layout()
        plt.title(name)
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', dpi=200, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        logger.image_summary(epoch, name, image)

    def construct_edges(self, nodes):
        lst = []
        prev = nodes[0]
        for node in nodes:
            if prev != node:
                lst.append((prev, node))
            prev = node
        return lst

    def find_routes(self,nodes, G):
        # sp = dict(nx.all_pairs_shortest_path(G))
        # path = []
        # routes = [(a, b) for idx, a in enumerate(nodes) for b in nodes[idx + 1:]]
        # for pair in routes:
        #     path.append(self.construct_edges(sp[pair[0]][pair[1]]))
        # return routes, path
        sp = dict(nx.all_pairs_shortest_path(G))
        path = []
        routes = [[a, b] for idx, a in enumerate(nodes) for b in nodes[idx + 1:]]
        for pair in routes:
            path.extend(self.construct_edges(sp[pair[0]][pair[1]]))

        path = self.removeDuplicates(path)
        return routes, path

    def removeDuplicates(self, lst):

        return [t for t in (set(tuple(i) for i in lst))]

if __name__ == '__main__':
    dataset = GpuAssignmentDataset(2, 16)

