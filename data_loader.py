import numpy as np
import random
import os
import pandas as pd


def load_data(args):
    n_nodea, n_nodeb,sl2id_np = load_sl2id(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg2id(args)

    print('data loaded.')
    print('n_nodea:',n_nodea)
    print('n_nodeb:', n_nodeb)
    print('n_entity:', n_entity)

    return n_nodea, n_nodeb, n_entity, n_relation, adj_entity, adj_relation,sl2id_np


def load_sl2id(args):
    print('reading sl2id file ...')

    # reading sl2id file
    sl_p = np.loadtxt('./data/SL.txt', dtype=np.int64, delimiter=',')
    sl_p = np.insert(sl_p, 2, np.ones(len(sl_p)), axis=1)
    sl_n = np.loadtxt('./data/yard.txt', delimiter='\t', dtype=np.int64)
    sl_n = np.insert(sl_n, 2, np.zeros(len(sl_n)), axis=1)
    sl2id_np = np.concatenate((sl_p, sl_n), axis=0)


    n_nodea = len(set(sl2id_np[:, 0]))
    n_nodeb = len(set(sl2id_np[:, 1]))

    return n_nodea, n_nodeb,sl2id_np


def load_kg2id(args):
    print('reading kg2id file ...')

    # reading kg2id file
    kg = np.loadtxt('./data/kg.txt', dtype=np.int64, delimiter=',')
    # kg = np.load('./data/kg.npy')
    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    kg2dict = construct_kg2dict(kg)
    adj_entity, adj_relation = construct_adj(args, kg2dict, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg2dict(kg):
    print('constructing knowledge graph dict ...')
    kg2dict = dict()
    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if head not in kg2dict:
            kg2dict[head] = []
        kg2dict[head].append((tail, relation))

        # treat the KG as an undirected graph
        if tail not in kg2dict:
            kg2dict[tail] = []
        kg2dict[tail].append((head, relation))

    return kg2dict


def construct_adj(args, kg2dict, n_entity):
    print('constructing adjacency matrix including entity and relation ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations

    isolated_point = []
    adj_entity = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)
    for entity,entity_name in enumerate(kg2dict.keys()):
        if entity in kg2dict.keys():
            neighbors = kg2dict[entity]
        else:
            neighbors = [(entity,24)]
            isolated_point.append(entity)

        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    # (pd.DataFrame(isolated_point)).to_csv('../results/isolated_point.csv')

    return adj_entity, adj_relation
