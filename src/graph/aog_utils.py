import numpy as np
import torch

from .graph import AOGNode, AOG
from pprint import pprint


def get_concept_description(attributes, single_feature):
    '''
    Generate a description of attributes
    :param attributes: basic attributes (in the original dataset) (n_dim,) list
    :param single_feature: Bool mask of a 'single features' (n_dim,)
    :return: a description string
    '''
    description = []
    for i in range(single_feature.shape[0]):
        if single_feature[i]: description.append(attributes[i])
    description = "\n".join(description)
    return description


def construct_AOG(attributes, single_features, concepts, interactions, attribute_suffixs=None):
    '''
    Construct a simple And-Or Graph
    :param attributes: basic attributes (in the original dataset) (n_dim,) list
    :param single_features: Bool masks of 'single features', including the aggregated ones (n_dim', n_dim)
    :param concepts: Bool masks representing extracted concepts (n_concept, n_dim')
    :param interactions: The multi-variate interaction of these concepts (n_concept,)
    :param attribute_suffixs: basic attributes (arrow compared with their baselines) (n_dim,) list
    :return: an AOG object
    '''
    if isinstance(interactions, torch.Tensor):
        interactions = interactions.cpu().numpy()
    if isinstance(single_features, torch.Tensor):
        single_features = single_features.cpu().numpy()
    if isinstance(concepts, torch.Tensor):
        concepts = concepts.cpu().numpy()
    if attribute_suffixs is None: attribute_suffixs = ["" for _ in attributes]
    single_feature_nodes = []
    for i in range(single_features.shape[0]):
        single_feature = single_features[i]
        if i < len(attributes): assert single_feature.sum() == 1
        description = get_concept_description(attributes, single_feature)
        if single_feature.sum() == 1:
            label = attributes[attributes.index(description)] + attribute_suffixs[attributes.index(description)]
            single_feature_node = AOGNode(type="leaf", name=str(single_feature) + "(word)",
                                          label=label, layer_id=1, children=None, value=i)
            single_feature_nodes.append(single_feature_node)
        else:
            single_feature_node = AOGNode(type="AND", name=str(single_feature), label=description, layer_id=2)
            single_feature_node.extend_children([single_feature_nodes[j]
                                                 for j in range(len(attributes)) if single_feature[j]])
            single_feature_nodes.append(single_feature_node)
    concept_nodes = []
    for i in range(concepts.shape[0]):
        concept = concepts[i]
        if not np.any(concept): continue
        concept_node = AOGNode(type="AND", name=str(concept),
                               label="{:+.2f}".format(interactions[i]), layer_id=3, value=interactions[i])
        concept_node.extend_children([single_feature_nodes[j] for j in range(len(single_feature_nodes)) if concept[j]])
        concept_nodes.append(concept_node)
    root = AOGNode(type="+", name="+", label="+", layer_id=4, children=concept_nodes)
    aog = AOG(root=root, compulsory_nodes=single_feature_nodes[:len(attributes)])
    return aog