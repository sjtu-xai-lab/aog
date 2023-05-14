import os
import os.path as osp
import networkx as nx
import mpld3, mpld3.utils, mpld3.plugins

import matplotlib, matplotlib.lines
import matplotlib.pyplot as plt
FILE_FOLDER = osp.dirname(__file__)


class HighlightPlugin(mpld3.plugins.PluginBase):
    """A simple plugin showing how multiple axes can be linked"""

    with open(osp.join(FILE_FOLDER, "static/javascript/highlight_plugin.js"), "r") as f:
        JAVASCRIPT = f.read()

    with open(osp.join(FILE_FOLDER, "static/css/remove_axis.css"), "r") as f:
        CSS = f.read()

    def __init__(self, node_bboxes, node_texts, edges,
                 highlight_edges, highlight_colors, default_color="black"):
        self.css_ = self.CSS
        self.dict_ = {
            "type": "highlight",
            "id_node_bboxes": [mpld3.utils.get_id(bbox) for bbox in node_bboxes],
            "id_node_texts": [mpld3.utils.get_id(text) for text in node_texts],
            "id_edges": [mpld3.utils.get_id(edge) for edge in edges],
            "highlight_edges": highlight_edges,
            "highlight_colors": highlight_colors,
            "default_color": default_color,
        }