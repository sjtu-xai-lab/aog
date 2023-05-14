import graphviz
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout, pydot_layout
import numpy as np
import random
import matplotlib.patches as patches
import matplotlib.colors
from .web_utils import HighlightPlugin
import mpld3, mpld3.utils, mpld3.plugins


def rgb_to_hex(rgb):
    r, g, b = rgb
    r = int(max(0, min(r * 255, 255)))
    g = int(max(0, min(g * 255, 255)))
    b = int(max(0, min(b * 255, 255)))
    return '#%02x%02x%02x' % (r, g, b)


def shrink_bbox(coords, factor=0.5):
    left, bottom, right, top = coords[0, 0], coords[0, 1], coords[1, 0], coords[1, 1]
    width = (right - left) * factor
    height = (top - bottom) * factor

    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    return np.array([(center_x - 0.5 * width, center_y - 0.5 * height),
                     (center_x + 0.5 * width, center_y + 0.5 * height)])



def rgba_to_rgb(source_rgba, bg_rgb=(1., 1., 1.)):
    r, g, b, a = source_rgba
    r_bg, g_bg, b_bg = bg_rgb
    r = r * a + r_bg * (1 - a)
    g = g * a + g_bg * (1 - a)
    b = b * a + b_bg * (1 - a)
    return r, g, b, 1.


class AOGNode(object):
    def __init__(self, type, name="anonymous", label=None, value=None, layer_id=-1, children=None):
        '''
        To create a node in an AOG, we have to specify:
          (1) The node type: is it an AND node? is it a 'plus' node (on the top of the AOG)? is it a leaf node?
          (2) The name of the node: The id that uniquely determines a node in an AOG.
              [WARNING!!!!!] All nodes need to have different names.
          (3) The label of the node: when visualizing the graph, the nodes' label. Different nodes can share the same label.
          (4) The value of the node (if any): the interaction
          (5) The layer_id of the node: This node is in the ?-th layer.
          (6) The children of this node. This do not need to be pre-defined, as its corresponding children can be set later.
        :param type: The type of the node. Supported types: AND, +, leaf
        :param name: The unique id of each node in a graph
        :param children: The children of the current node.
        '''
        # set the type of the node
        assert type in ["AND", "+", "leaf"]
        self.type = type
        # set the name
        assert name != "anonymous"
        self.name = name
        # set the label
        self.label = label if label is not None else name
        # set the value (if any)
        self.value = value
        # set the layer id
        assert isinstance(layer_id, int) and layer_id > 0
        self.layer_id = layer_id

        self.children = None
        if children is not None:
            self.extend_children(children)

    def extend_children(self, children: list):
        assert len(children) > 0
        if self.children is None:
            self.children = []
        for child in children:
            if child not in self.children: self.children.append(child)

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return f"<AOG Node> at {id(self)}: type={self.type}, name={self.name}"


class AOG(object):
    def __init__(self, root: AOGNode=None, compulsory_nodes=None):
        '''
        To construct an AOG, we only need to specify its root node.
        :param root: The root node of the AOG.
        '''
        self.root = root
        self.compulsory_nodes = compulsory_nodes

    def visualize(self, save_path, renderer="graphviz", **kwargs):
        '''
        Use this method to visualize the current AOG.
        :param save_path: The save path of the visualized graph.
        :param renderer: The renderer. Now supported: graphviz, networkx
        :param kwargs: Other parameters. TBD later.
        :return: (void)
        '''
        random.seed(0)
        save_as_html = "html" in save_path
        self._load_kwargs(**kwargs)
        if renderer == "graphviz":
            g = graphviz.Graph('AOG', format="svg", strict=True)
            g.graph_attr.update(ratio="0.3")
            if self.root is None:
                g.render(save_path)
                return
            g.node(self.root.name, label=self.root.type)
            self._add_node_to_graphviz(g, self.root)
            g.render(save_path)
            return
        elif renderer == "networkx":
            g = nx.DiGraph()
            if "figsize" in kwargs.keys():
                fig = plt.figure(figsize=kwargs["figsize"])
            else:
                if self.simplify:
                    fig = plt.figure(figsize=(7, 3.5))
                    plt.ylim(-0.1, 1.2)
                else:
                    fig = plt.figure(figsize=(14, 6))
                    plt.ylim(-0.1, 1.2)
                # plt.ylim([0, 1])
            if self.root is None:
                nx.draw(g)
                fig = plt.axis("off")
                plt.savefig(save_path, dpi=200)
                return
            # ================================================
            # start constructing the Graph object in networkx
            # ================================================
            g.add_node(self.root.name, label=self.root.label, layer_id=self.root.layer_id, value=self.root.value)
            self._add_node_to_networkx(g, self.root)
            self._add_compulsory_node_to_networkx(g)

            self._generate_aog_hierarchy_networkx(g)

            pos = self._generate_layout_networkx(g, **kwargs)
            self._generate_node_color_networkx(g, **kwargs)

            # plot nodes
            node_attr = self._generate_node_attr_networkx(g, **kwargs)
            nx.draw_networkx_nodes(g, pos, **node_attr)

            # plot labels on nodes
            label_boxes = []  # for HTML
            label_texts = []  # for HTML
            label_attr = self._generate_label_attr_networkx(g, **kwargs)
            labels = self._generate_label_networkx(g, **kwargs)
            for node_id, node_name in enumerate(g.nodes):
                label_attr_this = {k: v[node_id] if isinstance(v, list) else v for k, v in label_attr.items()}
                label_this = {node_name: labels[node_name]}

                label = nx.draw_networkx_labels(g, pos, labels=label_this, **label_attr_this)

                if save_as_html:  # for HTML
                    fig.canvas.draw()
                    patch = label[node_name].get_bbox_patch()
                    box = patch.get_extents()
                    coords = plt.gca().transData.inverted().transform(box)
                    coords = shrink_bbox(coords, factor=0.5)

                    bbox = patches.FancyBboxPatch(
                        coords[0], coords[1, 0] - coords[0, 0], coords[1, 1] - coords[0, 1],
                        boxstyle=patches.BoxStyle("round,pad=0.02"), fc=label_attr_this['bbox']['fc'],
                        ec=label_attr_this['bbox']['ec'], zorder=3
                    )
                    plt.gca().add_patch(bbox)
                    # label_boxes.append(bbox)

                    bbox_ = patches.FancyBboxPatch(
                        coords[0], coords[1, 0] - coords[0, 0], coords[1, 1] - coords[0, 1],
                        boxstyle=patches.BoxStyle("round,pad=0.02"), fc=label_attr_this['bbox']['fc'],
                        ec=label_attr_this['bbox']['ec'], zorder=10, alpha=0.0
                    )
                    plt.gca().add_patch(bbox_)
                    label_boxes.append(bbox_)

                    label_texts.append(label[node_name])

            # plot edges
            edge_attr = self._generate_edge_attr_networkx(g, **kwargs)
            if save_as_html:  # for HTML
                edge_attr["arrows"] = True
                edge_attr["width"] = [5 for _ in edge_attr["width"]]
            edges = nx.draw_networkx_edges(g, pos, **edge_attr)

            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)

            if not save_as_html:
                # self._add_annotation_networkx(plt.gca(), **kwargs)
                title = self._generate_title_networkx(g, **kwargs)
                plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'})
                plt.tight_layout()
                plt.axis("off")
                # plt.savefig(save_path, dpi=200, transparent=True)
                plt.savefig(save_path, dpi=200)
                plt.close("all")
            else:  # for HTML
                if "title" in kwargs.keys():
                    plt.title(kwargs["title"], fontdict={'fontsize': 20, 'fontweight': 'bold'})
                highlight_edges, highlight_colors, default_color = self._generate_highlight_info_html(g)

                mpld3.plugins.connect(fig, HighlightPlugin(label_boxes, label_texts, edges,
                                                           highlight_edges=highlight_edges,
                                                           highlight_colors=highlight_colors,
                                                           default_color=default_color))

                mpld3.save_html(fig, save_path)
                return fig

    def _load_kwargs(self, **kwargs):
        if "simplify" in kwargs.keys() and kwargs["simplify"]:
            self.simplify = True
        else:
            self.simplify = False

    def _add_node_to_graphviz(self, g, node):
        for child in node.children:
            if child.type == "leaf":
                g.node(child.name)
                g.edge(node.name, child.name)
            else:
                # g.node(child.name, label=child.type)
                g.node(child.name)
                g.edge(node.name, child.name)
                self._add_node_to_graphviz(g, child)

    def _add_node_to_networkx(self, g, node: AOGNode):
        for child in node.children:
            if child.type == "leaf":
                if not g.has_node(child.name):
                    g.add_node(child.name, label=child.label, layer_id=child.layer_id, value=child.value)
                if not g.has_edge(node.name, child.name):
                    g.add_edge(node.name, child.name)
            else:
                if not g.has_node(child.name):
                    g.add_node(child.name, label=child.label, layer_id=child.layer_id, value=child.value)
                if not g.has_edge(node.name, child.name):
                    g.add_edge(node.name, child.name)
                self._add_node_to_networkx(g, child)

    def _add_compulsory_node_to_networkx(self, g):
        if self.compulsory_nodes is None:
            return
        for node in self.compulsory_nodes:
            if not g.has_node(node.name):
                g.add_node(node.name, label=node.label, layer_id=node.layer_id, value=node.value)

    def _generate_aog_hierarchy_networkx(self, g):
        assert '+' in g.nodes
        layer_4 = ['+']
        layer_3 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 3]
        layer_2 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 2]
        layer_1 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 1]
        layer_1 = sorted(layer_1, key=lambda _node_name: g.nodes[_node_name]["value"])
        self.layer_4 = layer_4
        self.layer_3 = layer_3
        self.layer_2 = layer_2
        self.layer_1 = layer_1

    def _generate_layout_networkx(self, g, **kwargs):
        assert '+' in g.nodes
        if 'n_row_interaction' in kwargs.keys(): n_row_interaction = kwargs["n_row_interaction"]
        else: n_row_interaction = 2

        layer_4 = self.layer_4
        layer_3 = self.layer_3
        layer_2 = self.layer_2
        layer_1 = self.layer_1
        random.shuffle(layer_3)
        pos = {}

        # ys = [0., 2., 4., 5., 6., 7., 8.]
        # ys = np.array(ys) / max(ys)
        if n_row_interaction == 2:
            ys = np.array([0, 0.28, 0.56, 0.78, 1.0])  # ugly codes
        elif n_row_interaction == 3:
            # ys = np.array([0, 0.28, 0.55, 0.70, 0.85, 1.0])  # ugly codes
            ys = np.array([0, 0.25, 0.49, 0.66, 0.83, 1.0])  # ugly codes
        elif n_row_interaction == 4:
            # ys = np.array([0, 0.28, 0.53, 0.65, 0.77, 0.89, 1.0])  # ugly codes
            ys = np.array([0, 0.20, 0.40, 0.55, 0.70, 0.85, 1.0])  # ugly codes
        elif n_row_interaction == 6:
            ys = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # ugly codes
        else:
            ys = np.linspace(0, 1, 3+n_row_interaction)

        # set the position of layer-1 nodes
        layer_1_lengths = [len(g.nodes[node_name]["label"]) for node_name in layer_1]
        delta = 0.05
        d = (1 - delta*(len(layer_1_lengths)-1)) / (sum(layer_1_lengths) - 0.5 * (layer_1_lengths[0] + layer_1_lengths[-1]))
        layer_1_ratios = [0.] + [0.5 * (layer_1_lengths[i] + layer_1_lengths[i+1]) * d + delta for i in range(len(layer_1_lengths)-1)]
        layer_1_ratios = np.cumsum(layer_1_ratios)
        for i in range(len(layer_1)):
            pos[layer_1[i]] = np.array([layer_1_ratios[i], ys[0]])

        # set the position of layer-2 nodes
        if len(layer_2) > 0:
            # ugly codes
            if len(layer_2) == 1: x = np.array([0])
            elif len(layer_2) == 2: x = np.array([0, 0.4])
            elif len(layer_2) == 3: x = np.array([0, 0.3, 0.6])
            elif len(layer_2) == 4: x = np.array([0, 0.2, 0.4, 0.6])
            elif len(layer_2) == 5: x = np.array([0, 0.2, 0.4, 0.6, 0.8])
            else: raise NotImplementedError
            x = x + (0.5 - x.mean())
            for i in range(len(layer_2)):
                pos[layer_2[i]] = np.array([x[i], ys[1]])

        # set the position of layer-3 nodes
        n_col_interaction = int(np.ceil(len(layer_3) / n_row_interaction))
        x = np.linspace(0.02, 0.98, n_col_interaction)
        y = ys[2:2+n_row_interaction][::-1]
        for i in range(len(layer_3)):
            if i // n_col_interaction == n_row_interaction - 1 and i % n_col_interaction == 0:
                n_rest = n_col_interaction * n_row_interaction - len(layer_3)
                x += 0.5 * n_rest * (x[1] - x[0])
            pos[layer_3[i]] = np.array([x[i%n_col_interaction], y[i//n_col_interaction]])

        # set the position of layer-4 nodes
        y = ys[2+n_row_interaction]
        pos[layer_4[0]] = np.array([0.5, y])
        return pos

    def _generate_node_attr_networkx(self, g, **kwargs):
        min_alpha = 0.05
        # colors = ["#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]  # layer-4 to layer-1
        colors = ["#9e9e9e", {'pos': "#e53935", 'neg': "#1e88e5"}, "#ce93d8", "#9e9e9e"]  # layer-4 to layer-1  # positive interaction is red
        if "reverse_color" in kwargs and kwargs["reverse_color"] == True:
            colors = ["#9e9e9e", {'neg': "#e53935", 'pos': "#1e88e5"}, "#ce93d8", "#9e9e9e"]  # layer-4 to layer-1  # positive interaction is blue
        node_attr = {
            "node_shape": 's',
            "node_size": [],
            "node_color": [],
            "alpha": [],
        }
        max_value = max([abs(g.nodes[node_name]['value']) for node_name in self.layer_3])
        for node in g.nodes:
            # node_attr["node_size"].append(3000)
            node_size = (g.nodes[node]["label"].count("\n") + 1) * 1000
            node_attr["node_size"].append(node_size)
            if node in self.layer_4: node_attr["node_color"].append(colors[0])
            elif node in self.layer_3:
                if g.nodes[node]['value'] > 0: node_attr["node_color"].append(colors[1]['pos'])
                else: node_attr["node_color"].append(colors[1]['neg'])
            elif node in self.layer_2: node_attr["node_color"].append(colors[2])
            elif node in self.layer_1: node_attr["node_color"].append(colors[3])
            else: raise Exception
            if node in self.layer_3:
                alpha = abs(g.nodes[node]['value']) / max_value
                alpha = alpha * (1 - min_alpha) + min_alpha
                node_attr["alpha"].append(alpha)
            else:
                node_attr["alpha"].append(0.8)

        node_attr["alpha"] = [0 for _ in node_attr["alpha"]]

        return node_attr


    def _generate_highlight_info_html(self, g):

        gray_rgb = matplotlib.colors.to_rgba_array("gray").squeeze().tolist()[:3]
        # default_color = rgb_to_hex(rgba_to_rgb(source_rgba=(*gray_rgb, 0.3))[:3])
        default_color = "rgba(128, 128, 128, 0.3)"

        edge_lookup_table = {edge: i for i, edge in enumerate(g.edges)}

        highlight_edges = []
        highlight_colors = []

        for node in g.nodes:
            if g.nodes[node]["layer_id"] != 3:
                highlight_edges.append([])
                highlight_colors.append(default_color)
            else:
                # run bfs and take the related edge ids
                h_edge = []
                h_edge.append(edge_lookup_table[(self.root.name, node)])
                for (u, v) in nx.bfs_edges(g, node):
                    h_edge.append(edge_lookup_table[(u, v)])
                highlight_edges.append(h_edge)
                # the color of highlighting
                color_rgba = self.node_colors_rgba[node]
                h_color = rgba_to_rgb([*color_rgba[:3], max(0.8, color_rgba[3])])
                h_color = rgb_to_hex(h_color[:3])
                highlight_colors.append(h_color)

        return highlight_edges, highlight_colors, default_color

    def _generate_edge_attr_networkx(self, g, **kwargs):
        highlight_path = None
        if "highlight_path" in kwargs.keys(): highlight_path = kwargs["highlight_path"]
        highlight_edges = {}
        if highlight_path == "max":
            target_node = sorted(self.layer_3, key=lambda x: abs(float(g.nodes[x]["value"])), reverse=True)[0]
            color = self.node_colors[target_node]
            highlight_edges[(self.root.name, target_node)] = color
            for (u, v) in nx.bfs_edges(g, target_node):
                highlight_edges[(u, v)] = color
        elif highlight_path == "max-20%":
            node_num = int(np.floor(0.2 * len(self.layer_3)))
            target_nodes = sorted(self.layer_3, key=lambda x: abs(float(g.nodes[x]["value"])), reverse=True)[:node_num]
            for target_node in target_nodes:
                color = self.node_colors[target_node]
                highlight_edges[(self.root.name, target_node)] = color
                for (u, v) in nx.bfs_edges(g, target_node):
                    highlight_edges[(u, v)] = color
        elif isinstance(highlight_path, str) and highlight_path.startswith("rank-"):
            rank = min(len(self.layer_3), int(highlight_path.split("-")[1])) - 1
            target_node = sorted(self.layer_3, key=lambda x: abs(float(g.nodes[x]["value"])), reverse=True)[rank]
            # color = self.node_colors[target_node]

            color_rgba = self.node_colors_rgba[target_node]
            color = rgba_to_rgb([*color_rgba[:3], max(0.8, color_rgba[3])])

            highlight_edges[(self.root.name, target_node)] = color
            for (u, v) in nx.bfs_edges(g, target_node):
                highlight_edges[(u, v)] = color
        edge_attr = {
            "width": [],
            "edge_color": [],
            "arrows": False
        }

        for u, v in g.edges:
            if (u, v) in highlight_edges.keys() or (v, u) in highlight_edges.keys():
                color = highlight_edges[(u, v)] if (u, v) in highlight_edges.keys() \
                                                else highlight_edges[(v, u)]
                edge_attr["width"].append(5)
                edge_attr["edge_color"].append(color)
            else:
                edge_attr["width"].append(3)
                gray_rgb = matplotlib.colors.to_rgba_array("gray").squeeze().tolist()[:3]
                edge_attr["edge_color"].append((*gray_rgb, 0.3))

        return edge_attr

    def _generate_node_color_networkx(self, g, **kwargs):
        self.node_colors = {}
        self.node_colors_rgba = {}

        min_alpha = 0.05
        # colors = ["#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]  # layer-4 to layer-1
        colors = ["#F2F2F2", {'pos': "#e53935", 'neg': "#1e88e5"}, "#EBDCEC", "#F2F2F2"]  # layer-4 to layer-1  # positive interaction is red
        if "reverse_color" in kwargs and kwargs["reverse_color"] == True:
            colors = ["#F2F2F2", {'neg': "#e53935", 'pos': "#1e88e5"}, "#EBDCEC", "#F2F2F2"]  # layer-4 to layer-1  # positive interaction is blue

        max_value = max([abs(g.nodes[node_name]['value']) for node_name in self.layer_3])
        for node in g.nodes:
            rgba: list = list()
            if node in self.layer_4:
                rgba = matplotlib.colors.to_rgba_array(colors[0]).squeeze().tolist()[:3]
            elif node in self.layer_3:
                if g.nodes[node]['value'] > 0: color = colors[1]['pos']
                else: color = colors[1]['neg']
                rgba = matplotlib.colors.to_rgba_array(color).squeeze().tolist()[:3]
            elif node in self.layer_2:
                rgba = matplotlib.colors.to_rgba_array(colors[2]).squeeze().tolist()[:3]
            elif node in self.layer_1:
                rgba = matplotlib.colors.to_rgba_array(colors[3]).squeeze().tolist()[:3]
            else:
                raise Exception
            if node in self.layer_3:
                alpha = abs(g.nodes[node]['value']) / max_value
                alpha = alpha * (1 - min_alpha) + min_alpha
                rgba.append(alpha)
            else:
                rgba.append(0.95)
            assert len(rgba) == 4
            rgb = rgba_to_rgb(rgba)
            self.node_colors[node] = tuple(rgb)
            self.node_colors_rgba[node] = tuple(rgba)


    def _generate_label_networkx(self, g, **kwargs):
        if self.simplify:
            return self._generate_label_networkx_simplify(g, **kwargs)
        else:
            return self._generate_label_networkx_default(g, **kwargs)

    def _generate_label_networkx_default(self, g, **kwargs):
        return dict(g.nodes.data("label"))

    def _generate_label_networkx_simplify(self, g, **kwargs):
        labels = {}
        for node_name in g.nodes:
            if node_name in self.layer_1: labels[node_name] = g.nodes[node_name]["label"]
            elif node_name in self.layer_2 or node_name in self.layer_3: labels[node_name] = " "
            elif node_name in self.layer_4: labels[node_name] = "+"
            else: raise Exception
        return labels


    def _generate_label_attr_networkx(self, g, **kwargs):
        if self.simplify:
            return self._generate_label_attr_networkx_simplify(g, **kwargs)
        else:
            return self._generate_label_attr_networkx_default(g, **kwargs)



    def _generate_label_attr_networkx_default(self, g, **kwargs):
        label_attr = {
            "font_size": [],
            "bbox": []
        }

        for node in g.nodes:
            label_attr["font_size"].append(20)

            bbox_attr = {
                "boxstyle": 'round,pad=0.5',
                "ec": "none",
                "fc": self.node_colors[node],
            }

            label_attr["bbox"].append(bbox_attr)

        return label_attr

    def _generate_label_attr_networkx_simplify(self, g, **kwargs):
        label_attr = {
            "font_size": [],
            "bbox": []
        }

        for node in g.nodes:
            label_attr["font_size"].append(20)

            if node in self.layer_2 or node in self.layer_3: boxstyle = "circle"
            else: boxstyle = 'round,pad=0.5'

            bbox_attr = {
                "boxstyle": boxstyle,
                "ec": "none",
                "fc": self.node_colors[node],
            }

            label_attr["bbox"].append(bbox_attr)

        return label_attr

    def _add_annotation_networkx(self, ax, **kwargs):
        if 'n_row_interaction' in kwargs.keys(): n_row_interaction = kwargs["n_row_interaction"]
        else: n_row_interaction = 2
        y = np.linspace(0, 1, n_row_interaction + 3)
        interval = y[1] - y[0]
        rect = patches.Rectangle(
            (-0.01, y[2] - 0.3 * interval),
            1.02, y[-2] - y[2] + 0.6 * interval,
            linewidth=5, linestyle="dashed",
            edgecolor='lightsteelblue', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(-0.03, (y[-2] + y[2]) / 2, "The Most Salient Patterns",
                rotation=90, va="center", fontsize=15, color="lightsteelblue", weight="bold")

    def _generate_title_networkx(self, g, **kwargs):
        title = ""
        if "title" in kwargs.keys():
            title = kwargs["title"]
        else:
            title += "AOG"
            title += f" | # edges: {g.number_of_edges()}"
            title += f" | # nodes: {g.number_of_nodes()}"
        return title




if __name__ == '__main__':

    A = AOGNode(type="leaf", name="A", children=None)
    B = AOGNode(type="leaf", name="B", children=None)
    C = AOGNode(type="leaf", name="C", children=None)
    D = AOGNode(type="leaf", name="D", children=None)
    E = AOGNode(type="leaf", name="E", children=None)

    aog = AOG(root=AOGNode(
        type="OR", name="AB(C+D)+DE",
        children=[
            AOGNode(type="AND", name="AB(C+D)", children=[
                AOGNode(type="AND", name="AB", children=[A, B]),
                AOGNode(type="+", name="(C+D)", children=[C, D])
            ]),
            AOGNode(type="AND", name="DE", children=[D, E])
        ]
    ))

    aog.visualize("test")