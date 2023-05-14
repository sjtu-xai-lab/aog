mpld3.register_plugin("highlight", HighlightPlugin);
HighlightPlugin.prototype = Object.create(mpld3.Plugin.prototype);
HighlightPlugin.prototype.constructor = HighlightPlugin;
HighlightPlugin.prototype.requiredProps = ["id_node_bboxes", "id_node_texts", "id_edges",
    "highlight_edges", "highlight_colors", "default_color"];
HighlightPlugin.prototype.defaultProps = {}
function HighlightPlugin(fig, props){
    mpld3.Plugin.call(this, fig, props);
};


function update_linebreak(){
    var text_collection = document.getElementsByClassName("mpld3-text");
    for (i = 0; i < text_collection.length; i++){
        if (text_collection[i].innerHTML.includes("\n")){
            lines = text_collection[i].innerHTML.split("\n");
            x_pos = text_collection[i].getAttribute("x");
            new_html = "";
            new_html += "<tspan x='" + x_pos + "' dy='-" + ((lines.length - 1) * 0.5) + "em'>" + lines[0] + "</tspan>"
            for (j = 1; j < lines.length; j++){
                line = lines[j];
                new_html += "<tspan x='" + x_pos + "' dy='1em'>" + line + "</tspan>"
            }
            text_collection[i].innerHTML = new_html;
        }
    }
}


HighlightPlugin.prototype.draw = function(){
  update_linebreak();
  var node_boxes = [];
  for (i = 0; i < this.props.id_node_bboxes.length; i++){
      var node_box = mpld3.get_element(this.props.id_node_bboxes[i]);
      // console.log(this.props.id_node_bboxes[i]);
      node_boxes.push(node_box);
  }
  var node_texts = [];
  for (i = 0; i < this.props.id_node_texts.length; i++){
      var node_text = mpld3.get_element(this.props.id_node_texts[i]);
      // console.log(this.props.id_node_texts[i]);
      node_texts.push(node_text);
  }
  var edges = [];
  for (i = 0; i < this.props.id_edges.length; i++){
      var edge = mpld3.get_element(this.props.id_edges[i]);
      edges.push(edge);
  }
  var highlight_edges = this.props.highlight_edges;
  var highlight_colors = this.props.highlight_colors;
  var default_color = this.props.default_color;

  function mouseover(node_id){
      return () => {
          h_edges = highlight_edges[node_id];
          h_color = highlight_colors[node_id];
          if (h_edges.length > 0) {
              for (var j = 0; j < h_edges.length; j++){
                  edge_id = h_edges[j];
                  edges[edge_id].elements()
                      .transition()
                      .style("stroke", h_color);
              }
          }
      }

  }

  function mouseout(node_id){
      return () => {
        h_edges = highlight_edges[node_id];
        if (h_edges.length > 0) {
            for (var j = 0; j < h_edges.length; j++){
                edge_id = h_edges[j];
                edges[edge_id].elements()
                    .transition()
                    .style("stroke", default_color);
            }
        }
      }

  }

  for (i = 0; i < node_boxes.length; i++){
      node_boxes[i].elements()
          .on("mouseover", mouseover(i))
          .on("mouseout", mouseout(i));
  }

  // for (i = 0; i < node_texts.length; i++){
  //     node_texts[i].elements()
  //         .on("mouseover", mouseover(i))
  //         .on("mouseout", mouseout(i));
  // }
};

