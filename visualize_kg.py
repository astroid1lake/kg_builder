import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_kg(json_path, output_path=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    kg = data.get("knowledge_graph", {})
    entities = {e["id"]: e for e in kg.get("entities", [])}
    relationships = kg.get("relationships", [])
    
    G = nx.DiGraph()
    
    for e in entities.values():
        G.add_node(e["name"], type=e["type"], desc=e["description"])
    
    for r in relationships:
        G.add_edge(r["source"], r["target"], relation=r["relation"])
    
    plt.figure(figsize=(16, 12))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    node_types = {}
    for n in G.nodes():
        ntype = G.nodes[n].get("type", "未知")
        node_types[n] = node_types.get(ntype, len(node_types))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    node_colors = [colors[node_types.get(n, 0) % len(colors)] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, edge_color='#666666', arrows=True, 
                           arrowsize=20, alpha=0.7, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, 
                                  font_color='#333333', bbox=dict(boxstyle='round', 
                                  facecolor='white', alpha=0.8))
    
    plt.title(f"知识图谱: {data.get('source_file', '')}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图片已保存至: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="知识图谱可视化")
    parser.add_argument("input", help="输入JSON文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径")
    args = parser.parse_args()
    
    visualize_kg(args.input, args.output)
