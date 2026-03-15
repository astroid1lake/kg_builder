import os
import yaml
import json
import time
import re
import argparse
from pathlib import Path
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_config(config_path="config.yaml"):
    project_root = Path(__file__).parent
    full_path = project_root / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class KGraphBuilder:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.client = OpenAI(
            api_key=self.config["api"]["key"],
            base_url=self.config["api"]["base_url"]
        )
    
    def call_llm(self, prompt, max_tokens=2000):
        max_retries = self.config["api"].get("max_retries", 3)
        timeout = self.config["api"].get("timeout", 60)
        base_url = self.config["api"].get("base_url", "")
        
        for attempt in range(max_retries):
            try:
                system_msg = "你是一位知识图谱专家，擅长从文本中提取实体、关系并构建结构化知识。"
                
                if "openrouter" in base_url:
                    response = self.client.chat.completions.create(
                        model=self.config["api"].get("model", "qwen/qwen3-8b"),
                        messages=[
                            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}
                        ],
                        temperature=0.3,
                        max_tokens=max_tokens,
                        timeout=timeout
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config["api"].get("model", "deepseek-chat"),
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=max_tokens,
                        timeout=timeout
                    )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _is_classical_chinese(self, text):
        classical_chars = ['之', '乎', '者', '也', '曰', '乃', '遂', '焉', '矣', '哉']
        count = sum(1 for c in text if c in classical_chars)
        return count >= 5

    def build_kg(self, text, literature_type="auto"):
        if literature_type == "auto":
            literature_type = "classical" if self._is_classical_chinese(text) else "modern"
        
        prompt = f"""请从以下文本中提取知识图谱，以JSON格式输出，包含：
- entities: 实体列表，每个实体有 id, name, type, description
- relationships: 关系列表，包含 source, target, relation

输出格式必须是有效的JSON：
{{
  "entities": [
    {{"id": "e1", "name": "实体名", "type": "类型", "description": "描述"}}
  ],
  "relationships": [
    {{"source": "实体1", "target": "实体2", "relation": "关系类型"}}
  ]
}}

原文：
{text}
"""
        result = self.call_llm(prompt, max_tokens=2500)
        
        if result:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def visualize(self, kg_data, output_path):
        if not kg_data:
            print("无可视化的知识图谱数据")
            return
        
        entities = {e["id"]: e for e in kg_data.get("entities", [])}
        relationships = kg_data.get("relationships", [])
        
        G = nx.DiGraph()
        for e in entities.values():
            G.add_node(e["name"], type=e["type"], desc=e["description"])
        for r in relationships:
            G.add_edge(r["source"], r["target"], relation=r["relation"])
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        node_types = {}
        for n in G.nodes():
            ntype = G.nodes[n].get("type", "未知")
            node_types[ntype] = node_types.get(ntype, len(node_types))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        node_colors = [colors[node_types.get(G.nodes[n].get("type", "未知"), 0) % len(colors)] for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, edge_color='#666666', arrows=True, 
                               arrowsize=15, alpha=0.7, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title("知识图谱", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"图片已保存: {output_path}")
        plt.close()

    def process(self, input_path, output_dir="output", literature_type="auto"):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"=== 正在处理: {input_path.name} ===")
        print(f"文本长度: {len(text)} 字符")
        
        if literature_type == "auto":
            literature_type = "classical" if self._is_classical_chinese(text) else "modern"
            print(f"检测为: {'古文' if literature_type == 'classical' else '现代文学'}")
        
        print("\n[1/2] 正在构建知识图谱...")
        kg_data = self.build_kg(text, literature_type)
        
        json_path = output_dir / f"{input_path.stem}_kg.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"knowledge_graph": kg_data}, f, ensure_ascii=False, indent=2)
        print(f"JSON已保存: {json_path}")
        
        print("[2/2] 正在生成可视化图片...")
        img_path = output_dir / f"{input_path.stem}_kg.png"
        self.visualize(kg_data, img_path)
        
        print(f"\n=== 完成! ===")
        print(f"JSON: {json_path}")
        print(f"图片: {img_path}")
        
        if kg_data:
            print(f"实体: {len(kg_data.get('entities', []))}, 关系: {len(kg_data.get('relationships', []))}")


def main():
    parser = argparse.ArgumentParser(description="一键生成知识图谱及可视化")
    parser.add_argument("input", help="输入TXT文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件")
    parser.add_argument("-t", "--type", choices=["auto", "classical", "modern"], 
                        default="auto", help="文学类型")
    
    args = parser.parse_args()
    
    builder = KGraphBuilder(args.config)
    builder.process(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
