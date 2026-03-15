import os
import yaml
import json
import time
import re
from pathlib import Path
from openai import OpenAI

def load_config(config_path="config.yaml"):
    project_root = Path(__file__).parent
    full_path = project_root / config_path
    
    if not full_path.exists():
        default_config = {
            "api": {
                "key": "sk-your-api-key-here",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "timeout": 60,
                "max_retries": 3
            },
            "output": {
                "knowledge_graph_file": "output/knowledge_graph.json",
                "summary_file": "output/summary.txt"
            }
        }
        with open(full_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, allow_unicode=True)
        return default_config
    
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

    def summarize(self, text, literature_type="auto"):
        if literature_type == "auto":
            if self._is_classical_chinese(text):
                literature_type = "classical"
            else:
                literature_type = "modern"
        
        prompts = {
            "classical": """请对以下古文进行知识总结，包括：
1. 文章主旨
2. 主要人物
3. 关键情节
4. 重要地点
5. 主题思想

请用中文输出。""",
            "modern": """请对以下现代文学/文章进行知识总结，包括：
1. 主题/核心思想
2. 主要人物及其特点
3. 重要情节/事件
4. 故事背景/场景
5. 作品风格/意义

请用中文输出。"""
        }
        
        prompt = f"{prompts.get(literature_type, prompts['modern'])}\n\n原文：\n{text}"
        return self.call_llm(prompt, max_tokens=1500)

    def build_kg(self, text, literature_type="auto"):
        if literature_type == "auto":
            if self._is_classical_chinese(text):
                literature_type = "classical"
            else:
                literature_type = "modern"
        
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

    def _is_classical_chinese(self, text):
        classical_chars = ['之', '乎', '者', '也', '曰', '乃', '遂', '焉', '矣', '哉']
        count = sum(1 for c in text if c in classical_chars)
        return count >= 5

    def process_file(self, input_path, output_dir="output", literature_type="auto"):
        input_path = Path(input_path)
        
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"=== 正在分析: {input_path.name} ===")
        print(f"文本长度: {len(text)} 字符")
        
        if literature_type == "auto":
            if self._is_classical_chinese(text):
                literature_type = "classical"
                print("检测为: 古文")
            else:
                literature_type = "modern"
                print("检测为: 现代文学")
        
        print("\n[1/2] 正在进行知识总结...")
        summary = self.summarize(text, literature_type)
        
        print("[2/2] 正在构建知识图谱...")
        kg_data = self.build_kg(text, literature_type)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        result = {
            "source_file": str(input_path.name),
            "literature_type": literature_type,
            "summary": summary,
            "knowledge_graph": kg_data
        }
        
        output_path = output_dir / f"{input_path.stem}_knowledge.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if kg_data:
            print(f"\n=== 完成! ===")
            print(f"实体数量: {len(kg_data.get('entities', []))}")
            print(f"关系数量: {len(kg_data.get('relationships', []))}")
        
        print(f"结果已保存至: {output_path}")
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="知识图谱构建工具")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-t", "--type", choices=["auto", "classical", "modern"], 
                        default="auto", help="文学类型")
    
    args = parser.parse_args()
    
    builder = KGraphBuilder(args.config)
    builder.process_file(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
