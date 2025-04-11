import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 dimension: int = 384, index_path: str = "vector_store.index"):
        """
        初始化向量存储
        :param model_name: 使用的句子嵌入模型名称
        :param dimension: 向量维度
        :param index_path: FAISS索引文件路径
        """
        print("开始初始化向量存储...")
        try:
            # 初始化模型
            print(f"加载句子嵌入模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = dimension
            self.index_path = index_path
            self.metadata_path = index_path.replace('.index', '_metadata.json')
            
            # 初始化或加载FAISS索引
            if os.path.exists(index_path) and os.path.exists(self.metadata_path):
                print("加载现有向量存储...")
                self.index = faiss.read_index(index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"已加载 {len(self.metadata)} 条历史记录")
            else:
                print("创建新的向量存储...")
                # 创建基础索引
                self.index = faiss.IndexFlatL2(dimension)
                self.metadata = []
                # 保存初始化的索引和元数据
                self._save()
                
            print("向量存储初始化完成")
        except Exception as e:
            print(f"初始化向量存储时出错: {str(e)}")
            # 创建空的索引和元数据
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []
            print("已创建空的向量存储")
            
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """
        添加文本到向量存储
        :param texts: 文本列表
        :param metadatas: 元数据列表
        """
        if not texts:
            return
            
        # 生成文本嵌入
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        # 添加到索引
        self.index.add(embeddings)
        
        # 添加元数据
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
            
        # 保存更新后的索引和元数据
        self._save()
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似文本
        :param query: 查询文本
        :param k: 返回结果数量
        :return: 相似文本列表，包含文本、相似度和元数据
        """
        # 如果metadata为空，直接返回空列表
        if not self.metadata:
            return []
            
        # 生成查询向量
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
        
        # 搜索相似向量
        distances, indices = self.index.search(query_embedding, k)
        
        # 获取结果
        results = []
        for i, idx in enumerate(indices[0]):
            # 确保索引在有效范围内
            if 0 <= idx < len(self.metadata):
                results.append((
                    self.metadata[idx].get('text', ''),
                    float(distances[0][i]),
                    self.metadata[idx]
                ))
                
        return results
    
    def _save(self):
        """保存索引和元数据"""
        print("保存向量存储...")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(self.metadata)} 条记录")
            
    def clear(self):
        """清空向量存储"""
        # 创建基础索引
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save() 