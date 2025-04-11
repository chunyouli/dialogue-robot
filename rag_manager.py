from vector_store import VectorStore
from typing import List, Dict, Optional
import json
import os
from datetime import datetime

class RAGManager:
    def __init__(self, vector_store_path: str = "vector_store.index"):
        """
        初始化RAG管理器
        :param vector_store_path: 向量存储文件路径
        """
        print("开始初始化RAG管理器...")
        try:
            # 检查向量存储文件是否存在
            if not os.path.exists(vector_store_path):
                print(f"向量存储文件不存在，将创建新文件: {vector_store_path}")
                # 创建空的历史记录文件
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            
            # 初始化向量存储
            self.vector_store = VectorStore(index_path=vector_store_path)
            print("向量存储初始化成功")
            
            # 加载历史记录
            self.history_file = "conversation_history.json"
            self.conversation_history = self._load_history()
            print(f"已加载 {len(self.conversation_history)} 条历史对话")
            
        except Exception as e:
            print(f"初始化RAG管理器时出错: {str(e)}")
            # 创建空的向量存储和历史记录
            self.vector_store = VectorStore(index_path=vector_store_path)
            self.conversation_history = []
            print("已创建空的向量存储和历史记录")
    
    def add_conversation(self, user_input: str, system_response: str, 
                        audio_file: Optional[str] = None, timestamp: Optional[str] = None):
        """
        添加对话到历史记录和向量存储
        :param user_input: 用户输入
        :param system_response: 系统回复
        :param audio_file: 音频文件路径
        :param timestamp: 时间戳
        """
        print(f"添加新对话 - 用户输入: {user_input[:50]}...")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # 创建对话记录
        conversation = {
            "user_input": user_input,
            "system_response": system_response,
            "audio_file": audio_file,
            "timestamp": timestamp
        }
        
        # 添加到历史记录
        self.conversation_history.append(conversation)
        self._save_history()
        
        # 添加到向量存储
        try:
            texts = [user_input, system_response]
            metadatas = [
                {"type": "user_input", "timestamp": timestamp, "text": user_input},
                {"type": "system_response", "timestamp": timestamp, "text": system_response}
            ]
            self.vector_store.add_texts(texts, metadatas)
            print("对话已成功添加到向量存储")
        except Exception as e:
            print(f"添加对话到向量存储时出错: {str(e)}")
        
    def retrieve_relevant_history(self, query: str) -> List[Dict]:
        """
        检索相关历史对话
        :param query: 查询文本
        :return: 相关历史对话列表
        """
        print(f"检索相关历史对话 - 查询: {query[:50]}...")
        try:
            # 获取最近的20条对话
            if self.conversation_history:
                # 返回最近的20条对话
                return self.conversation_history[-20:]
            return []
        except Exception as e:
            print(f"检索历史对话时出错: {str(e)}")
            return []
    
    def generate_prompt_with_context(self, query: str) -> str:
        """
        生成带上下文的提示词
        :param query: 当前查询
        :return: 增强后的提示词
        """
        print(f"生成带上下文的提示词 - 查询: {query[:50]}...")
        try:
            # 获取相关历史
            relevant_history = self.retrieve_relevant_history(query)
            
            if relevant_history:
                # 构建带上下文的提示词
                prompt = "以下是最近的20条对话历史：\n\n"
                for i, conv in enumerate(relevant_history, 1):
                    prompt += f"对话{i}:\n"
                    prompt += f"用户: {conv['user_input']}\n"
                    prompt += f"系统: {conv['system_response']}\n\n"
                prompt += f"当前对话：\n用户: {query}\n系统: "
                return prompt
            else:
                # 如果没有历史记录，直接使用当前查询
                return query
        except Exception as e:
            print(f"生成提示词时出错: {str(e)}")
            return query
    
    def _load_history(self) -> List[Dict]:
        """加载对话历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"加载历史记录时出错: {str(e)}")
            return []
    
    def _save_history(self):
        """保存对话历史"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"历史记录已保存到 {self.history_file}")
        except Exception as e:
            print(f"保存历史记录时出错: {str(e)}")
            
    def clear_history(self):
        """清空历史记录"""
        print("正在清空历史记录...")
        self.conversation_history = []
        self._save_history()
        self.vector_store.clear()
        print("历史记录已清空") 