"""
Redis 缓存管理器
用于缓存检索结果和生成的回答，提高系统性能
"""
import redis
import json
import hashlib
from typing import Any, Optional, List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Redis 缓存管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,  # 缓存过期时间（秒），默认 1 小时
    ):
        """
        初始化 Redis 缓存管理器
        
        Args:
            host: Redis 主机地址
            port: Redis 端口
            db: Redis 数据库编号
            password: Redis 密码
            ttl: 缓存过期时间（秒）
        """
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # 测试连接
            self.redis_client.ping()
            logger.info(f"✅ Redis 缓存连接成功: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"❌ Redis 连接失败: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, content: str) -> str:
        """
        生成缓存键
        
        Args:
            prefix: 键前缀（如 'search', 'answer'）
            content: 要缓存的内容
        
        Returns:
            缓存键
        """
        # 使用 MD5 哈希生成唯一键
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{prefix}:{content_hash}"
    
    def get(self, prefix: str, content: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            prefix: 键前缀
            content: 要缓存的内容
        
        Returns:
            缓存的内容，如果不存在则返回 None
        """
        if self.redis_client is None:
            return None
        
        try:
            cache_key = self._generate_cache_key(prefix, content)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"🎯 缓存命中: {cache_key[:20]}...")
                return json.loads(cached_data)
            else:
                logger.debug(f"❌ 缓存未命中: {cache_key[:20]}...")
                return None
        except Exception as e:
            logger.error(f"❌ 获取缓存失败: {e}")
            return None
    
    def set(self, prefix: str, content: str, value: Any) -> bool:
        """
        设置缓存
        
        Args:
            prefix: 键前缀
            content: 要缓存的内容
            value: 要缓存的值
        
        Returns:
            是否设置成功
        """
        if self.redis_client is None:
            return False
        
        try:
            cache_key = self._generate_cache_key(prefix, content)
            # 将值序列化为 JSON
            serialized_value = json.dumps(value, ensure_ascii=False)
            
            # 设置缓存，并指定过期时间
            self.redis_client.setex(cache_key, self.ttl, serialized_value)
            logger.info(f"💾 缓存已保存: {cache_key[:20]}... (TTL: {self.ttl}s)")
            return True
        except Exception as e:
            logger.error(f"❌ 设置缓存失败: {e}")
            return False
    
    def get_search_cache(self, query: str) -> Optional[List[Document]]:
        """
        获取检索缓存
        
        Args:
            query: 查询字符串
        
        Returns:
            缓存的文档列表，如果不存在则返回 None
        """
        cached_data = self.get("search", query)
        if cached_data is None:
            return None
        
        # 将缓存的字典数据转换回 Document 对象
        documents = []
        for doc_dict in cached_data:
            doc = Document(
                page_content=doc_dict["page_content"],
                metadata=doc_dict["metadata"],
            )
            documents.append(doc)
        
        return documents
    
    def set_search_cache(self, query: str, documents: List[Document]) -> bool:
        """
        设置检索缓存
        
        Args:
            query: 查询字符串
            documents: 文档列表
        
        Returns:
            是否设置成功
        """
        # 将 Document 对象转换为字典
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        
        return self.set("search", query, doc_dicts)
    
    def get_answer_cache(self, query: str, context: str) -> Optional[str]:
        """
        获取回答缓存
        
        Args:
            query: 查询字符串
            context: 上下文（检索到的法律条文）
        
        Returns:
            缓存的回答，如果不存在则返回 None
        """
        # 组合查询和上下文作为缓存键
        cache_content = f"{query}|||{context[:200]}"  # 只使用前 200 个字符
        return self.get("answer", cache_content)
    
    def set_answer_cache(self, query: str, context: str, answer: str) -> bool:
        """
        设置回答缓存
        
        Args:
            query: 查询字符串
            context: 上下文（检索到的法律条文）
            answer: 生成的回答
        
        Returns:
            是否设置成功
        """
        # 组合查询和上下文作为缓存键
        cache_content = f"{query}|||{context[:200]}"  # 只使用前 200 个字符
        return self.set("answer", cache_content, answer)
    
    def clear_cache(self, prefix: Optional[str] = None) -> bool:
        """
        清除缓存
        
        Args:
            prefix: 键前缀，如果为 None 则清除所有缓存
        
        Returns:
            是否清除成功
        """
        if self.redis_client is None:
            return False
        
        try:
            if prefix:
                # 清除指定前缀的所有键
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"🗑️  已清除缓存: {prefix} (共 {len(keys)} 个键)")
            else:
                # 清除所有键
                self.redis_client.flushdb()
                logger.info("🗑️  已清除所有缓存")
            return True
        except Exception as e:
            logger.error(f"❌ 清除缓存失败: {e}")
            return False
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        if self.redis_client is None:
            return {"status": "disconnected"}
        
        try:
            info = self.redis_client.info()
            db_stats = info.get("keyspace", {})
            # 获取 db0 的键数
            db0_keys = 0
            for key, value in db_stats.items():
                if key.startswith("db"):
                    # 解析 "db0:keys=10,expires=0,avg_ttl=0" 格式
                    parts = value.split(",")
                    for part in parts:
                        if part.startswith("keys="):
                            db0_keys += int(part.split("=")[1])
            
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": db0_keys,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"❌ 获取缓存统计失败: {e}")
            return {"status": "error", "error": str(e)}


# 创建全局缓存管理器实例
_cache_manager: Optional[RedisCacheManager] = None


def get_cache_manager() -> RedisCacheManager:
    """
    获取全局缓存管理器实例（单例模式）
    
    Returns:
        Redis 缓存管理器实例
    """
    global _cache_manager
    
    if _cache_manager is None:
        # 从环境变量读取 Redis 配置
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        redis_ttl = int(os.getenv("REDIS_TTL", "3600"))
        
        _cache_manager = RedisCacheManager(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            ttl=redis_ttl,
        )
    
    return _cache_manager
