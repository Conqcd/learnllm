import subprocess
from dotenv import load_dotenv
load_dotenv()
import asyncio
import duckdb
import fire
from os import getenv
from openai import OpenAI
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader


class QdrantRAGAction(Action):
    """
    使用 Qdrant 向量数据库实现检索增强生成（RAG）的 Action
    功能：
    1. 将用户查询转换为向量
    2. 从 Qdrant 检索相关文档
    3. 将检索结果注入 LLM 上下文
    """

    def __init__(
            self,
            name: str = "QdrantRAG",
            collection_name: str = "knowledge_base",
            qdrant: QdrantClient = getenv("QDRANT_URL"),
            embed_model: str = "text-embedding-ada-002",
            top_k: int = 3,
            **kwargs
    ):
        super().__init__(name, **kwargs)
        self.collection_name = collection_name
        self.top_k = top_k

        # 初始化 Qdrant 客户端
        self.client = qdrant

        # 初始化嵌入模型（本地运行，无需API密钥）
        emb = OpenAI(base_url=getenv("OpenAI_API_EMBEDDING_BASE"), api_key=getenv("OPENAI_API_KEY"))

        self.embed_model = emb.embeddings
        logging.info(f"✅ Qdrant RAG Action 初始化完成 | 集合: {collection_name} | 模型: {embed_model}")

    async def run(self, query: str, filters: dict = None, **kwargs) -> str:
        """
        执行 RAG 检索

        参数:
        query: 用户查询文本
        filters: 元数据过滤条件 (例如 {"source": "manual.pdf", "year": 2023})

        返回:
        检索到的相关文本（拼接后的字符串）
        """
        # 1. 将查询转换为向量
        query_vector = self.embed_model.create(model=self.embed_model,input=query).tolist()

        # 2. 构建元数据过滤器
        qdrant_filter = self._build_filter(filters) if filters else None

        # 3. 执行向量搜索
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=self.top_k,
                with_payload=True  # 返回存储的文本内容
            )

            # 4. 提取并拼接结果
            context = "\n\n".join([
                f"📄 [来源: {hit.payload.get('source', '未知')}]\n{hit.payload['text']}"
                for hit in search_result
            ])

            logging.info(f"🔍 检索成功 | 查询: '{query}' | 返回 {len(search_result)} 条结果")
            return context if context else "未找到相关文档"

        except Exception as e:
            logging.error(f"❌ 检索失败: {str(e)}")
            return "检索服务暂时不可用"

    def _build_filter(self, filters: dict) -> Filter:
        """ 将字典过滤器转换为 Qdrant 过滤器 """
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",  # 假设元数据存储在 metadata 字段
                    match=MatchValue(value=value)
                )
            )
        return Filter(must=conditions)

    def add_documents(self, documents: list[dict]):
        """
        批量添加文档到 Qdrant（初始化知识库时使用）
        文档格式: [{
            "text": "文档内容",
            "metadata": {"source": "file.pdf", "page": 42}
        }]
        """
        points = []
        for doc in documents:
            vector = self.embed_model.encode(doc["text"]).tolist()
            points.append({
                "vector": vector,
                "payload": {
                    "text": doc["text"],
                    "source": doc["metadata"].get("source", "unknown"),
                    **doc["metadata"]  # 展开所有元数据
                }
            })

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logging.info(f"📥 已添加 {len(documents)} 条文档到集合 {self.collection_name}")



def main():

    client = QdrantClient(
        url=getenv("QDRANT_URL"),
        api_key=getenv("QDRANT_API_KEY"),
    )
    # client.create_collection(collection_name="docs", vectors_config=VectorParams(
    #     size=1536,
    #     distance=Distance.COSINE,
    # ),)

    # 批量插入向量（带 Payload 元数据）
    emb = OpenAI(base_url=getenv("OpenAI_API_EMBEDDING_BASE"), api_key=getenv("OPENAI_API_KEY"))
    documents = "The food was delicious and the waiter was very attentive. The ambiance was cozy and inviting, making it a perfect place for a date night or a family gathering. I highly recommend trying the chef's special, which was a delightful blend of flavors and textures. Overall, a wonderful dining experience that I would love to repeat."

    pdf = "中华人民共和国劳动法.pdf"

    documentsa = PDFReader().load_data(file=pdf)
    docstr = [doc.text for doc in documentsa]
    docstr = "".join(docstr)

    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = parser.split_text(documents)

    embedding = emb.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks,
        encoding_format="float"
    )
    points = [
        PointStruct(
            id=idx,
            vector=data.embedding,
            payload={"text": text},
        )
        for idx, (data, text) in enumerate(zip(embedding.data, chunks))
    ]
    client.upsert(collection_name="docs", points=points)


    query = "The food was?"
    qembedding = emb.embeddings.create(
        model="text-embedding-ada-002",
        input=query,
        encoding_format="float"
    )
    # 相似性搜索 + 元数据过滤
    results = client.search(
        collection_name="docs",
        query_vector=qembedding.data[0].embedding,  # 查询向量
        # query_filter={"must": [{"key": "year", "range": {"gte": 2024}}]},  # 过滤条件
        # limit=3
    )
    print(results)

if __name__ == "__main__":
    fire.Fire(main)