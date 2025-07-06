from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import fire
from os import getenv
from openai import OpenAI
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message, UserMessage
from metagpt.actions.add_requirement import UserRequirement
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from typing import List, Dict, Optional

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")

class QdrantRAGAction(Action):
    """
    使用 Qdrant 向量数据库实现检索增强生成（RAG）的 Action
    功能：
    1. 将用户查询转换为向量
    2. 从 Qdrant 检索相关文档
    3. 将检索结果注入 LLM 上下文
    """
    name: str = "QdrantRAG"
    def __init__(
            self,
            collection_name: str = "knowledge_base",
            qdrant: QdrantClient = getenv("QDRANT_URL"),
            embed_model: str = "text-embedding-ada-002",
            top_k: int = 3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.top_k = top_k

        # 初始化 Qdrant 客户端
        self.client = qdrant

        # 初始化嵌入模型（本地运行，无需API密钥）
        emb = OpenAI(base_url=getenv("OpenAI_API_EMBEDDING_BASE"), api_key=getenv("OPENAI_API_KEY"))

        self.embed_model = emb.embeddings
        self.model_name = embed_model
        logging.info(f"✅ Qdrant RAG Action 初始化完成 | 集合: {collection_name} | 模型: {embed_model}")

    async def run(self, query: str, **kwargs) -> str:
        """
        执行 RAG 检索

        参数:
        query: 用户查询文本
        filters: 元数据过滤条件 (例如 {"source": "manual.pdf", "year": 2023})

        返回:
        检索到的相关文本（拼接后的字符串）
        """
        # 1. 将查询转换为向量
        query_vector = self.embed_model.create(model=self.model_name, input=query).tolist()
        # 2. 执行向量搜索
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
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

class DuckDuckGoSearch(Action):
    """
    DuckDuckGo 搜索 Action
    支持三种搜索模式：网页搜索、新闻搜索、图片搜索
    """

    name: str = "DuckDuckGoSearch"

    def __init__(
            self,
            default_max_results: int = 5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.default_max_results = default_max_results
        logging.info("🦆 DuckDuckGoSearch Action 初始化完成")

    async def run(
            self,
            query: str,
            search_type: str = "text",  # text, news
            max_results: Optional[int] = None,
            region: str = "wt-wt",  # 地区代码，默认全球
            safesearch: str = "moderate",  # off, moderate, strict
            timelimit: Optional[str] = None,  # d (天), w (周), m (月)
            **kwargs
    ) -> str:
        """
        执行 DuckDuckGo 搜索

        参数:
        - query: 搜索关键词
        - search_type: 搜索类型 (text, news, images)
        - max_results: 返回结果数量
        - region: 地区代码 (如: us-en, cn-zh, ru-ru)
        - safesearch: 安全搜索级别
        - timelimit: 时间限制 (d, w, m)

        返回:
        格式化后的搜索结果字符串或 JSON
        """
        if not max_results:
            max_results = self.default_max_results

        try:
            with DDGS() as ddgs:
                if search_type == "text":
                    results = self._text_search(ddgs, query, max_results, region, safesearch, timelimit)
                elif search_type == "news":
                    results = self._news_search(ddgs, query, max_results, region, safesearch, timelimit)
                else:
                    raise ValueError(f"不支持的搜索类型: {search_type}")

                return self.format_results(results, search_type)

        except Exception as e:
            logging.error(f"❌ DuckDuckGo 搜索失败: {str(e)}")
            return json.dumps({"error": f"搜索失败: {str(e)}"})

    def _text_search(
            self,
            ddgs: DDGS,
            query: str,
            max_results: int,
            region: str,
            safesearch: str,
            timelimit: Optional[str]
    ) -> List[Dict]:
        """执行文本搜索"""
        return [
            {
                "title": r["title"],
                "url": r["href"],
                "description": r["body"],
                "source": r.get("source", "Unknown")
            }
            for r in ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
        ]

    def _news_search(
            self,
            ddgs: DDGS,
            query: str,
            max_results: int,
            region: str,
            safesearch: str,
            timelimit: Optional[str]
    ) -> List[Dict]:
        """执行新闻搜索"""
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "description": r["body"],
                "source": r["source"],
                "date": r["date"]
            }
            for r in ddgs.news(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
        ]
    def format_results(
            self,
            results: List[Dict],
            search_type: str
    ) -> str:
        """格式化搜索结果"""
        if not results:
            return "🔍 未找到相关搜索结果"

        if search_type == "text":
            return self._format_text_results(results)
        elif search_type == "news":
            return self._format_news_results(results)
        else:
            return json.dumps(results, ensure_ascii=False, indent=2)

    def _format_text_results(self, results: List[Dict]) -> str:
        """格式化文本搜索结果"""
        formatted = ["📄 网页搜索结果:"]
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. [{r['title']}]({r['url']})\n"
                f"   {r['description'][:200]}...\n"
                f"   来源: {r['source']}"
            )
        return "\n\n".join(formatted)

    def _format_news_results(self, results: List[Dict]) -> str:
        """格式化新闻搜索结果"""
        formatted = ["📰 新闻搜索结果:"]
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. [{r['title']}]({r['url']})\n"
                f"   {r['description'][:200]}...\n"
                f"   来源: {r['source']} | 日期: {r['date']}"
            )
        return "\n\n".join(formatted)

class MakeResearch(Action):
    async def run(self, contract:str, websearch:str, query: str) -> str:
        logger.info(f"⚠️ 研究专家开始分析: {query[:30]}...")
        prompt = f"""
        ## 用户查询 ##
        {query}

        ## 你的角色 ##
        你是风险控制专家，擅长识别和评估各类商业和技术风险。

        ## 任务 ##
        请从以下角度提供专业分析：
        1. 技术实施风险
        2. 市场接受度风险
        3. 合规与法律风险
        4. 财务与运营风险
        5. 风险缓解策略

        输出格式：Markdown 风险评估报告
        """
        return await self._aask(prompt)
class LegalResearcher(Role):
    def __init__(self,
            collection_name: str = "knowledge_base",
            qdrant: QdrantClient = getenv("QDRANT_URL"),
            **kwargs):
        super().__init__(**kwargs)

        self.set_actions([QdrantRAGAction(qdrant=qdrant, collection_name=collection_name),DuckDuckGoSearch,MakeResearch])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.profile not in task_msg.instruct_to:
            return Message(content="非分配任务", role=self.profile)

        # 执行搜索qdrant数据库
        legal_search = await self.rc.todo[0].run(task_msg.content)

        # 执行搜索duckduckgo
        duckduckgo_search = await self.rc.todo[1].run(task_msg.content)
        # 执行法律总结
        legal_summarize = await self.rc.todo[2].run(contract =legal_search,websearch = duckduckgo_search ,query= task_msg.content)

        # 将分析结果发送给领导
        return Message(
            content=legal_summarize,
            role=self.profile,
            cause_by=MakeResearch,
            send_to=task_msg.sent_from  # 发送给领导
        )


class MakeAnalyst(Action):
    async def run(self, contract: str, query: str) -> str:
        logger.info(f"⚠️ 研究专家开始分析: {query[:30]}...")
        prompt = f"""
        ## 用户查询 ##
        {query}

        ## 你的角色 ##
        你是风险控制专家，擅长识别和评估各类商业和技术风险。

        ## 任务 ##
        请从以下角度提供专业分析：
        1. 技术实施风险
        2. 市场接受度风险
        3. 合规与法律风险
        4. 财务与运营风险
        5. 风险缓解策略

        输出格式：Markdown 风险评估报告
        """
        return await self._aask(prompt)
class ContractAnalyst(Role):
    def __init__(self,
                collection_name: str = "knowledge_base",
                qdrant: QdrantClient = getenv("QDRANT_URL"),
                 **kwargs):
        super().__init__(**kwargs)

        self.set_actions([QdrantRAGAction(qdrant=qdrant, collection_name=collection_name),MakeAnalyst])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.profile not in task_msg.instruct_to:
            return Message(content="非分配任务", role=self.profile)

        # 执行法律分析
        biz_analysis = await self.rc.todo.run(task_msg.content)

        # 将分析结果发送给领导
        return Message(
            content=biz_analysis,
            role=self.profile,
            cause_by=MakeAnalyst,
            send_to=task_msg.sent_from  # 发送给领导
        )


class MakeStratege(Action):
    """风险专家分析 - 关注潜在风险"""

    async def run(self, contract: str, query: str) -> str:
        logger.info(f"⚠️ 风险专家开始分析: {query[:30]}...")
        prompt = f"""
        ## 用户查询 ##
        {query}

        ## 你的角色 ##
        你是风险控制专家，擅长识别和评估各类商业和技术风险。

        ## 任务 ##
        请从以下角度提供专业分析：
        1. 技术实施风险
        2. 市场接受度风险
        3. 合规与法律风险
        4. 财务与运营风险
        5. 风险缓解策略

        输出格式：Markdown 风险评估报告
        """
        return await self._aask(prompt)
class LegalStrategist(Role):

    def __init__(self,
                collection_name: str = "knowledge_base",
                qdrant: QdrantClient = getenv("QDRANT_URL"),
                 **kwargs):
        super().__init__(**kwargs)


        self.set_actions([QdrantRAGAction(qdrant = qdrant, collection_name=collection_name),MakeStratege])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.profile not in task_msg.instruct_to:
            return Message(content="非分配任务", role=self.profile)

        # 执行法律战略计划
        biz_analysis = await self.rc.todo.run(task_msg.content)

        # 将分析结果发送给领导
        return Message(
            content=biz_analysis,
            role=self.profile,
            cause_by=MakeStratege,
            send_to=task_msg.sent_from  # 发送给领导
        )


class TaskAssignment(Action):
    """任务分配动作 - 决定由谁处理查询"""

    async def run(self, query: str) -> dict:
        """分析查询内容并决定分配哪些专家"""
        prompt = f"""
        ## 用户查询 ##
        {query}

        ## 专家团队 ##
        1. 技术专家 - 负责技术可行性分析
        2. 商业分析师 - 负责市场与商业模式分析
        3. 用户体验专家 - 负责用户需求与体验设计
        4. 风险控制专家 - 负责风险评估

        ## 你的任务 ##
        1. 分析查询内容，确定需要哪些专家参与
        2. 为每位专家分配具体的分析任务
        3. 说明分配理由

        ## 输出格式 ##
        {{
            "分配说明": "简要说明分配理由",
            "分配列表": [
                {{
                    "专家类型": "技术专家",
                    "任务描述": "具体分析任务描述"
                }},
                {{
                    "专家类型": "商业分析师",
                    "任务描述": "具体分析任务描述"
                }}
            ]
        }}
        """
        assignment = await self._aask(prompt)

        # 尝试解析JSON，如果失败则返回原始文本
        try:
            import json
            return json.loads(assignment)
        except:
            return {"分配说明": "无法解析分配方案", "分配列表": []}


class SummarizeReports(Action):
    """汇总所有分析报告"""

    async def run(self, reports: list) -> str:
        report_text = "\n\n".join([
            f"## {report['role']}报告 ##\n{report['content']}"
            for report in reports
        ])

        prompt = f"""
        ## 分析报告汇总 ##
        {report_text}

        ## 你的任务 ##
        作为团队领导，你需要：
        1. 综合所有专业角度的分析
        2. 识别关键共识与分歧点
        3. 提出综合建议和决策方案
        4. 制定下一步行动计划

        ## 输出格式 ##
        # 综合决策报告
        ## 关键共识
        - 点1
        - 点2

        ## 主要分歧
        - 分歧点1：不同观点分析
        - 分歧点2：不同观点分析

        ## 综合建议
        - 建议1（技术+商业+用户体验平衡）
        - 建议2

        ## 行动计划
        1. 短期行动（1周内）
        2. 中期行动（1个月内）
        3. 长期行动（3个月+）
        """
        return await self._aask(prompt)

class TeamLeader(Role):
    """团队领导角色 - 动态分配任务并汇总结果"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([TaskAssignment, SummarizeReports])
        # 监听用户查询和专家报告
        self._watch([UserRequirement, MakeResearch,MakeAnalyst, MakeStratege])
        self.assigned_tasks = {}
        self.expected_reports = 0
        self.received_reports = []

    async def _observe(self):
        """观察环境中的消息"""
        await super()._observe()

        # 收集专家报告
        for msg in self.rc.memory.get():
            if msg.cause_by in [MakeResearch, MakeAnalyst,
                                MakeStratege]:
                self.received_reports.append({
                    "role": msg.role,
                    "content": msg.content
                })
                logger.info(f"📥 收到 {msg.role} 的报告")

    async def _act(self) -> Message:
        """领导的核心决策逻辑"""
        # 获取最新消息
        latest_msg = self.rc.memory.get()[-1]

        # 处理用户查询 - 分配任务
        if isinstance(latest_msg.cause_by, UserRequirement):
            return await self._handle_user_query(latest_msg.content)

        # 处理专家报告 - 汇总结果
        elif len(self.received_reports) >= self.expected_reports:
            return await self._summarize_reports()

        # 等待更多报告
        else:
            return Message(
                content=f"等待报告中 ({len(self.received_reports)}/{self.expected_reports})",
                role=self.profile
            )

    async def _handle_user_query(self, query: str) -> Message:
        """处理用户查询并分配任务"""
        # 分析查询并分配任务
        assignment = await self.rc.todo.run(query)

        # 解析分配方案
        if "分配列表" not in assignment:
            return Message(
                content="无法解析任务分配方案",
                role=self.profile,
                send_to="USER"
            )

        # 准备任务分配消息
        task_descriptions = []
        experts_to_assign = []

        for task in assignment["分配列表"]:
            expert_type = task["专家类型"]
            task_desc = task["任务描述"]
            task_descriptions.append(f"- {expert_type}: {task_desc}")
            experts_to_assign.append(expert_type)

        # 记录预期报告数量
        self.expected_reports = len(experts_to_assign)
        self.received_reports = []  # 重置报告收集

        # 创建任务分配消息
        task_msg = f"""
        ## 任务分配说明 ##
        {assignment.get("分配说明", "基于查询内容分配")}

        ## 具体任务分配 ##
        {chr(10).join(task_descriptions)}
        """

        logger.info(f"📝 领导分配任务给: {', '.join(experts_to_assign)}")

        # 发送给所有专家
        return Message(
            content=task_msg,
            role=self.profile,
            cause_by=TaskAssignment,
            send_to="ALL",  # 发送给所有专家
            instruct_to=experts_to_assign,  # 指定分配给哪些专家
            original_query=query  # 保存原始查询
        )

    async def _summarize_reports(self) -> Message:
        """汇总所有专家报告"""
        # 汇总报告
        summary = await SummarizeReports().run(self.received_reports)

        # 添加任务分配信息
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]
        full_report = f"""
        # 综合决策报告
        ## 原始查询
        {task_msg.original_query}

        ## 任务分配
        {task_msg.content}

        {summary}
        """

        logger.info("✅ 领导完成报告汇总")

        # 发送最终报告给用户
        return Message(
            content=full_report,
            role=self.profile,
            cause_by=SummarizeReports,
            send_to="USER"
        )


def main():
    role = Role()
    role.set_actions([DuckDuckGoSearch(default_max_results=5), QdrantRAGAction(collection_name="nonono")])

def qdrant_trial():

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