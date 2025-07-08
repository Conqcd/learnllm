from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import fire
from os import getenv
from openai import OpenAI
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message, UserMessage
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from typing import List, Dict, Optional, Any

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")

def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"
def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)
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
        query_vector = self.embed_model.create(model=self.model_name, input=query)
        # 2. 执行向量搜索
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.data[0].embedding,
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
    async def run(self, websearch:str, query: str) -> str:
        logger.info(f"⚠️ 研究专家开始分析: {query[:30]}...")
        prompt = f"""
        ## 用户查询 ##
        {query}

        ## 案例搜索结果和用户合同内容 ##
        {websearch}

        ## 任务 ##
        根据上述资料，请从以下步骤提供专业分析：
        1.Find and cite relevant legal cases and precedents,
        2.Provide detailed research summaries with sources,
        3.Reference specific sections from the uploaded document,
        4.Always search the knowledge base for relevant information

        输出格式：Markdown 法律案例报告
        """
        return await self._aask(prompt)
class LegalResearcher(Role):
    def __init__(self,
            collection_name: str = "knowledge_base",
            qdrant: QdrantClient = getenv("QDRANT_URL"),
            **kwargs):
        super().__init__(**kwargs)

        self.qdrantAction = QdrantRAGAction(qdrant=qdrant, collection_name=collection_name)

        self.set_actions([MakeResearch])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.name not in task_msg.send_to:
            return Message(content="非分配任务", role=self.profile,cause_by="null")

        key_categories = [
            "合同标的与范围",
            "履行期限与交付",
            "双方权利义务",
            "价格与支付条款",
            "保证与陈述",
            "违约责任与赔偿",
            "合同期限与终止条件",
            "争议解决与管辖",
            "保密与知识产权",
            "不可抗力条款"
        ]

        results = {}
        for cat in key_categories:
            # 每次检索时，将分类名称作为 query，让 Agent 去拉相关段落
            prompt = f"请检索这份合同中与「{cat}」相关的所有条款，并做简要归纳。"
            item = await self.qdrantAction.run(prompt)
            contract = "法律条款" + item
            case = await DuckDuckGoSearch().run(query=contract)
            results[cat] = "法律条款" + item + "\n\n" + "案例搜索结果" + case

        legal_search = "\n".join(f"{k}: {v}" for k, v in results.items())
        # 执行法律总结
        legal_summarize = await self.rc.todo.run(websearch = legal_search ,query= task_msg.content)

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
        
        ## 用户合同内容 ##
        {contract}

        ## 任务 ##
        请从以下步骤进行专业分析：
        1.Review contracts thoroughly,
        2.Identify key terms and potential issues,
        3.Reference specific clauses from the document

        输出格式：Markdown 法律合同分析
        """
        return await self._aask(prompt)
class ContractAnalyst(Role):
    def __init__(self,
                collection_name: str = "knowledge_base",
                qdrant: QdrantClient = getenv("QDRANT_URL"),
                 **kwargs):
        super().__init__(**kwargs)

        self.qdrantAction = QdrantRAGAction(qdrant=qdrant, collection_name=collection_name)

        self.set_actions([MakeAnalyst])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.name not in task_msg.send_to:
            return Message(content="非分配任务", role=self.profile,cause_by="null")

        key_categories = [
            "合同标的与范围",
            "履行期限与交付",
            "双方权利义务",
            "价格与支付条款",
            "保证与陈述",
            "违约责任与赔偿",
            "合同期限与终止条件",
            "争议解决与管辖",
            "保密与知识产权",
            "不可抗力条款"
        ]

        results = {}
        for cat in key_categories:
            # 每次检索时，将分类名称作为 query，让 Agent 去拉相关段落
            prompt = f"请检索这份合同中与「{cat}」相关的所有条款，并做简要归纳。"
            results[cat] = await self.qdrantAction.run(prompt)

        contract = "\n".join(f"{k}: {v}" for k, v in results.items())

        # 执行法律分析
        legal_analysis = await self.rc.todo.run(contract=contract,query=task_msg.content)

        # 将分析结果发送给领导
        return Message(
            content=legal_analysis,
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
        你是法律战略，为用户定制做法，符合法律且有利于用户。

        ## 任务 ##
        请从以下角度提供专业分析：
        1.Develop comprehensive legal strategies,
        2.Provide actionable recommendations,
        3.Consider both risks and opportunities

        输出格式：Markdown 风险评估报告
        """
        return await self._aask(prompt)
class LegalStrategist(Role):

    def __init__(self,
                collection_name: str = "knowledge_base",
                qdrant: QdrantClient = getenv("QDRANT_URL"),
                 **kwargs):
        super().__init__(**kwargs)

        self.qdrantAction = QdrantRAGAction(qdrant=qdrant, collection_name=collection_name)

        self.set_actions([MakeStratege])
        # 监听用户查询和专家报告
        self._watch([TaskAssignment])


    async def _act(self) -> Message:
        # 获取领导分配的任务
        task_msg = self.rc.memory.get_by_action(TaskAssignment)[0]

        # 检查是否是给自己的任务
        if self.name not in task_msg.send_to:
            return Message(content="非分配任务", role=self.profile,cause_by="null")

        key_categories = [
            "合同标的与范围",
            "履行期限与交付",
            "双方权利义务",
            "价格与支付条款",
            "保证与陈述",
            "违约责任与赔偿",
            "合同期限与终止条件",
            "争议解决与管辖",
            "保密与知识产权",
            "不可抗力条款"
        ]

        results = {}
        for cat in key_categories:
            # 每次检索时，将分类名称作为 query，让 Agent 去拉相关段落
            prompt = f"请检索这份合同中与「{cat}」相关的所有条款，并做简要归纳。"
            results[cat] = await self.qdrantAction.run(prompt)

        contract = "\n".join(f"{k}: {v}" for k, v in results.items())

        # 执行法律战略计划
        legal_stratege = await self.rc.todo.run(contract=contract,query=task_msg.content)

        # 将分析结果发送给领导
        return Message(
            content=legal_stratege,
            role=self.profile,
            cause_by=MakeStratege,
            send_to=task_msg.sent_from  # 发送给领导
        )


class TaskAssignment(Action):
    """任务分配动作 - 决定由谁处理查询"""

    async def run(self, query: str) -> dict:
        """分析查询内容并决定分配哪些专家"""
        prompt = f"""
        ## 用户查询 和 专家定义 ##
        {query}

        ## 你的任务 ##
        1. 分析查询内容，选择上面的所述的专家，专家类型要用英文输出，与上述保持一致
        2. 为每位专家分配具体的分析任务
        3.Always search the knowledge base before delegating tasks

        ## 输出格式 ##
        输出格式为 JSON，类似于以下字段：
        {{
            "分配说明": "简要说明分配理由",
            "分配列表": [
                {{
                    "专家类型": "专家1类型",
                    "任务描述": "具体分析任务描述"
                }},
                {{
                    "专家类型": "专家2类型",
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

    async def run(self, report_text: str) -> str:
        prompt = f"""

        You are the Leader of the legal team (“Legal Team Leader Agent”). Your mission is to:

        1. 收集并阅读所有团队成员最新一次的输出结果。
        2. 按照“要点—细节—风险”三个层次，对各成员的结论和发现进行梳理：  
           a. 要点：提炼最核心的结论或建议；
           b. 细节：列出支持结论的关键事实、法规条文、判例或论证逻辑；
           c. 风险：识别潜在法律风险、冲突或遗漏，并标注优先级。
        3. 对比各成员之间的观点和数据，若存在不一致或矛盾，标出并提出需要进一步验证或讨论的问题清单。
        4. 基于上述汇总，生成一份“法律风险评估与下一步行动”建议，包括：  
           - 主要关注点（Top 3）；
           - 建议的补充调研方向；
           - 预计的时间节点和负责人（若团队已有明确分工）。
        5. 保持语言客观、中立，并注重条理清晰，确保整个团队都能快速阅读、理解并执行。
        6. 输出格式：
           - 摘要（不超过 5 行）
           - 详细汇总（分项列表）
           - 风险与建议（表格或列点形式）
        7.Coordinate analysis between team members,
        8.Provide comprehensive responses,
        9.Ensure all recommendations are properly sourced,
        10.Reference specific parts of the uploaded document,
        
        Begin: 
        “以下是法律团队各成员的最新输出摘要，请从中提炼核心要点并完成上述任务：”  
        然后依次粘入各成员内容，启动自动汇总。
        {report_text}
        """
        return await self._aask(prompt)

class TeamLeader(Role):
    """团队领导角色 - 动态分配任务并汇总结果"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([TaskAssignment])
        # 监听用户查询和专家报告
        self._watch([UserMessage, MakeResearch,MakeAnalyst, MakeStratege])
        self.assigned_tasks = {}
        self.expected_reports = 0  # 预期收到的报告数量
        self.received_reports = []

    async def _observe(self):
        """观察环境中的消息"""
        num :int = await super()._observe()

        # 收集专家报告
        for msg in self.rc.memory.get():
            if msg.cause_by in ['legalTeamAgent.MakeResearch', 'legalTeamAgent.MakeAnalyst',
                                'legalTeamAgent.MakeStratege']:
                self.received_reports.append({
                    "role": msg.role,
                    "content": msg.content
                })
                logger.info(f"📥 收到 {msg.role} 的报告")
        return num

    async def _act(self) -> Message:
        """领导的核心决策逻辑"""
        # 获取最新消息
        latest_msg = self.rc.memory.get()[-1]

        # 处理用户查询 - 分配任务
        if latest_msg.cause_by == any_to_str(UserMessage) or latest_msg.cause_by == any_to_str(UserRequirement):
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
            send_to=set(experts_to_assign),  # 发送给所有专家
            original_query=query  # 保存原始查询
        )

    async def _summarize_reports(self) -> Message:
        """汇总所有专家报告"""

        report_text = "\n\n".join([
            f"## {report['role']}报告 ##\n{report['content']}"
            for report in self.received_reports
        ])

        # 汇总报告
        summary = await SummarizeReports().run(report_text)

        full_report = f"""
        # 综合决策报告
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