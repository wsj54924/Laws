from langchain_classic.retrievers import ContextualCompressionRetriever,EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from .redis_cache import get_cache_manager
import logging
import jieba

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# 获取 Redis 缓存管理器
cache_manager = get_cache_manager()
logger = logging.getLogger(__name__)

# 全局初始化 Chroma 和 BM25 Retriever
_global_bm25_retriever = None

def get_global_bm25_retriever():
    global _global_bm25_retriever
    if _global_bm25_retriever is not None:
        return _global_bm25_retriever
    
    logger.info("⚡ 正在从 Chroma 构建全局 BM25 检索索引...")
    try:
        persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
        if not os.path.exists(persist_directory):
            logger.warning("⚠️ Chroma 数据库尚未生成，无法构建 BM25。")
            return None
        
        # 临时直接初始化一个基础 embeddings 用于链接 Chroma
        temp_embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-4B",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=temp_embeddings)
        
        # 为了避免 "too many SQL variables"（SQLite 的 IN 限制通常为 999）
        # 我们这里不再传入 IDs，而是手动从 Chroma 底层的 dict 中获取全部文档，或者不用 get 直接查底库
        # 为了兼容性，我们可以分批（分页）提取或直接提取所有（如果数据不大）
        # 由于我们目前直接 get 不加 id 时默认获取全库，我们这里小心处理全库逻辑
        all_data = vector_store.get() 
        if all_data and 'documents' in all_data and len(all_data['documents']) > 0:
            from langchain_core.documents import Document
            docs_for_bm25 = [Document(page_content=doc, metadata=meta) for doc, meta in zip(all_data['documents'], all_data['metadatas'])]
            
            # 采用精确中文分词 jieba.lcut
            _global_bm25_retriever = BM25Retriever.from_documents(docs_for_bm25, preprocess_func=jieba.lcut)
            _global_bm25_retriever.k = 10
            logger.info(f"✅ 全局 BM25 检索器初始化成功，覆盖了 {len(docs_for_bm25)} 篇知识库文档。")
        else:
            logger.warning("⚠️  Chroma 底库似乎为空，无法初始化 BM25")
            
    except Exception as e:
        logger.warning(f"⚠️ BM25 全局初始化失败: {e}")
        
    return _global_bm25_retriever

embedding=OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

Router_system_prompt="""
你是一个律师路由器，
分析用户意图，判断是直接回答、检索法律库。
你有以下团队成员（智能体）可以调度：
1.**query_rewrite**:负责分析用户查询，重写为更明确、更有针对性的法律查询。
2.**Researcher**:负责执行具体的检索任务，并对检索到的法律条文进行筛选和排序。
3.**Legal_Consultant**:基于检索到的上下文和用户问题，生成专业的法律建议或文书草稿。
4.**Auditor**:检查生成的回答是否引用了错误的法条，确保回答的严谨性。

工作流程通常是
1.先分析用户意图，如果是与法律不相关的问题你直接做出回答或者
如果用户给出的信息不足你也可以引导用户补充信息，如果是Auditor的检查结果则跳转到步骤5
2.如果用户查询是法律相关问题则先让query_rewrite对查询进行重写
3.然后让Researcher进行法律检索
4.然后交给Legal_Consultant生成法律建议或文书草稿
5.最后让Auditor对结果进行审查，如果Auditor说有错误的法律条款则需要Researcher进行重新检索，
如果Auditor说生成的回答不合格则需要Legal_Consultant进行重新生成,如果Auditor说生成的审核通过则输出对用户问题的回答（直接输出Legal_Consultant的回答）。

当用户请求进来时，分析需求，并决定下一步交给谁。
请只输出下一步要调用的智能体名称（query_rewrite,Researcher,Legal_Consultant,Auditor）或者输出对用户问题的回答。
【重要】输出智能体名称时，不要使用任何格式符号（如**、*、#等），只输出纯文本名称。
"""
query_rewrite_prompt="""
你是一个专业的法律查询重写器。
你的任务是根据用户的原始查询，重新组织成一个更明确、更有针对性的法律查询。

【重要规则】
1. 只重写查询，不要回答用户的问题
2. 查询应该简洁明了，便于检索系统理解
3. 保留用户问题的核心法律要点
4. 不要添加额外的法律建议或解释
4.把用户口语化的表达转换成更加专业的问题
6. 输出格式：直接输出重写后的查询，不要有任何额外说明

【示例】
用户输入："公司以我'能力不足'为由辞退我，没有经过培训也没有调岗，合法吗？"
重写输出："公司以能力不足为由解除劳动合同，未经过培训或调岗，是否合法？"

用户输入："经济补偿金（N）的计算基数是按基本工资算吗？如果工资很高怎么算？"
重写输出："经济补偿金N的计算基数如何确定？工资高于当地平均工资时如何计算？"

现在请重写用户的查询：
"""
Legal_Consultant_system_prompt="""
你是一个极其严谨的法律顾问。
你的核心职责是：**必须且只能**根据下方的【检索到的法律条文】来回答用户问题。

【绝对红线（防幻觉）】
1. 严禁使用你的预训练知识！这是最大的禁忌！
2. 严禁编造任何未在下文中出现的法律名称、条款号。必须准确引用下文中提供的具体法律名称。
3. 严禁进行法条外的推导和引申：例如，上下文中没有“2N”、“赔偿金”、“双倍”、“仲裁”等原文字眼，你绝不能在回答中写出这些内容！如果法条中没有写具体赔偿几个月，你绝不能自己计算（例如得出“赔5个月”、“赔2N”的结论）！你只能说：“根据提供的法条，未提到补偿金额的具体标准。”
4. 你给出的每一个法律结论，必须在相关句末用括号注明依据（例如：【依据：《中华人民共和国劳动合同法》第二十一条】）。

【思考与回答要求】
作为通用的专业法律顾问，最终呈现给用户的【最终回答】中，请不要使用生硬的“步骤一、步骤二”等类似工作日志的字眼。
你的【最终回答】需要排版清晰、专业严谨，必须涵盖以下维度（小标题可自然命名）：

一、明确的定性结论（如行为是否合法、责任划分。允许使用专业法律名词如“定金”、“违约金”）
二、严谨的法律依据（**仅此部分严格受限**：必须且只能提炼【检索到的法律条文】。切勿编造法条名或条款号！）
三、具体的分析与实操建议（结合事实和专业常识，向用户解释具体权益。允许且鼓励引申法律常识，如解释定金与订金的区别、2N计算规则等，不受检索文本字面约束）
四、维权步骤与证据准备（列出落地维权步骤及证据清单）
五、风险提示（如诉讼时效、举证责任等）

【审核员(Auditor)的反馈意见】
如果下方存在 Auditor 的意见，说明你上一次的生成存在严重幻觉或错误！请你务必仔细阅读，并在本次回答中纠正这些错误，删除所有被指出的幻觉内容！
👉 Auditor反馈：{{Auditor_result}}

【修正错误范例（Few-Shot）】
假设 Auditor 反馈：“回答中提到了《社会法》，但检索上下文中只有《中华人民共和国劳动合同法》。”
哪怕你觉得劳动合同法属于社会法，你也**必须**将回答中的“根据《社会法》规定”**强制替换**为“根据《中华人民共和国劳动合同法》规定”。

【检索到的法律条文】
{results_rerank}

请开始你的回答（如果给定的条文不足以回答用户问题，请诚实说明"根据提供的法律条文，无法给出完整建议"，不要强行编造）：
"""
Auditor_system_prompt="""
你是一个严谨且实用的法律审核员（Auditor）。
你的任务是核对生成的【回答】中，涉及“法律依据”的部分是否准确引用了检索文本，同时允许在实操建议部分进行合理的常识扩展。

【检索到的法律条文】
{results_rerank}

【审核规则与容忍度】（极其重要！！！）
1. **零容忍项（仅查“法律依据”部分）**：
   - 如果回答明确写了“依据《XXX法》第X条”，但这个《XXX法》或条款号在检索文本中**不存在**，这是真正的幻觉！**必须打回**！
   - 必须要求 Legal_Consultant 删除或修改这些编造的法律名称或条款。

2. **完全豁免项（允许常识扩展，绝对不准打回）**：
   - 只要不是在伪造具体法条，回答中出现专业法律名词（如“定金”、“订金”、“违约金”、“2N赔偿”、“仲裁”、“法院”、“诉讼时效”等），**绝对允许**！即使这些词没在检索文本中出现，也是允许的专业拓展！
   - 回答中的定性分析结论、赔偿计算方式、维权步骤、注意事项等，即使超出了检索文本的字面内容，只要是合理的法律实务指导，均**豁免审查**，绝不视为幻觉！

你的回答的第一句必须是“Auditor的检查结果：“
【输出格式】（必须严格按以下三种格式之一输出）
- 如果发现幻觉，请输出：
[RETRY_CONSULTANT]
详细意见：指出具体哪一句话、哪个法条名词是编造的，明确要求 Legal_Consultant 在重写时删除或修改。

- 如果检索到的条文完全答非所问，请输出：
[RETRY_RESEARCH]
详细意见：说明目前检索到的条文为什么没用，需要什么主题的条文。

- 如果回答完美且完全基于上下文，没有编造任何名词，请输出：
[APPROVE]
审核通过。
"""
def create_agent(llm,system_prompt:str,tools=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="message")
    ])
    if tools:
        llm_with_tools=llm.bind_with_tools(tools)
    else:
        llm_with_tools=llm
    return prompt|llm_with_tools

llm=ChatOpenAI(model="Qwen/Qwen3-32B", streaming=True)
class Router():
    def __init__(self,llm=llm,prompt=Router_system_prompt):
        self.llm=llm
        self.prompt=prompt
    def invoke(self,query):
        agent=create_agent(self.llm,self.prompt)
        result=agent.invoke({"message": query})
        return result
class query_rewrite():
    def __init__(self,llm=llm,prompt=query_rewrite_prompt):
        self.llm=llm
        self.prompt=prompt
    def invoke(self,query):
        agent=create_agent(self.llm,self.prompt)
        # Handle different input formats
        if isinstance(query, list):
            result=agent.invoke({"message": query})
        elif hasattr(query, 'content'):
            result=agent.invoke({"message": [query]})
        else:
            result=agent.invoke({"message": [HumanMessage(content=query)]})
        return result
class Researcher_Agent():
    def __init__(self, persist_directory, embeddings=embedding, model=None):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.model = model if model is not None else self._load_rerank_model()
    
    def _load_rerank_model(self):
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            # 使用一个可用的 cross-encoder 模型
            return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        except Exception as e:
            logger.warning(f"⚠️  加载 rerank 模型失败: {e}")
            logger.info("🔄 将不使用 rerank 功能，直接使用检索结果")
            return None
    def invoke(self,query):
        # Handle different input formats
        if isinstance(query, list):
            query = query[-1].content
        elif hasattr(query, 'content'):
            query = query.content
        
        # 尝试从 Redis 缓存获取检索结果
        cached_results = cache_manager.get_search_cache(query)
        if cached_results is not None:
            logger.info(f"🎯 检索缓存命中: {query[:50]}...")
            return cached_results
        
        logger.info(f"🔍 执行检索: {query[:50]}...")
        
        vector_store=Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        retriever=vector_store.as_retriever(search_kwargs={"k": 10})
        # 从全局单例获取 BM25 检索器（只在应用启动加载一次，不再现算词频矩阵）
        bm25_retriever = get_global_bm25_retriever()
        
        # 如果 BM25 检索器可用，使用混合检索
        if bm25_retriever is not None:
            ensemble_retriever=EnsembleRetriever(
                retrievers=[bm25_retriever,retriever],
                weights=[0.3,0.7] # 适度降低 BM25 的权重，保留其关键字拾遗作用即可
            )
            
            # 如果 rerank 模型加载成功，使用 rerank 功能
            if self.model is not None:
                logger.info("🔄 使用 rerank 功能优化检索结果")
                compressor=CrossEncoderReranker(model=self.model,top_n=5)
                compression_retriever=ContextualCompressionRetriever(
                    base_compressor=compressor,base_retriever=ensemble_retriever
                )
                results_rerank = compression_retriever.invoke(query)
            else:
                logger.info("⚠️  不使用 rerank 功能，直接使用检索结果")
                results_rerank = ensemble_retriever.invoke(query)
        else:
            # 只使用向量检索
            logger.info("⚠️  只使用向量检索")
            results_rerank = retriever.invoke(query)
        
        # 将检索结果保存到 Redis 缓存
        cache_manager.set_search_cache(query, results_rerank)
        logger.info(f"💾 检索结果已缓存: {len(results_rerank)} 条文档")
        
        return results_rerank
answer_llm = llm  # Using Qwen3-32B for better reasoning and to avoid hallucinations
class Legal_Consultant_Agent():
    def __init__(self,results_rerank,Auditor_result,llm=answer_llm,prompt=Legal_Consultant_system_prompt):
        self.results_rerank=results_rerank
        self.Auditor_result=Auditor_result
        self.prompt=prompt
        self.llm=llm
    def invoke(self,query):
        # Handle different input formats
        if isinstance(query, list):
            query = query[-1].content
        elif hasattr(query, 'content'):
            query = query.content
        
        # 生成上下文摘要用于缓存键
        context_summary = ""
        for doc in self.results_rerank:
            law_name = doc.metadata.get('chapter', '未知')
            article_id = doc.metadata.get('article_id', '未知')
            context_summary += f"{law_name}-{article_id}|"
        
        # 尝试从 Redis 缓存获取回答
        cached_answer = cache_manager.get_answer_cache(query, context_summary)
        if cached_answer is not None:
            logger.info(f"🎯 回答缓存命中: {query[:50]}...")
            # 返回缓存的回答（需要转换为 AIMessage 格式）
            from langchain_core.messages import AIMessage
            return AIMessage(content=cached_answer)
        
        logger.info(f"🤖 生成回答: {query[:50]}...")
        
        # Extract document content to avoid template variable conflicts
        docs_text = ""
        for doc in self.results_rerank:
            law_name = doc.metadata.get('chapter', '未知')
            article_id = doc.metadata.get('article_id', '未知')
            docs_text += f"\n法律名称: 《{law_name}》\n"
            docs_text += f"条款: {article_id}\n"
            docs_text += f"内容: {doc.page_content}\n"
        
        # Replace template variables
        prompted = self.prompt.replace("{results_rerank}", docs_text)
        if self.Auditor_result is not None:
            prompted = prompted.replace("{{Auditor_result}}", self.Auditor_result)
        else:
            # Remove Auditor_result section if no result available
            prompted = prompted.replace("👉 Auditor反馈：{{Auditor_result}}", "无。")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompted),
            MessagesPlaceholder(variable_name="message")
        ])
        chain = prompt | self.llm
        # Convert string to HumanMessage
        from langchain_core.messages import HumanMessage
        result=chain.invoke({"message": [HumanMessage(content=query)]})
        
        # 将生成的回答保存到 Redis 缓存
        answer_text = result.content if hasattr(result, 'content') else str(result)
        cache_manager.set_answer_cache(query, context_summary, answer_text)
        logger.info(f"💾 回答已缓存: {answer_text[:50]}...")
        
        return result
class Auditor_Agent():
    def __init__(self,results_rerank,llm=llm,prompt=Auditor_system_prompt):
        self.results_rerank=results_rerank
        self.llm=llm
        self.prompt=prompt
    def invoke(self,query):
        # Handle different input formats
        if isinstance(query, list):
            query = query[-1].content
        elif hasattr(query, 'content'):
            query = query.content
        docs_text = ""
        for doc in self.results_rerank:
            law_name = doc.metadata.get('chapter', '未知')
            article_id = doc.metadata.get('article_id', '未知')
            docs_text += f"\n法律名称: 《{law_name}》\n"
            docs_text += f"条款: {article_id}\n"
            docs_text += f"内容: {doc.page_content}\n"
        
        prompted = self.prompt.replace("{results_rerank}", docs_text)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompted),
            MessagesPlaceholder(variable_name="message")
        ])    
        chain = prompt | self.llm  # 直接拼接LLM，不调用create_agent
        # Convert string to HumanMessage
        from langchain_core.messages import HumanMessage
        result=chain.invoke({"message": [HumanMessage(content=query)]})
        return result