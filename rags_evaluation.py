"""
RAGAS 评估脚本（适配 ragas 0.2.x + 法律RAG）
"""
import os
import sys
import time
import logging

# Ensure UTF-8 output for Windows
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# ============================================================
# 1. 环境配置（优先加载）
# ============================================================
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["ANONYMIZED_TELEMETRY"]      = "False"

from dotenv import load_dotenv
load_dotenv()

# OpenAI 配置（兼容国产大模型的OpenAI接口）
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "dummy-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

# ============================================================
# 2. 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 3. 导入 ragas（0.2.x 版本）
# ============================================================
import openai
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# ============================================================
# 4. 初始化 Agent（加载向量库）
# ============================================================
from langchain_core.messages import HumanMessage
from src import agents, graph

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir      = os.path.join(current_dir, "chroma_db")

logger.info(f"正在加载 Researcher Agent，向量库路径：{db_dir}")
researcher = agents.Researcher_Agent(persist_directory=db_dir)

# ============================================================
# 5. 测试数据
# ============================================================

test_data = [
    {
        "question": "婚前我付首付买的房子，婚后共同还贷，离婚时房子该怎么分？",
        "ground_truth": (
            "根据《民法典》婚姻家庭编司法解释第七十八条，该房产产权归你（首付方）所有，但婚后共同还贷支付的款项及其相对应财产增值部分，需作为夫妻共同财产予以分割。"
            "具体分割方式：1. 你需补偿对方「共同还贷本金的一半 + 对应增值部分」；2. 若房产证婚后添加了对方名字，视为赠与，房产转为夫妻共同财产，一般按均等原则分割（需结合贡献度调整）；3. 未还清的贷款属于你的个人债务。"
        ),
    },
    # 3. 房产纠纷 - 二手房买卖违约
    {
        "question": "我买二手房签了合同并付了定金，卖家突然反悔不想卖了，我能要求他继续履行合同吗？",
        "ground_truth": (
            "可以。根据《民法典》第五百七十七条、第五百八十六条，定金合同自实际交付定金时成立，卖家违约的，你有两种主张方式：1. 要求卖家继续履行合同（办理过户手续）；2. 放弃继续履行，要求卖家双倍返还定金，或按合同约定主张违约金（择一主张，若违约金低于实际损失可要求调高）。"
            "若卖家已将房产过户给第三方且第三方是善意取得，你无法要求继续履行，只能主张赔偿损失（包括房屋差价损失）。"
        ),
    },
    # 4. 合同纠纷 - 定金 vs 订金
    {
        "question": "签合同时写的是“订金”，现在商家违约，我能要求双倍返还吗？",
        "ground_truth": (
            "不能。“订金”与“定金”法律性质不同：1. 定金（《民法典》第五百八十六条）具有担保性质，收受方违约需双倍返还，支付方违约无权要求返还；2. 订金仅为预付款，不适用定金罚则，商家违约时，你只能要求返还订金本金，或按合同约定主张违约金/赔偿损失。"
            "若合同中虽写“订金”，但明确约定了“违约双倍返还”，则该条款视为定金约定，可主张双倍返还。"
        ),
    },
    # 5. 知识产权 - 图片侵权
    {
        "question": "我在公众号用了网上找的图片，被作者起诉侵权，该怎么处理？",
        "ground_truth": (
            "该行为涉嫌侵犯图片的著作权（信息网络传播权）。处理方式：1. 立即删除侵权图片，停止侵权行为；2. 与权利人协商和解，赔偿金额参考图片授权使用费、侵权传播范围、获利情况等（通常几百至数千元）；3. 若权利人主张金额过高，可举证证明无主观过错（如已尽到合理注意义务），争取降低赔偿或免责。"
            "注意：“网上找的图片”≠“免费使用”，商用场景需获得著作权人授权，即使注明出处也不免除侵权责任。"
        ),
    }
]
# ============================================================
# 6. 运行 Agent，收集评估数据
# ============================================================
questions:          list[str]       = []
answers:            list[str]       = []
retrieved_contexts: list[list[str]] = []
ground_truths:      list[str]       = []

MAX_RETRIES = 3
INIT_DELAY  = 2   # 初始重试间隔（秒）

for idx, item in enumerate(test_data, 1):
    q  = item["question"]
    gt = item["ground_truth"]
    logger.info(f"▶ [{idx}/{len(test_data)}] 问题：{q[:40]}…")

    # ── 6a. Researcher 检索上下文 ────────────────────────────
    try:
        docs          = researcher.invoke(q)
        if docs:
            context_texts = [f"法律名称: 《{doc.metadata.get('chapter', '未知')}》\n条款: {doc.metadata.get('article_id', '未知')}\n内容: {doc.page_content}" for doc in docs]
        else:
            context_texts = []
        logger.info(f"   检索到 {len(context_texts)} 条上下文")
        if not context_texts:
            logger.warning("   ⚠️  未检索到任何文档")
    except Exception as exc:
        logger.error(f"   ❌ 检索阶段出错：{exc}", exc_info=True)
        context_texts = []

    # ── 6b. Graph 生成最终回答（带指数退避重试）───────────────
    answer       = "No answer generated"
    retry_delay  = INIT_DELAY
    result_state = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            config       = {"configurable": {"thread_id": f"eval-{idx}"}}
            inputs       = {"messages": [HumanMessage(content=q)]}
            result_state = graph.app.invoke(inputs, config)
            logger.info(f"   ✅ Graph 调用成功（第 {attempt} 次）")
            break
        except openai.RateLimitError as exc:
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"   ⚠️  API 限流，{retry_delay:.1f}s 后重试 [{attempt}/{MAX_RETRIES}]：{exc}"
                )
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                logger.error(f"   ❌ 已达最大重试次数，跳过：{exc}")
        except Exception as exc:
            logger.error(f"   ❌ Graph 调用出错：{exc}", exc_info=True)
            break

    # ── 6c. 提取回答文本 ─────────────────────────────────────
    if result_state and result_state.get("messages"):
        last = result_state["messages"][-1]
        text = getattr(last, "content", None) or str(last)
        if text.strip():
            answer = text.strip()

    logger.info(f"   📝 回答摘要：{answer[:60]}…")

    # ── 6d. 收集数据 ─────────────────────────────────────────
    questions.append(q)
    answers.append(answer)
    retrieved_contexts.append(context_texts)
    ground_truths.append(gt)

# ============================================================
# 7. 构建 RAGAS Dataset 并评估
#    ragas 0.2.x 标准列名：
#      question / answer / contexts / ground_truth
# ============================================================
logger.info("🔍 构建 RAGAS 数据集并开始评估…")

ragas_dataset = Dataset.from_dict(
    {
        "question":     questions,
        "answer":       answers,
        "contexts":     retrieved_contexts,   # List[List[str]]
        "ground_truth": ground_truths,
    }
)

# RAGAS 0.2.x 使用预定义的 metrics 对象
metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

# 配置 RAGAS 使用 ModelScope 的模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ragas_llm = ChatOpenAI(
    model="Qwen/Qwen3.5-35B-A3B",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.1,
    model_kwargs={"extra_body": {"enable_thinking": False}},
    # 添加请求超时和重试配置
    request_timeout=120,
    max_retries=5,
)

# 使用 ModelScope embedding 避免 Ollama 服务问题
ragas_embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

results = evaluate(
    dataset          = ragas_dataset,
    metrics          = metrics,
    llm              = ragas_llm,
    embeddings       = ragas_embeddings,
    raise_exceptions = False,   # 单条评估出错不整体崩溃
    show_progress    = True,
)

# ============================================================
# 8. 展示结果
# ============================================================
df = results.to_pandas()

SCORE_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
SHOW_COLS  = [c for c in SCORE_COLS if c in df.columns]

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 130)
pd.set_option("display.float_format", "{:.4f}".format)

print("\n" + "=" * 80)
print("📊  RAGAS 法律多智能体评估结果")
print("=" * 80)
print("问题：", questions[0])
print("回答：", answers[0])
print("\n评估指标：")
print(df[SHOW_COLS].to_string(index=False))
print()

# 打印各指标平均分 + ASCII 进度条
valid_score_cols = [c for c in SCORE_COLS if c in df.columns]
if valid_score_cols:
    print("📈  各指标平均得分（满分 1.0）：")
    for col in valid_score_cols:
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            print(f"   {col:<22}: N/A  (评估失败)")
        else:
            filled   = int(mean_val * 20)
            bar      = "█" * filled + "░" * (20 - filled)
            print(f"   {col:<22}: {mean_val:.4f}  |{bar}|")
    print()

# ============================================================
# 9. 保存到 CSV
# ============================================================
output_path = os.path.join(current_dir, "ragas_results.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
logger.info(f"✅ 完整评估结果已保存至：{output_path}")
