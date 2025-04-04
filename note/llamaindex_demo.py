import socks # 安装pysocks
import socket
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# 加载环境变量
# load_dotenv('./note/.env')
print(f"环境变量设置{load_dotenv()}")

# 保存原始socket对象
# original_socket = socket.socket
# # 设置代理时替换socket
# socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
# socket.socket = socks.socksocket  # ‌:ml-citation{ref="2" data="citationList"}

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="./model_caches/bge-small-zh")

# 取消代理时恢复原始socket
# socket.socket = original_socket

# 创建 Deepseek LLM
llm = DeepSeek(
    model="deepseek-reasoner",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 

# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 第四行代码：创建问答引擎
query_engine = index.as_query_engine(
    llm=llm
)

# 第五行代码: 开始问答
print(query_engine.query("请告诉我, 黑神话悟空中有哪些战斗工具?用中文回答"))
