from dotenv import load_dotenv
# print(f"环境变量设置{load_dotenv()}")

# import nltk
# # 下载常用的 tokenizer 模块
# nltk.download('punkt')
# #下载词性标注器
# nltk.download('averaged_perceptron_tagger')
# # 下载 WordNet 词典
# nltk.download('wordnet')
# #下载停用词表
# nltk.download('stopwords')

file_path = ("90-文档-Data/山西文旅/云冈石窟-en.pdf")
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,  # PDF文件路径
    strategy="hi_res",    # 使用高分辨率策略进行文档处理
    # partition_via_api=True,  # 通过API进行文档分块
    # coordinates=True,     # 提取文本坐标信息
)
docs = []

# lazy_load() 是一种延迟加载方法
# 它不会一次性将所有文档加载到内存中，而是在需要时才逐个加载文档
# 这对于处理大型PDF文件时可以节省内存使用
for doc in loader.lazy_load():
    docs.append(doc)

print(docs)
