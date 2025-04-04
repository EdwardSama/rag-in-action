from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import DeepSeek
import os
from glob import glob

class RAGSystem:
    def __init__(
        self,
        deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
        vector_store_path: str = "./vector_store",
        docs_path: str = "./90-文档-Data",
        embedding_model_name: str = "shibing624/text2vec-base-chinese",
        device: str = "cpu",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        llm_model_name: str = "deepseek-chat",
        llm_temperature: float = 0.7
    ):
        """初始化RAG系统
        Args:
            deepseek_api_key: DeepSeek API密钥，默认从环境变量获取
            vector_store_path: 向量存储路径，默认'./vector_store'
            docs_path: 文档存储路径，默认'./90-文档-Data'
            embedding_model_name: 嵌入模型名称，默认'shibing624/text2vec-base-chinese'
            device: 计算设备，默认'cpu'
            chunk_size: 文本分块大小，默认1000
            chunk_overlap: 文本分块重叠大小，默认200
            llm_model_name: LLM模型名称，默认'deepseek-chat'
            llm_temperature: LLM温度参数，默认0.7
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': device}
            )
        except Exception as e:
            raise RuntimeError(f"加载嵌入模型失败: {str(e)}") from e
        
        self.DEEPSEEK_API_KEY = deepseek_api_key
        self.VECTOR_STORE_PATH = vector_store_path
        self.DOCS_PATH = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature

    def _load_documents(self):
        """加载指定路径下的所有txt文档
        Returns:
            包含所有文档内容的Document对象列表
        """
        documents = []
        try:
            for file_path in glob(os.path.join(self.DOCS_PATH, "*.txt")):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except FileNotFoundError:
                    raise FileNotFoundError(f"文件 {file_path} 不存在")
                except PermissionError:
                    raise PermissionError(f"无权访问文件 {file_path}")
                except Exception as e:
                    print(f"加载文件 {file_path} 失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"加载文档时发生错误: {str(e)}") from e
        return documents

    def _split_text(self, documents):
        """将文档按指定大小进行分块
        Args:
            documents: 待分块的文档列表
        Returns:
            分块后的文档列表
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            raise ValueError(f"文本分块失败: {str(e)}") from e

    def create_vector_store(self):
        """创建并保存向量存储
        Returns:
            创建好的FAISS向量存储对象
        """
        try:
            documents = self._load_documents()
            chunks = self._split_text(documents)
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(self.VECTOR_STORE_PATH)
            return vector_store
        except FileNotFoundError:
            raise FileNotFoundError(f"向量存储路径 {self.VECTOR_STORE_PATH} 不存在")
        except PermissionError:
            raise PermissionError(f"无权写入向量存储路径 {self.VECTOR_STORE_PATH}")
        except Exception as e:
            raise RuntimeError(f"创建向量存储失败: {str(e)}") from e

    def load_vector_store(self):
        """加载或创建向量存储
        Returns:
            加载或新建的FAISS向量存储对象
        """
        try:
            if os.path.exists(self.VECTOR_STORE_PATH):
                return FAISS.load_local(self.VECTOR_STORE_PATH, self.embeddings)
            else:
                return self.create_vector_store()
        except FileNotFoundError:
            raise FileNotFoundError(f"向量存储路径 {self.VECTOR_STORE_PATH} 不存在")
        except Exception as e:
            raise RuntimeError(f"加载向量存储失败: {str(e)}") from e

    def setup_qa_chain(self):
        """设置问答链
        Returns:
            配置好的RetrievalQA问答链对象
        """
        try:
            llm = DeepSeek(
                api_key=self.DEEPSEEK_API_KEY,
                model_name=self.llm_model_name,
                temperature=self.llm_temperature
            )
        except Exception as e:
            raise ConnectionError(f"初始化LLM失败: {str(e)}. 请检查API密钥和网络连接") from e
        
        try:
            vector_store = self.load_vector_store()
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3})
            )
        except Exception as e:
            raise RuntimeError(f"构建问答链失败: {str(e)}") from e

    def run(self):
        """启动问答系统交互界面
        持续接收用户输入，直到输入'q'退出
        """
        try:
            qa_chain = self.setup_qa_chain()
        except Exception as e:
            print(f"初始化问答系统失败: {str(e)}")
            return

        while True:
            try:
                question = input("\n请输入您的问题（输入'q'退出）: ")
                if question.lower() == 'q':
                    break
                answer = qa_chain.run(question)
                print("\n答案:", answer)
            except KeyboardInterrupt:
                print("\n强制退出问答系统")
                break
            except Exception as e:
                print(f"处理问题时发生错误: {str(e)}")

if __name__ == "__main__":
    try:
        rag_system = RAGSystem()
        rag_system.run()
    except Exception as e:
        print(f"系统启动失败: {str(e)}")