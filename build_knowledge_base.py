"""
一键构建软工竞赛知识库
运行方式：python build_knowledge_base.py
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查API Key
if not os.getenv("DASHSCOPE_API_KEY"):
    print("❌ 错误：请先在 .env 文件中设置 DASHSCOPE_API_KEY")
    sys.exit(1)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_dashscope import DashScopeEmbeddings

def build_knowledge_base():
    print("🚀 开始构建软工竞赛知识库...")
    
    # 初始化嵌入模型
    print("📡 正在连接通义千问嵌入模型...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 资料文件夹
    kb_dir = "./knowledge_base"
    db_dir = "./chroma_db"
    
    if not os.path.exists(kb_dir):
        print(f"❌ 错误：找不到 {kb_dir} 文件夹")
        sys.exit(1)
    
    # 收集所有文档
    documents = []
    
    print(f"📁 正在扫描 {kb_dir} 文件夹...")
    
    for root, dirs, files in os.walk(kb_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in ['.pdf', '.txt', '.md']:
                try:
                    print(f"  📄 正在处理: {file}")
                    
                    if ext == '.pdf':
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    else:
                        # TXT/MD 文件，尝试多种编码
                        content = None
                        for encoding in ['utf-8', 'gbk', 'gb2312']:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                break
                            except:
                                continue
                        
                        if content is None:
                            raise Exception("无法识别文件编码")
                        
                        from langchain_core.documents import Document
                        docs = [Document(page_content=content, metadata={"source": file})]
                    
                    # 添加来源信息
                    category = os.path.basename(root)
                    for doc in docs:
                        doc.metadata.update({
                            "source": file,
                            "category": category,
                            "file_path": file_path
                        })
                    
                    documents.extend(docs)
                    print(f"     ✅ 成功加载 {len(docs)} 页/段")
                except Exception as e:
                    print(f"     ❌ 失败: {e}")
            else:
                print(f"  ⏭️  跳过: {file} (不支持的格式)")
    
    if not documents:
        print("❌ 错误：没有找到可处理的文档")
        sys.exit(1)
    
    print(f"\n📊 共加载 {len(documents)} 个文档片段")
    
    # 分割文本
    print("✂️  正在分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"📑 分割为 {len(texts)} 个检索片段")
    
    # 构建向量库
    print(f"🧠 正在生成向量嵌入（调用通义千问API）...")
    
    # 删除旧库
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
        print("🗑️  已清除旧向量库")
    
    # 创建新库
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    # 持久化保存
    vectordb.persist()
    
    print(f"\n✅ 知识库构建完成！")
    print(f"📂 向量库位置: {os.path.abspath(db_dir)}")
    print(f"📚 共索引 {len(texts)} 个片段")
    print(f"\n💡 现在可以运行: streamlit run app.py")
    
    # 显示统计
    print(f"\n📈 知识库统计:")
    categories = {}
    for doc in texts:
        cat = doc.metadata.get("category", "未分类")
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count} 个片段")

if __name__ == "__main__":
    build_knowledge_base()