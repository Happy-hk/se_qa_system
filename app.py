"""
软件工程智能问答系统
三合一：通用问答 | PDF上传 | 软工竞赛专区
"""

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi

from dashscope import Generation, TextEmbedding
import dashscope

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not API_KEY:
    st.error("❌ 请在 .env 文件或 Streamlit Secrets 中设置 API Key")
    st.stop()

dashscope.api_key = API_KEY

# ========== 页面设置 ==========
st.set_page_config(
    page_title="软工智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 软件工程智能问答系统")

# ========== 初始化模型（缓存）==========
@st.cache_resource
def get_models():
    # 改用 dashscope 原生 Embeddings，不依赖任何额外库
    from langchain_core.embeddings import Embeddings
    import numpy as np

    class DashScopeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            responses = []
            for text in texts:
                rsp = TextEmbedding.call(model="text-embedding-v2", input=text)
                responses.append(rsp.output.embeddings[0].embedding)
            return responses
        
        def embed_query(self, text):
            return self.embed_documents([text])[0]

    embeddings = DashScopeEmbeddings()
    llm = Tongyi(
        model_name="qwen-turbo",
        temperature=0.7,
        dashscope_api_key=API_KEY
    )
    return embeddings, llm

embeddings, llm = get_models()

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    mode = st.radio(
        "选择问答模式",
        ["💬 通用问答", "📄 PDF上传问答", "🏆 软工竞赛专区"],
        index=0
    )
    
    st.divider()
    
    st.subheader("📊 系统状态")
    st.success("🟢 通义千问已连接")
    
    if os.path.exists("./chroma_db"):
        st.success("🟢 竞赛知识库已加载")
    else:
        st.warning("🟡 竞赛知识库未构建")
        st.info("运行: python build_knowledge_base.py")
    
    st.divider()
    st.caption("Powered by 通义千问 + LangChain + Streamlit")

# ========== 通用问答模式 ==========
if mode == "💬 通用问答":
    st.header("💬 通用问答模式")
    
    if "general_history" not in st.session_state:
        st.session_state.general_history = []
    
    # 显示历史对话
    for msg in st.session_state.general_history:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("请输入你的问题...", key="general_input"):
        # 添加用户消息到历史
        st.session_state.general_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 思考中..."):
                try:
                    # 构建带历史的消息（只保留最近10轮 = 20条消息）
                    messages = []
                    recent_history = st.session_state.general_history[-20:]  # 最近10轮
                    
                    for msg in recent_history:
                        role = "user" if msg["role"] == "user" else "assistant"
                        messages.append({"role": role, "content": msg["content"]})
                    
                    # 调用通义千问（带上下文）
                    response = Generation.call(
                        model="qwen-turbo",
                        messages=messages,
                        result_format="message"
                    )
                    answer = response.output.choices[0].message.content
                    st.markdown(answer)
                    
                    # 添加AI回复到历史
                    st.session_state.general_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"调用失败: {e}")
        
        # 限制历史长度，防止太长（保留最近12轮，给用户一点缓冲）
        if len(st.session_state.general_history) > 24:
            st.session_state.general_history = st.session_state.general_history[-20:]
elif mode == "📄 PDF上传问答":
    st.header("📄 PDF上传问答模式")
    
    def clear_pdf_state():
        for key in list(st.session_state.keys()):
            if key.startswith("pdf_"):
                del st.session_state[key]
    
    uploaded_files = st.file_uploader(
        "拖拽或点击上传PDF（支持多个）", 
        type=["pdf"],
        accept_multiple_files=True,
        on_change=clear_pdf_state
    )
    
    if uploaded_files:
        # 每次上传都重新处理（不缓存）
        st.session_state.pdf_chat_history = []
        
        # 清除旧的临时文件
        import glob
        for old_temp in glob.glob("temp_*.pdf"):
            try:
                os.remove(old_temp)
            except:
                pass
        
        # 保存所有新上传的文件
        all_documents = []
        file_names = []
        
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                all_documents.extend(documents)
                file_names.append(uploaded_file.name)
            except Exception as e:
                st.warning(f"⚠️ {uploaded_file.name} 解析失败: {e}")
        
        if all_documents:
            st.success(f"✅ 成功加载 {len(file_names)} 个文件，共 {len(all_documents)} 页")
            st.info(f"📄 已加载: {', '.join(file_names)}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "，", " ", ""]
            )
            texts = text_splitter.split_documents(all_documents)
            
            # 内存模式向量库
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embeddings
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
                memory=memory
            )
            
            st.info(f"📑 共切分为 {len(texts)} 个片段，可以开始提问")
            
            # 显示历史
            for msg in st.session_state.pdf_chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])
            
            # 提问
            if prompt := st.chat_input("基于所有PDF内容提问...", key="pdf_input"):
                st.session_state.pdf_chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🔍 检索所有文档中..."):
                        result = qa_chain.invoke({"question": prompt})
                        answer = result["answer"]
                        st.markdown(answer)
                        
                        with st.expander("📚 查看参考来源"):
                            docs = vectordb.similarity_search(prompt, k=3)
                            for i, doc in enumerate(docs, 1):
                                source = doc.metadata.get('source', '未知文件')
                                page = doc.metadata.get('page', '?')
                                st.markdown(f"**片段{i}**（来自: {source} 第{page}页）")
                                st.text(doc.page_content[:300] + "...")
                                st.divider()
                
                st.session_state.pdf_chat_history.append({"role": "assistant", "content": answer})
            
            # 清理临时文件
            for temp_path in glob.glob("temp_*.pdf"):
                try:
                    os.remove(temp_path)
                except:
                    pass
# ========== 软工竞赛专区 ==========
elif mode == "🏆 软工竞赛专区":
    st.header("🏆 软件工程竞赛专区")
    
    db_dir = "./chroma_db"
    
    if not os.path.exists(db_dir):
        st.error("❌ 竞赛知识库尚未构建")
        st.markdown("""
        ### 构建步骤：
        1. 创建文件夹 `knowledge_base/`
        2. 放入软工竞赛资料（PDF/TXT）
        3. 运行：`python build_knowledge_base.py`
        """)
        
        with st.expander("📁 查看推荐资料结构"):
            st.code("""
knowledge_base/
├── 蓝桥杯/
│   ├── 历年真题解析.pdf
│   └── 常用算法模板.txt
├── ACM/
│   ├── 数据结构精讲.pdf
│   └── 动态规划专题.pdf
└── 软件工程/
    ├── 设计模式详解.pdf
    └── 系统架构案例.pdf
            """)
    else:
        @st.cache_resource
        def load_competition_kb():
            return Chroma(
                persist_directory=db_dir,
                embedding_function=embeddings
            )
        
        try:
            vectordb = load_competition_kb()
            
            with st.expander("📊 知识库概况", expanded=True):
                all_docs = vectordb.get()
                categories = {}
                for meta in all_docs['metadatas']:
                    cat = meta.get('category', '未分类')
                    categories[cat] = categories.get(cat, 0) + 1
                
                cols = st.columns(min(len(categories), 3))
                for i, (cat, count) in enumerate(sorted(categories.items())):
                    with cols[i % 3]:
                        st.metric(f"📁 {cat}", f"{count} 片段")
            
            # 快捷问题
            st.subheader("🚀 快捷提问")
            quick_questions = [
                "蓝桥杯常考哪些算法？",
                "ACM动态规划入门技巧",
                "单例模式的应用场景",
                "软件架构设计原则",
                "如何准备软件工程竞赛？"
            ]
            
            cols = st.columns(3)
            for i, q in enumerate(quick_questions):
                with cols[i % 3]:
                    if st.button(q, key=f"quick_{i}"):
                        st.session_state["se_chat_input"] = q
                        st.rerun()
            
            # 初始化历史
            if "se_history" not in st.session_state:
                st.session_state.se_history = []
            
            # 显示历史
            for msg in st.session_state.se_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏆"):
                    st.markdown(msg["content"])
            
            # 输入框
            prompt = st.chat_input("提问软工竞赛相关问题...", key="se_chat_input")            
            if prompt:
                if "se_input" in st.session_state:
                    del st.session_state["se_input"]
                
                st.session_state.se_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant", avatar="🏆"):
                    st.spinner("🔍 检索竞赛资料...")
                    try:
                        docs = vectordb.similarity_search(prompt, k=4)
                        context = "\n\n".join([
                            f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}" 
                            for doc in docs
                        ])
                        
                        system_prompt = f"""你是软件工程竞赛专家助手。请基于以下竞赛资料回答问题：
1. 回答准确、专业，适合竞赛备赛
2. 如涉及代码，给出完整可运行示例
3. 如涉及算法，说明时间复杂度和适用场景
4. 必须注明参考来源

=== 参考资料 ===
{context}

=== 用户问题 ===
{prompt}

请详细回答："""
                        
                        response = Generation.call(
                            model="qwen-turbo",
                            messages=[{"role": "user", "content": system_prompt}],
                            result_format="message"
                        )
                        answer = response.output.choices[0].message.content
                        st.markdown(answer)
                        
                        with st.expander("📖 参考来源"):
                            sources = set()
                            for doc in docs:
                                src = doc.metadata.get('source', '未知')
                                cat = doc.metadata.get('category', '未分类')
                                sources.add(f"📁 {cat} / {src}")
                            for s in sorted(sources):
                                st.markdown(f"- {s}")
                        
                        st.session_state.se_history.append({
                            "role": "assistant", 
                            "content": answer
                        })
                        
                    except Exception as e:
                        st.error(f"检索失败: {e}")
                            
        except Exception as e:
            st.error(f"加载知识库失败: {e}")

st.divider()
st.caption("© 2024 软件工程智能助手 | 使用通义千问大模型 | 仅供学习交流")
