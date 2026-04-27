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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 去掉 langchain_dashscope，改用原生 dashscope
from dashscope import Generation
import dashscope

API_KEY = st.secrets.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    st.error("请在 Streamlit Secrets 配置 DASHSCOPE_API_KEY")
    st.stop()

dashscope.api_key = API_KEY

# ========== 页面设置 ==========
st.set_page_config(
    page_title="软工智能助手",
   # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("软件工程智能问答系统")

# ========== 初始化模型（缓存）==========
@st.cache_resource
def get_models():
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    llm = ChatDashScope(model_name="qwen-turbo", temperature=0.7)
    return embeddings, llm

embeddings, llm = get_models()

# ========== 侧边栏 ==========
with st.sidebar:
    st.header(" 控制面板")
    
    mode = st.radio(
        "选择问答模式",
        ["💬 通用问答", "📄 PDF上传问答", "🏆 软工竞赛专区"],
        index=0
    )
    
    st.divider()

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
        st.session_state.general_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 思考中..."):
                try:
                    messages = []
                    recent_history = st.session_state.general_history[-20:]
                    
                    for msg in recent_history:
                        role = "user" if msg["role"] == "user" else "assistant"
                        messages.append({"role": role, "content": msg["content"]})
                    
                    response = Generation.call(
                        model="qwen-turbo",
                        messages=messages,
                        result_format="message"
                    )
                    answer = response.output.choices[0].message.content
                    st.markdown(answer)
                    
                    st.session_state.general_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"调用失败: {e}")
        
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
        st.session_state.pdf_chat_history = []
        
        import glob
        for old_temp in glob.glob("temp_*.pdf"):
            try:
                os.remove(old_temp)
            except:
                pass
        
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
            
            for msg in st.session_state.pdf_chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])
            
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
            
            for temp_path in glob.glob("temp_*.pdf"):
                try:
                    os.remove(temp_path)
                except:
                    pass

# ========== 软工竞赛专区 ==========
elif mode == "🏆 软工竞赛专区":
    st.header("🏆 软件工程竞赛专区")
    
import glob
from langchain_community.document_loaders import PyPDFLoader

@st.cache_resource
def build_competition_kb():
    all_docs = []
    # 自动加载 knowledge_base 里的所有 PDF
    for pdf_path in glob.glob("knowledge_base/**/*.pdf", recursive=True):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # 给文档加分类标签
            for doc in docs:
                category = os.path.basename(os.path.dirname(pdf_path))
                doc.metadata["category"] = category
                doc.metadata["source"] = os.path.basename(pdf_path)
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"加载 {pdf_path} 失败: {e}")
    
    if not all_docs:
        return None
    
    # 分块并构建向量库（内存模式，不持久化）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(all_docs)
    return Chroma.from_documents(documents=texts, embedding=embeddings)

vectordb = build_competition_kb()

if not vectordb:
    st.error("❌ 未找到竞赛知识库文件，请检查 knowledge_base 文件夹")
    st.stop()
        
        try:
            vectordb = load_competition_kb()
            
            with st.expander("📁 知识库概况", expanded=True):
                all_docs = vectordb.get()
                categories = {}
                for meta in all_docs['metadatas']:
                    cat = meta.get('category', '未分类')
                    categories[cat] = categories.get(cat, 0) + 1
                
                cols = st.columns(min(len(categories), 3))
                for i, (cat, count) in enumerate(sorted(categories.items())):
                    with cols[i % 3]:
                        st.metric(f"📁 {cat}", f"{count} 片段")
            
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
            
            if "se_history" not in st.session_state:
                st.session_state.se_history = []
            
            for msg in st.session_state.se_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏆"):
                    st.markdown(msg["content"])
            
            prompt = st.chat_input("提问软工竞赛相关问题...", key="se_chat_input")            
            if prompt:
                if "se_input" in st.session_state:
                    del st.session_state["se_input"]
                
                st.session_state.se_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant", avatar="🏆"):
                    with st.spinner("🔍 检索竞赛资料..."):
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
