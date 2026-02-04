import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import pickle
import os

# ページ設定
st.set_page_config(
    page_title="相続ナレッジ検索",
    page_icon="🔍",
    layout="wide"
)

# Gemini API設定（Streamlit Secretsから読み込み）
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"

@st.cache_data
def load_data():
    """データとembeddingを読み込み（事前作成済みキャッシュから）"""
    df = pd.read_excel("RAGデータ/columns_articles_with_summary.xlsx")

    # タイトル+本文のEmbedding（RAG用）
    with open("embeddings_cache.pkl", "rb") as f:
        embeddings_full = pickle.load(f)
    df["embedding"] = embeddings_full

    # タイトル+要約のEmbedding（比較用）
    with open("embeddings_summary_cache.pkl", "rb") as f:
        embeddings_summary = pickle.load(f)
    df["embedding_summary"] = embeddings_summary

    # タイトル+ペルソナ要約のEmbedding（新規）
    with open("embeddings_cache_new_rin.pkl", "rb") as f:
        embeddings_new_rin = pickle.load(f)
    df["embedding_new_rin"] = embeddings_new_rin

    return df

def get_query_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return result['embedding']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_titles(df, query, top_k=5, embedding_col='embedding'):
    query_embedding = get_query_embedding(query)

    similarities = []
    for idx, row in df.iterrows():
        sim = cosine_similarity(query_embedding, row[embedding_col])
        
        similarities.append({
            'title': row['title'],
            'body': row['body'],
            'summary': row['summary'] if pd.notna(row['summary']) else '',
            'generic_summary': row['汎用要約'] if pd.notna(row['汎用要約']) else '',
            'similarity': sim
        })

    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

def generate_rag_response(df, user_query, top_k=3, embedding_col='embedding'):
    model = genai.GenerativeModel("gemini-2.5-flash")
    search_results = search_similar_titles(df, user_query, top_k=top_k, embedding_col=embedding_col)
    
    context = "\n\n".join([
        f"【記事{i+1}】タイトル: {r['title']}\n内容: {r['body']}"
        for i, r in enumerate(search_results)
    ])
    
    prompt = f"""以下の参考記事を元に、ユーザーの質問に回答してください。

【参考記事】
{context}

【ユーザーの質問】
{user_query}

【回答】"""
    
    response = model.generate_content(prompt)
    return response.text, search_results

# メイン画面
st.title("🔍 相続ナレッジ検索")
st.caption("Gemini 2.5 Flash + RAG デモ版")

# サイドバーにキャッシュクリアボタン
with st.sidebar:
    if st.button("キャッシュクリア"):
        st.cache_data.clear()
        st.rerun()

# データ読み込み
with st.spinner("データを読み込み中..."):
    df = load_data()

# Embedding次元数を表示
embed_dim = len(df["embedding"].iloc[0]) if len(df) > 0 else 0
st.success(f"✅ {len(df)}件の記事を読み込みました（Embedding: {embed_dim}次元）")

# 検索UI
st.divider()
query = st.text_input("🔍 質問を入力してください", placeholder="例: 相続税について教えて")

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    top_k = st.selectbox("検索件数", [3, 5, 10], index=0)
with col2:
    embedding_type = st.radio(
        "Embedding種類",
        ["タイトル+本文", "タイトル+要約", "タイトル+ペルソナ要約"],
        horizontal=True
    )
with col3:
    search_mode = st.radio("検索モード", ["タイトル検索のみ", "AI回答生成"], horizontal=True)

# 選択に応じてembeddingカラムを決定
if embedding_type == "タイトル+本文":
    embedding_col = "embedding"
elif embedding_type == "タイトル+要約":
    embedding_col = "embedding_summary"
else:  # タイトル+ペルソナ要約
    embedding_col = "embedding_new_rin"

if st.button("検索", type="primary") and query:
    with st.spinner("検索中..."):
        if search_mode == "タイトル検索のみ":
            results = search_similar_titles(df, query, top_k=top_k, embedding_col=embedding_col)

            st.subheader("検索結果")
            st.caption(f"使用Embedding: {embedding_type}")
            for i, r in enumerate(results, 1):
                with st.expander(f"{i}. {r['title']} (類似度: {r['similarity']:.3f})"):
                    if embedding_type == "タイトル+要約":
                        st.write(r['summary'])
                    elif embedding_type == "タイトル+ペルソナ要約":
                        st.write(r['generic_summary'])
                    else:
                        st.write(r['body'])
        else:
            answer, sources = generate_rag_response(df, query, top_k=top_k, embedding_col=embedding_col)
            
            st.subheader("AI回答")
            st.caption(f"使用Embedding: {embedding_type}")
            st.write(answer)

            st.subheader("参照元記事")
            for i, r in enumerate(sources, 1):
                with st.expander(f"{i}. {r['title']} (類似度: {r['similarity']:.3f})"):
                    if embedding_type == "タイトル+要約":
                        st.write(r['summary'])
                    elif embedding_type == "タイトル+ペルソナ要約":
                        st.write(r['generic_summary'])
                    else:
                        st.write(r['body'])