import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import pickle
import random
import re

# ページ設定
st.set_page_config(
    page_title="拡張コラムチャット",
    page_icon="💬",
    layout="wide"
)

# Gemini API設定（Streamlit Secretsから読み込み）
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-001"

@st.cache_data
def load_data():
    """データとembeddingを読み込み"""
    df = pd.read_excel("RAGデータ/columns_articles_with_summary.xlsx")

    # タイトル+本文のEmbedding（RAG検索用）
    with open("embeddings_cache.pkl", "rb") as f:
        embeddings = pickle.load(f)
    df["embedding"] = embeddings

    return df

def get_random_articles(df, n=10):
    """ランダムにn件の記事を選択"""
    return df.sample(n=min(n, len(df))).reset_index(drop=True)

def get_query_embedding(text):
    """クエリのEmbeddingを取得"""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return result['embedding']

def cosine_similarity(a, b):
    """コサイン類似度を計算"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_articles(df, query, top_k=3, exclude_titles=None):
    """類似記事を検索"""
    query_embedding = get_query_embedding(query)

    similarities = []
    for idx, row in df.iterrows():
        # 除外するタイトルはスキップ
        if exclude_titles and row['title'] in exclude_titles:
            continue

        sim = cosine_similarity(query_embedding, row['embedding'])
        similarities.append({
            'title': row['title'],
            'body': row['body'],
            'similarity': sim
        })

    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

def generate_response_step1(selected_article, referenced_articles, user_message, chat_history):
    """ステップ1: 選択記事 + 参照済み記事で回答を試みる"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    # チャット履歴を構築
    history_text = ""
    if chat_history:
        history_text = "\n\n【これまでの会話履歴】\n"
        for msg in chat_history[-6:]:  # 直近6件のみ
            history_text += f"{msg['role']}: {msg['content']}\n"

    # 追加参照記事のコンテキスト
    ref_context = ""
    if referenced_articles:
        ref_context = "\n\n【追加参照記事】\n"
        for i, article in enumerate(referenced_articles, 1):
            ref_context += f"記事{i}: {article['title']}\n{article['body']}\n\n"

    prompt = f"""あなたは相続に関する専門家です。
以下の記事の内容に基づいて、ユーザーの質問に答えてください。

【選択中の記事】
タイトル: {selected_article['title']}
本文: {selected_article['body']}
{ref_context}{history_text}

【ユーザーの質問】
{user_message}

【回答ルール】
- 提供された記事から回答できる場合は、丁寧に回答する
- 提供されたどの記事にも情報がない場合は「[[SEARCH_NEEDED]]」とだけ回答する
- 簡潔かつ分かりやすく説明する（300文字以内）
- 回答の最後に、参照した記事のタイトルを「[[REF:記事タイトル]]」の形式で必ず記載する（複数ある場合は複数記載）
"""

    response = model.generate_content(prompt)
    return response.text

def generate_response_step2(selected_article, referenced_articles, search_results, user_message, chat_history):
    """ステップ2: RAG検索結果を含めて回答を試みる"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    # チャット履歴を構築
    history_text = ""
    if chat_history:
        history_text = "\n\n【これまでの会話履歴】\n"
        for msg in chat_history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"

    # 参照記事のコンテキスト（既存 + 新規検索結果）
    all_refs = (referenced_articles or []) + search_results
    ref_context = "\n\n【追加参照記事】\n"
    for i, article in enumerate(all_refs, 1):
        ref_context += f"記事{i}: {article['title']}\n{article['body']}\n\n"

    prompt = f"""あなたは相続に関する専門家です。
以下の記事の内容に基づいて、ユーザーの質問に答えてください。

【選択中の記事】
タイトル: {selected_article['title']}
本文: {selected_article['body']}
{ref_context}{history_text}

【ユーザーの質問】
{user_message}

【回答ルール】
- 提供された記事から回答できる場合は、丁寧に回答する
- 提供されたどの記事にも情報がない場合は「[[NO_INFO]]」とだけ回答する
- 簡潔かつ分かりやすく説明する（300文字以内）
- 回答の最後に、参照した記事のタイトルを「[[REF:記事タイトル]]」の形式で必ず記載する（複数ある場合は複数記載）
"""

    response = model.generate_content(prompt)
    return response.text

# セッションステートの初期化
if 'random_articles' not in st.session_state:
    df = load_data()
    st.session_state.random_articles = get_random_articles(df, 10)

if 'selected_article_idx' not in st.session_state:
    st.session_state.selected_article_idx = None

if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

if 'referenced_articles' not in st.session_state:
    st.session_state.referenced_articles = {}

# データ読み込み（RAG検索用）
df_full = load_data()

# メイン画面
st.title("💬 拡張コラムチャット")
st.caption("記事を選んで質問してください（他の記事も自動検索）")

# 記事が選択されていない場合：記事一覧を表示
if st.session_state.selected_article_idx is None:
    st.divider()
    st.subheader("📚 記事一覧")
    st.markdown("質問したい記事を選択してください")

    # 記事一覧を表示
    articles = st.session_state.random_articles

    # 2列レイアウトで表示
    col1, col2 = st.columns(2)

    for idx, row in articles.iterrows():
        with col1 if idx % 2 == 0 else col2:
            with st.container():
                st.markdown(f"### 📄 記事 {idx + 1}")
                st.markdown(f"**{row['title']}**")

                # 要約があれば表示
                if pd.notna(row.get('summary')) and row['summary']:
                    with st.expander("要約を見る"):
                        st.write(row['summary'])

                if st.button(f"この記事でチャット", key=f"select_{idx}", type="primary"):
                    st.session_state.selected_article_idx = idx
                    st.rerun()

                st.divider()

    # 記事を再選択するボタン
    if st.button("🔄 別の記事を表示", type="secondary"):
        df = load_data()
        st.session_state.random_articles = get_random_articles(df, 10)
        st.rerun()

# 記事が選択されている場合：チャット画面を表示
else:
    selected_idx = st.session_state.selected_article_idx
    selected_article = st.session_state.random_articles.iloc[selected_idx]

    # 選択中の記事を表示
    st.info(f"📄 選択中の記事: **{selected_article['title']}**")

    # 記事選択に戻るボタン
    if st.button("← 記事一覧に戻る"):
        st.session_state.selected_article_idx = None
        st.rerun()

    st.divider()

    # チャット履歴の取得（記事ごとに管理）
    article_key = f"article_{selected_idx}"
    if article_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[article_key] = [
            {
                "role": "assistant",
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。\n\nこの記事に情報がない場合は、他の記事も自動で検索して回答します。"
            }
        ]

    if article_key not in st.session_state.referenced_articles:
        st.session_state.referenced_articles[article_key] = []

    chat_history = st.session_state.chat_histories[article_key]
    referenced_articles = st.session_state.referenced_articles[article_key]

    # チャット履歴を表示
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # 参照元があれば表示（expander形式）
            if "references" in message and message["references"]:
                st.markdown("---")
                st.markdown("**📚 参照元記事:**")
                for ref in message["references"]:
                    # 辞書形式の場合（新形式）
                    if isinstance(ref, dict):
                        with st.expander(ref['title']):
                            st.write(ref['body'])
                    else:
                        # 文字列の場合（旧形式、後方互換性）
                        st.markdown(f"- {ref}")

    # ユーザー入力
    if user_input := st.chat_input("質問を入力してください"):
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.write(user_input)

        # チャット履歴に追加
        chat_history.append({"role": "user", "content": user_input})

        # AI回答を生成
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                # 最近の会話履歴
                recent_history = chat_history[-7:-1] if len(chat_history) > 7 else chat_history[:-1]

                # ステップ1: 選択記事 + 参照済み記事で回答
                response = generate_response_step1(
                    selected_article,
                    referenced_articles,
                    user_input,
                    recent_history
                )

                references = []
                new_search_results = []

                # 回答判定
                if "[[SEARCH_NEEDED]]" in response:
                    # ステップ2: RAG検索を実行
                    with st.spinner("他の記事を検索中..."):
                        # 除外するタイトル（選択記事 + 既存参照記事）
                        exclude_titles = [selected_article['title']]
                        exclude_titles += [r['title'] for r in referenced_articles]

                        # 類似記事を検索
                        new_search_results = search_similar_articles(
                            df_full,
                            user_input,
                            top_k=3,
                            exclude_titles=exclude_titles
                        )

                        # 検索結果をセッションに保存
                        st.session_state.referenced_articles[article_key].extend(new_search_results)

                        # ステップ2: 検索結果を含めて再回答
                        response = generate_response_step2(
                            selected_article,
                            referenced_articles,
                            new_search_results,
                            user_input,
                            recent_history
                        )

                # 最終判定
                if "[[NO_INFO]]" in response:
                    final_response = "申し訳ございません。提供されているコラムからは、ご質問に関する情報を見つけることができませんでした。"
                    references = []
                else:
                    # [[REF:記事タイトル]] を抽出
                    ref_pattern = r'\[\[REF:(.+?)\]\]'
                    ref_titles = re.findall(ref_pattern, response)

                    # 回答から[[REF:...]]を削除
                    final_response = re.sub(ref_pattern, '', response)
                    final_response = final_response.replace("[[SEARCH_NEEDED]]", "").replace("[[NO_INFO]]", "").strip()

                    # 参照元を収集（タイトルと本文を含む辞書形式）
                    references = []
                    # 全ての利用可能な記事（選択記事 + 参照済み記事 + 新規検索結果）
                    all_articles = [{'title': selected_article['title'], 'body': selected_article['body']}]
                    all_articles += [{'title': r['title'], 'body': r['body']} for r in referenced_articles]
                    all_articles += [{'title': r['title'], 'body': r['body']} for r in new_search_results]

                    for ref_title in ref_titles:
                        # タイトルが一致する記事を探す
                        for article in all_articles:
                            if ref_title in article['title'] or article['title'] in ref_title:
                                if article not in references:
                                    references.append(article)
                                break

                st.write(final_response)

                # 参照元を表示（expander形式）
                if references:
                    st.markdown("---")
                    st.markdown("**📚 参照元記事:**")
                    for ref in references:
                        with st.expander(ref['title']):
                            st.write(ref['body'])

        # AI回答をチャット履歴に追加
        chat_history.append({
            "role": "assistant",
            "content": final_response,
            "references": references
        })
        st.rerun()

    # チャット履歴をクリア
    if st.button("🗑️ チャット履歴をクリア"):
        st.session_state.chat_histories[article_key] = [
            {
                "role": "assistant",
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。\n\nこの記事に情報がない場合は、他の記事も自動で検索して回答します。"
            }
        ]
        st.session_state.referenced_articles[article_key] = []
        st.rerun()
