import streamlit as st
import pandas as pd
import google.generativeai as genai
import pickle
import random

# ページ設定
st.set_page_config(
    page_title="コラムチャット",
    page_icon="💬",
    layout="wide"
)

# Gemini API設定（Streamlit Secretsから読み込み）
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_data
def load_data():
    """データを読み込み"""
    df = pd.read_excel("RAGデータ/columns_articles_with_summary.xlsx")
    return df

def get_random_articles(df, n=10):
    """ランダムにn件の記事を選択"""
    return df.sample(n=min(n, len(df))).reset_index(drop=True)

def generate_chat_response(article_title, article_body, user_message, chat_history):
    """チャット回答を生成"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # チャット履歴を構築
    history_text = ""
    if chat_history:
        history_text = "\n\n【これまでの会話履歴】\n"
        for msg in chat_history:
            history_text += f"{msg['role']}: {msg['content']}\n"
    
    prompt = f"""あなたは相続に関する専門家です。
以下の記事の内容に基づいて、ユーザーの質問に答えてください。

【記事タイトル】
{article_title}

【記事本文】
{article_body}
{history_text}

【ユーザーの質問】
{user_message}

【回答のルール】
- 記事の内容に基づいて丁寧に回答する
- 記事に書かれていない内容は推測せず、「記事には記載されていません」と答える
- 記事タイトルを引用する
- 簡潔かつ分かりやすく説明する（300文字以内）
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

# メイン画面
st.title("💬 コラムチャット")
st.caption("記事を選んで質問してください")

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
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。"
            }
        ]
    
    chat_history = st.session_state.chat_histories[article_key]
    
    # チャット履歴を表示
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
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
                # 最後の数件の会話のみを履歴として渡す（トークン制限対策）
                recent_history = chat_history[-6:-1] if len(chat_history) > 6 else chat_history[:-1]
                
                response = generate_chat_response(
                    selected_article['title'],
                    selected_article['body'],
                    user_input,
                    recent_history
                )
                st.write(response)
        
        # AI回答をチャット履歴に追加
        chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    # チャット履歴をクリア
    if st.button("🗑️ チャット履歴をクリア"):
        st.session_state.chat_histories[article_key] = [
            {
                "role": "assistant",
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。"
            }
        ]
        st.rerun()