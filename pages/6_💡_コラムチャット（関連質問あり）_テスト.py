import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types
import pickle
import random
import re
import io
import wave
import base64
from audio_recorder_streamlit import audio_recorder

# ページ設定
st.set_page_config(
    page_title="コラムチャット（関連質問あり）",
    page_icon="💡",
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

    history_text = ""
    if chat_history:
        history_text = "\n\n【これまでの会話履歴】\n"
        for msg in chat_history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"

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
- 回答の最後に、ユーザーが次に聞きそうな関連質問を3つ「[[Q:質問文]]」の形式で記載する
"""

    response = model.generate_content(prompt)
    return response.text

def generate_response_step2(selected_article, referenced_articles, search_results, user_message, chat_history):
    """ステップ2: RAG検索結果を含めて回答を試みる"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    history_text = ""
    if chat_history:
        history_text = "\n\n【これまでの会話履歴】\n"
        for msg in chat_history[-6:]:
            history_text += f"{msg['role']}: {msg['content']}\n"

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
- 回答の最後に、ユーザーが次に聞きそうな関連質問を3つ「[[Q:質問文]]」の形式で記載する
"""

    response = model.generate_content(prompt)
    return response.text

def text_to_speech(text, max_retries=2):
    """テキストを音声に変換（リトライあり）"""
    if len(text) > 300:
        text = text[:300] + "。"

    for attempt in range(max_retries):
        try:
            tts_client = genai_new.Client(api_key=st.secrets["GEMINI_API_KEY"])
            tts_response = tts_client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name="Aoede"
                            )
                        )
                    ),
                ),
            )
            audio_data = tts_response.candidates[0].content.parts[0].inline_data.data
            mime_type = tts_response.candidates[0].content.parts[0].inline_data.mime_type

            if "pcm" in mime_type.lower() or "raw" in mime_type.lower():
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(audio_data)
                return wav_buffer.getvalue()
            else:
                return audio_data
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            st.warning(f"音声生成をスキップしました: {e}")
            return None

def speech_to_text(audio_bytes):
    """音声をテキストに変換（旧SDKのマルチモーダル機能を使用）"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        stt_response = model.generate_content([
            "この音声を正確にテキストに書き起こしてください。テキストのみを出力し、余計な説明は不要です。",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return stt_response.text.strip() if stt_response.text else None
    except Exception as e:
        st.error(f"STTエラー: {e}")
        return None

# セッションステートの初期化
if 's_random_articles' not in st.session_state:
    df = load_data()
    st.session_state.s_random_articles = get_random_articles(df, 10)

if 's_selected_article_idx' not in st.session_state:
    st.session_state.s_selected_article_idx = None

if 's_chat_histories' not in st.session_state:
    st.session_state.s_chat_histories = {}

if 's_referenced_articles' not in st.session_state:
    st.session_state.s_referenced_articles = {}

if 's_recorder_key' not in st.session_state:
    st.session_state.s_recorder_key = 0

if 's_should_autoplay' not in st.session_state:
    st.session_state.s_should_autoplay = False

if 's_clicked_question' not in st.session_state:
    st.session_state.s_clicked_question = None

# データ読み込み（RAG検索用）
df_full = load_data()

# メイン画面
st.title("💡 コラムチャット（関連質問あり）")
st.caption("記事を選んで質問してください（関連質問の提案あり・音声対応）")

# 記事が選択されていない場合：記事一覧を表示
if st.session_state.s_selected_article_idx is None:
    st.divider()
    st.subheader("📚 記事一覧")
    st.markdown("質問したい記事を選択してください")

    articles = st.session_state.s_random_articles

    col1, col2 = st.columns(2)

    for idx, row in articles.iterrows():
        with col1 if idx % 2 == 0 else col2:
            with st.container():
                st.markdown(f"### 📄 記事 {idx + 1}")
                st.markdown(f"**{row['title']}**")

                if pd.notna(row.get('summary')) and row['summary']:
                    with st.expander("要約を見る"):
                        st.write(row['summary'])

                if st.button(f"この記事でチャット", key=f"s_select_{idx}", type="primary"):
                    st.session_state.s_selected_article_idx = idx
                    st.rerun()

                st.divider()

    if st.button("🔄 別の記事を表示", key="s_refresh", type="secondary"):
        df = load_data()
        st.session_state.s_random_articles = get_random_articles(df, 10)
        st.rerun()

# 記事が選択されている場合：チャット画面を表示
else:
    selected_idx = st.session_state.s_selected_article_idx
    selected_article = st.session_state.s_random_articles.iloc[selected_idx]

    st.info(f"📄 選択中の記事: **{selected_article['title']}**")

    if st.button("← 記事一覧に戻る", key="s_back"):
        st.session_state.s_selected_article_idx = None
        st.rerun()

    st.divider()

    # チャット履歴の取得（記事ごとに管理）
    article_key = f"s_article_{selected_idx}"
    if article_key not in st.session_state.s_chat_histories:
        st.session_state.s_chat_histories[article_key] = [
            {
                "role": "assistant",
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。\n\nテキスト・音声で質問できます。回答後に関連質問も提案します。"
            }
        ]

    if article_key not in st.session_state.s_referenced_articles:
        st.session_state.s_referenced_articles[article_key] = []

    chat_history = st.session_state.s_chat_histories[article_key]
    referenced_articles = st.session_state.s_referenced_articles[article_key]

    # チャット履歴を表示
    for i, message in enumerate(chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # 音声があれば再生ボタンを表示
            if "audio" in message and message["audio"]:
                is_last = (i == len(chat_history) - 1)
                if is_last and st.session_state.s_should_autoplay:
                    b64_audio = base64.b64encode(message["audio"]).decode()
                    st.markdown(
                        f'<audio autoplay controls src="data:audio/wav;base64,{b64_audio}"></audio>',
                        unsafe_allow_html=True
                    )
                    st.session_state.s_should_autoplay = False
                else:
                    st.audio(message["audio"], format="audio/wav")
            # 参照元があれば表示
            if "references" in message and message["references"]:
                st.markdown("---")
                st.markdown("**📚 参照元記事:**")
                for ref in message["references"]:
                    if isinstance(ref, dict):
                        with st.expander(ref['title']):
                            st.write(ref['body'])
                    else:
                        st.markdown(f"- {ref}")
            # 関連質問を表示（最新メッセージのみボタン、過去はテキスト）
            if "suggested_questions" in message and message["suggested_questions"]:
                is_last = (i == len(chat_history) - 1)
                st.markdown("**💡 関連質問:**")
                if is_last:
                    cols = st.columns(len(message["suggested_questions"]))
                    for qi, q in enumerate(message["suggested_questions"]):
                        with cols[qi]:
                            if st.button(q, key=f"s_q_{qi}_{len(chat_history)}"):
                                st.session_state.s_clicked_question = q
                                st.rerun()
                else:
                    for q in message["suggested_questions"]:
                        st.markdown(f"- {q}")

    # クリックされた質問を取得
    clicked_q = st.session_state.s_clicked_question
    st.session_state.s_clicked_question = None

    # 音声入力
    st.markdown("**🎙️ 音声で質問:**")
    audio_bytes = audio_recorder(
        text="クリックして録音（もう一度クリックで停止）",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        pause_threshold=10.0,
        energy_threshold=0.01,
        key=f"s_recorder_{st.session_state.s_recorder_key}",
    )

    voice_input = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("音声をテキストに変換中..."):
            voice_input = speech_to_text(audio_bytes)
            if not voice_input:
                st.warning("音声を認識できませんでした。もう一度お試しください。")

    # テキスト入力
    text_input = st.chat_input("質問を入力してください")

    # 入力ソース: クリック質問 > 音声 > テキスト
    user_input = clicked_q or voice_input or text_input

    if user_input:
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.write(user_input)

        chat_history.append({"role": "user", "content": user_input})

        # AI回答を生成
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
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

                if "[[SEARCH_NEEDED]]" in response:
                    with st.spinner("他の記事を検索中..."):
                        exclude_titles = [selected_article['title']]
                        exclude_titles += [r['title'] for r in referenced_articles]

                        new_search_results = search_similar_articles(
                            df_full,
                            user_input,
                            top_k=3,
                            exclude_titles=exclude_titles
                        )

                        st.session_state.s_referenced_articles[article_key].extend(new_search_results)

                        response = generate_response_step2(
                            selected_article,
                            referenced_articles,
                            new_search_results,
                            user_input,
                            recent_history
                        )

                # 関連質問を抽出
                q_pattern = r'\[\[Q:(.+?)\]\]'
                suggested_questions = re.findall(q_pattern, response)

                # 最終判定
                if "[[NO_INFO]]" in response:
                    final_response = "申し訳ございません。提供されているコラムからは、ご質問に関する情報を見つけることができませんでした。"
                    references = []
                    suggested_questions = []
                else:
                    # [[REF:記事タイトル]] を抽出
                    ref_pattern = r'\[\[REF:(.+?)\]\]'
                    ref_titles = re.findall(ref_pattern, response)

                    # 回答からタグを削除
                    final_response = re.sub(ref_pattern, '', response)
                    final_response = re.sub(q_pattern, '', final_response)
                    final_response = final_response.replace("[[SEARCH_NEEDED]]", "").replace("[[NO_INFO]]", "").strip()

                    # 参照元を収集
                    references = []
                    all_articles = [{'title': selected_article['title'], 'body': selected_article['body']}]
                    all_articles += [{'title': r['title'], 'body': r['body']} for r in referenced_articles]
                    all_articles += [{'title': r['title'], 'body': r['body']} for r in new_search_results]

                    for ref_title in ref_titles:
                        for article in all_articles:
                            if ref_title in article['title'] or article['title'] in ref_title:
                                if article not in references:
                                    references.append(article)
                                break

                st.write(final_response)

                # TTS: 回答を音声で生成
                tts_audio = None
                with st.spinner("音声を生成中..."):
                    tts_audio = text_to_speech(final_response)

                # 参照元を表示
                if references:
                    st.markdown("---")
                    st.markdown("**📚 参照元記事:**")
                    for ref in references:
                        with st.expander(ref['title']):
                            st.write(ref['body'])

                # 関連質問を表示
                if suggested_questions:
                    st.markdown("**💡 関連質問:**")
                    cols = st.columns(len(suggested_questions))
                    for qi, q in enumerate(suggested_questions):
                        with cols[qi]:
                            if st.button(q, key=f"s_q_{qi}_{len(chat_history)}"):
                                st.session_state.s_clicked_question = q
                                st.rerun()

        # AI回答をチャット履歴に追加
        chat_history.append({
            "role": "assistant",
            "content": final_response,
            "references": references,
            "audio": tts_audio,
            "suggested_questions": suggested_questions
        })
        st.session_state.s_recorder_key += 1
        if tts_audio:
            st.session_state.s_should_autoplay = True
        st.rerun()

    # チャット履歴をクリア
    if st.button("🗑️ チャット履歴をクリア", key="s_clear"):
        st.session_state.s_chat_histories[article_key] = [
            {
                "role": "assistant",
                "content": f"こんにちは！「{selected_article['title']}」についてご質問ください。\n\nテキスト・音声で質問できます。回答後に関連質問も提案します。"
            }
        ]
        st.session_state.s_referenced_articles[article_key] = []
        st.rerun()
