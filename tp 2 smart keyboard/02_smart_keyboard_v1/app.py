"""Lab 02 - Smart Keyboard V1 (Streamlit UI). Run:  streamlit run app.py"""
import random

import streamlit as st

from src.pipeline import NGramModel, EOS, load_documents, backoff_top_k

st.set_page_config(page_title="Smart Keyboard V1", page_icon="⌨️", layout="wide")

st.markdown(
    """
<style>
    .block-container {padding-top: 1.6rem; max-width: 1180px;}

    .hero {background: linear-gradient(120deg, #6a5cff 0%, #38bdf8 55%, #34d399 100%);
           border-radius: 22px; padding: 1.5rem 2rem 1.4rem 2rem; margin-bottom: 1.4rem;
           box-shadow: 0 10px 30px rgba(106, 92, 255, .25);}
    .hero h1 {color: #fff; font-size: 2.1rem; margin: 0; letter-spacing: .5px;}
    .hero p  {color: rgba(255,255,255,.92); margin: .35rem 0 0 0; font-size: .98rem;}

    div[data-testid="stTextInput"] label {font-weight: 600; color: #334155;}
    div[data-testid="stTextInput"] input {font-size: 1.25rem !important; font-weight: 600;
        border-radius: 14px !important; padding: .85rem 1rem !important;
        border: 2px solid #c7d2fe !important; background: #f8fafc;}
    div[data-testid="stTextInput"] input:focus {border-color: #6a5cff !important;
        box-shadow: 0 0 0 4px rgba(106,92,255,.18);}

    .card {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
           padding: .85rem .7rem .7rem .7rem; text-align: center;
           box-shadow: 0 4px 14px rgba(15,23,42,.06); height: 100%;}
    .card .word {font-size: 1.5rem; font-weight: 700; color: #0f172a; line-height: 1.2;
                 word-break: break-word;}
    .card .pct  {font-size: .85rem; color: #64748b; font-weight: 600; margin-top: .15rem;}
    .card .eos  {font-size: 1.25rem; color: #6a5cff; font-weight: 700;}
    .bar {height: 8px; border-radius: 6px; background: #e2e8f0; margin-top: .55rem; overflow: hidden;}
    .bar > span {display: block; height: 100%; border-radius: 6px;
                 background: linear-gradient(90deg, #6a5cff, #38bdf8);}
    div[data-testid="stButton"] > button {border-radius: 12px; font-weight: 600;
        border: 1px solid #c7d2fe; color: #4338ca; background: #eef2ff; width: 100%;}
    div[data-testid="stButton"] > button:hover {background: #6a5cff; color: #fff;
        border-color: #6a5cff;}

    .stat {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px;
           padding: .8rem 1rem; margin-bottom: .7rem;}
    .stat .k {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8;}
    .stat .v {font-size: 1.45rem; font-weight: 700; color: #0f172a;}
    .hint {background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
           border-radius: 12px; padding: .55rem .8rem; font-size: .9rem; margin-bottom: .8rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    docs = load_documents()
    sentences = [s for d in docs.values() for s in d]
    return [NGramModel(sentences, n, 0.1) for n in (3, 2, 1)], sentences


(tri, bi, uni), sentences = get_models()

st.markdown(
    """
<div class="hero">
  <h1>⌨️ Smart Keyboard V1</h1>
  <p>Next-word suggestions from a trigram model with add-k smoothing &amp; back-off —
     click a card to keep typing.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Preferences")
    top_k = st.slider("Number of suggestions", 2, 6, 3, help="How many candidate words to show.")
    st.divider()
    st.markdown(
        """
**🧠 How it works**
1. Counts 1/2/3-gram stats over the corpus.
2. Scores every word with add-k smoothing (k = 0.1).
3. If the trigram context was never seen, **back off** to the bigram, then unigram.

**📄 Corpus** — tiny *synthetic* dataset, so treat the percentages as model
estimates, not real typing statistics."""
    )

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.text_input("Type the beginning of a sentence ✍️", key="typed")
    words = st.session_state["typed"].lower().replace(".", " . ").split()
    unknown = [w for w in words if w not in uni.vocab]

    if unknown:
        st.markdown(
            f'<div class="hint">⚠️ Not in the training vocabulary (treated as &lt;unk&gt;): '
            f'<b>{", ".join(unknown)}</b></div>',
            unsafe_allow_html=True,
        )
    if not words:
        st.markdown('<div class="hint">💡 Type at least one word — showing how sentences usually start.</div>',
                    unsafe_allow_html=True)

    suggestions, order = backoff_top_k(tri, bi, uni, words, top_k)
    st.markdown(
        f"#### 💡 Suggestions &nbsp;"
        f"<span style='color:#64748b;font-weight:500'>· used the "
        f"<b style='color:#4338ca'>{order}</b> model</span>",
        unsafe_allow_html=True,
    )

    if suggestions:
        top_p = max(p for _, p in suggestions)
        cols = st.columns(len(suggestions))
        for col, (w, p) in zip(cols, suggestions):
            with col:
                eos = w == EOS
                word_html = '<span class="eos">(end of sentence)</span>' if eos else f'<span class="word">{w}</span>'
                frac = 100 if top_p == 0 else p / top_p * 100
                st.markdown(
                    f'<div class="card">{word_html}'
                    f'<div class="pct">{p:.1%}</div>'
                    f'<div class="bar"><span style="width:{frac:.0f}%"></span></div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Add ➜", key=f"add_{w}_{p:.6f}", use_container_width=True):
                    ending = "." if eos else " " + w
                    st.session_state["typed"] = st.session_state["typed"].rstrip() + ending
                    st.rerun()
    else:
        st.info("No suggestion available for this context.")

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("🎲 Surprise me", use_container_width=True):
        st.session_state["typed"] = " ".join(random.choice(sentences))
        st.rerun()
    if c2.button("🧹 Clear", use_container_width=True):
        st.session_state["typed"] = ""
        st.rerun()

with right:
    st.markdown("#### 📊 Model snapshot")
    st.markdown(
        f'<div class="stat"><div class="k">Vocabulary size</div><div class="v">{uni.V} words</div></div>'
        f'<div class="stat"><div class="k">Training sentences</div><div class="v">{len(sentences)}</div></div>'
        f'<div class="stat"><div class="k">Smoothing</div><div class="v">add-k, k = 0.1</div></div>'
        f'<div class="stat"><div class="k">Orders in play</div><div class="v">3 → 2 → 1</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("🔎 Peek at the current context"):
        if words:
            st.code(" ".join(words), language=None)
            st.caption(f"Cleaned context: {' '.join(tri.clean(words))}")
        else:
            st.caption("Nothing typed yet.")
