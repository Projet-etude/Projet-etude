import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import requests
from collections import Counter
from pathlib import Path


try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bluesky Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #11111b; }
[data-testid="stSidebar"] { background: #1e1e2e; border-right: 1px solid #313244; }
[data-testid="stSidebar"] .stMarkdown p { color: #a6adc8; font-size: 0.8rem; }

.kpi { background:#1e1e2e; border:1px solid #313244; border-radius:10px;
        padding:18px 14px; text-align:center; height:100%; }
.kpi-val { font-size:1.9rem; font-weight:700; margin:0; line-height:1.2; }
.kpi-lbl { font-size:0.72rem; color:#a6adc8; margin-top:5px;
            text-transform:uppercase; letter-spacing:.06em; }

.filter-badge { background:#313244; border-radius:20px; padding:2px 10px;
                font-size:0.75rem; color:#cdd6f4; display:inline-block; margin:2px; }
.active-filters { padding:8px 0 4px; }

.c-pos { color:#a6e3a1; }
.c-neu { color:#89b4fa; }
.c-neg { color:#f38ba8; }
.c-acc { color:#cba6f7; }
.c-def { color:#cdd6f4; }

div[data-testid="stHorizontalBlock"] > div { min-width: 0; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {"Positif": "#a6e3a1", "Neutre": "#89b4fa", "Négatif": "#f38ba8"}

STOPWORDS = {
    "the","and","for","that","this","with","are","was","have","not","but",
    "you","they","from","his","her","its","been","more","will","also","than",
    "what","who","all","one","has","can","out","just","into","their","about",
    "would","when","les","des","est","une","que","qui","dans","sur","par",
    "pour","pas","plus","avec","ces","elle","ils","https","http","www","com",
    "org","net","amp","your","our","any","very","some","how","now","new","get",
    "got","him","did","there","said","too","let","use","may","see","an","it",
    "is","be","do","at","by","as","or","if","up","so","we","me","my","no",
    "go","us","re","don","isn","aren","wasn","didn","doesn","haven","hadn",
    "won","couldn","shouldn","wouldn","ve","ll","just","like","even","still",
    "really","think","know","make","time","here","good","going","right","need",
}

def hex_to_rgba(h, a=0.18):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

def clean_label(raw):
    parts = raw.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts).title()

def top_words(texts, n=15):
    words = []
    for t in texts:
        words.extend(
            w for w in re.findall(r"\b[a-zA-Zà-ÿÀ-Ÿ]{3,}\b", str(t).lower())
            if w not in STOPWORDS
        )
    return Counter(words).most_common(n)

def kpi(val, lbl, cls="c-def"):
    return f'<div class="kpi"><p class="kpi-val {cls}">{val}</p><p class="kpi-lbl">{lbl}</p></div>'

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cdd6f4", size=12),
    margin=dict(t=16, b=16, l=16, r=16),
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/labeled_data_with_topics.csv")
    df["topic_label"] = df["topic"].apply(
        lambda x: clean_label(x) if x != "Autre" else "Non classifié"
    )
    return df

@st.cache_data
def load_metrics():
    p = Path("models/sentiment/export/metrics.json")
    return json.load(open(p)) if p.exists() else None

df_full = load_data()
metrics = load_metrics()

ALL_SENTIMENTS  = sorted(df_full["sentiment"].unique().tolist())
ALL_TOPICS      = sorted([t for t in df_full["topic_label"].unique() if t != "Non classifié"])
ALL_LANGS       = sorted(df_full["language"].dropna().unique().tolist())

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION + FILTERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Bluesky Analytics")
    page = st.radio(
        "Navigation",
        ["Vue d'ensemble", "Sentiments", "Topics", "Performance du modèle", "Prediction en temps reel"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Filtres")

    # 1. Sentiment
    sel_sentiments = st.multiselect(
        "Sentiment",
        options=ALL_SENTIMENTS,
        default=ALL_SENTIMENTS,
        placeholder="Tous les sentiments",
    )

    # 2. Topics
    include_unclassified = st.checkbox("Inclure les posts non classifiés", value=True)
    sel_topics = st.multiselect(
        "Topics",
        options=ALL_TOPICS,
        default=[],
        placeholder="Tous les topics",
    )

    # 3. Score de confiance
    score_range = st.slider(
        "Score de confiance (topic)",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.05,
        format="%.2f",
    )

    # 4. Recherche mot-clé
    keyword = st.text_input("Mot-clé dans le texte", placeholder="ex: vaccine, AI...")

    st.divider()

    # Reset button
    if st.button("Reinitialiser les filtres", use_container_width=True):
        st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()

# Sentiment filter
if sel_sentiments:
    df = df[df["sentiment"].isin(sel_sentiments)]

# Topic filter
if sel_topics:
    df = df[df["topic_label"].isin(sel_topics)]
elif not include_unclassified:
    df = df[df["topic_label"] != "Non classifié"]

# Score filter (applies only to classified posts; unclassified have score 0)
df = df[
    (df["topic_score"].between(score_range[0], score_range[1])) |
    (df["topic_label"] == "Non classifié")
]

# Keyword filter
if keyword.strip():
    df = df[df["text"].str.contains(keyword.strip(), case=False, na=False)]

n_filtered = len(df)
n_total = len(df_full)

# ── Filter summary bar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center;margin-top:4px'>"
        f"<span style='color:#cba6f7;font-weight:600;font-size:1.1rem'>{n_filtered:,}</span>"
        f"<span style='color:#a6adc8;font-size:0.8rem'> / {n_total:,} posts</span></div>",
        unsafe_allow_html=True,
    )

# ── Derived stats from filtered df ───────────────────────────────────────────
def safe_pct(n, total):
    return round(n / total * 100, 1) if total > 0 else 0.0

total    = len(df)
c_pos    = (df["sentiment"] == "Positif").sum()
c_neu    = (df["sentiment"] == "Neutre").sum()
c_neg    = (df["sentiment"] == "Négatif").sum()
pct_pos  = safe_pct(c_pos, total)
pct_neu  = safe_pct(c_neu, total)
pct_neg  = safe_pct(c_neg, total)
n_topics = df[df["topic_label"] != "Non classifié"]["topic_label"].nunique()
avg_conf = round(df[df["topic_label"] != "Non classifié"]["topic_score"].mean() * 100, 1) if n_topics > 0 else 0.0
dominant = df["sentiment"].value_counts().idxmax() if total > 0 else "—"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Vue d'ensemble":
    st.markdown("## Vue d'ensemble")
    if total == 0:
        st.warning("Aucun résultat avec les filtres actuels.")
        st.stop()

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(kpi(f"{total:,}",    "Posts analysés",  "c-def"), unsafe_allow_html=True)
    c2.markdown(kpi(str(n_topics),   "Topics détectés", "c-acc"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{pct_pos}%",   "Positif",         "c-pos"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{pct_neu}%",   "Neutre",          "c-neu"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{pct_neg}%",   "Négatif",         "c-neg"), unsafe_allow_html=True)
    c6.markdown(kpi(f"{avg_conf}%",  "Confiance moy.",  "c-acc"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Répartition des sentiments")
        fig = go.Figure(go.Pie(
            labels=["Positif","Neutre","Négatif"],
            values=[c_pos, c_neu, c_neg],
            hole=0.55,
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in ["Positif","Neutre","Négatif"]]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} posts (%{percent})<extra></extra>",
        ))
        fig.update_layout(**LAYOUT, height=290, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Posts par sentiment")
        fig2 = go.Figure(go.Bar(
            y=["Positif","Neutre","Négatif"],
            x=[c_pos, c_neu, c_neg],
            orientation="h",
            marker_color=[SENTIMENT_COLORS[s] for s in ["Positif","Neutre","Négatif"]],
            text=[f"{c_pos:,}", f"{c_neu:,}", f"{c_neg:,}"],
            textposition="outside",
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ))
        fig2.update_layout(**LAYOUT, height=290,
                           xaxis=dict(gridcolor="#313244"),
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    # Word cloud / top words
    st.markdown("#### Mots les plus fréquents (sélection filtrée)")
    if WORDCLOUD_AVAILABLE and total > 0:
        text_all = " ".join(df["text"].dropna().astype(str))
        wc = WordCloud(width=950, height=330, background_color="#11111b",
                       colormap="cool", stopwords=STOPWORDS, max_words=130,
                       prefer_horizontal=0.85).generate(text_all)
        fig_wc, ax = plt.subplots(figsize=(13, 4))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig_wc.patch.set_facecolor("#11111b")
        st.pyplot(fig_wc); plt.close(fig_wc)
    else:
        tw = top_words(df["text"], 20)
        wdf = pd.DataFrame(tw, columns=["Mot","Freq"])
        fig_tb = px.bar(wdf, x="Freq", y="Mot", orientation="h",
                        color="Freq", color_continuous_scale="Blues")
        fig_tb.update_layout(**LAYOUT, height=420,
                             xaxis=dict(gridcolor="#313244"),
                             yaxis=dict(autorange="reversed"),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_tb, use_container_width=True)

    # Data table (sample)
    with st.expander(f"Apercu des données filtrées ({total:,} posts)"):
        st.dataframe(
            df[["text","sentiment","topic_label","topic_score"]]
            .rename(columns={"text":"Texte","sentiment":"Sentiment",
                             "topic_label":"Topic","topic_score":"Score"})
            .head(200),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — SENTIMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiments":
    st.markdown("## Analyse des sentiments")
    if total == 0:
        st.warning("Aucun résultat avec les filtres actuels.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi(dominant,      "Sentiment dominant", "c-neu"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{c_pos:,}",  "Posts positifs",    "c-pos"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{c_neu:,}",  "Posts neutres",     "c-neu"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{c_neg:,}",  "Posts négatifs",    "c-neg"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution détaillée")
        fig = go.Figure(go.Pie(
            labels=["Positif","Neutre","Négatif"],
            values=[c_pos, c_neu, c_neg],
            hole=0.55,
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in ["Positif","Neutre","Négatif"]]),
            textinfo="label+percent+value",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        fig.update_layout(**LAYOUT, height=300, showlegend=True,
                          legend=dict(font=dict(color="#cdd6f4")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Top mots par sentiment")
        sent_sel = st.selectbox("Sentiment", ["Positif","Neutre","Négatif"], key="sent_words")
        subset = df[df["sentiment"] == sent_sel]["text"]
        tw = top_words(subset, 15)
        if tw:
            wdf = pd.DataFrame(tw, columns=["Mot","Freq"])
            fig_tw = px.bar(wdf, x="Freq", y="Mot", orientation="h",
                            color_discrete_sequence=[SENTIMENT_COLORS[sent_sel]])
            fig_tw.update_layout(**LAYOUT, height=300,
                                 yaxis=dict(autorange="reversed"),
                                 xaxis=dict(gridcolor="#313244"))
            st.plotly_chart(fig_tw, use_container_width=True)
        else:
            st.info("Pas assez de données.")

    # Sentiment par topic (heatmap)
    st.markdown("#### Répartition des sentiments par topic (Top 12)")
    real = df[df["topic_label"] != "Non classifié"]
    if len(real) > 0:
        top12 = real.groupby("topic_label").size().nlargest(12).index.tolist()
        heat_df = (
            real[real["topic_label"].isin(top12)]
            .groupby(["topic_label","sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        heat_df_long = heat_df.melt(id_vars="topic_label", var_name="Sentiment", value_name="Posts")
        fig_heat = px.bar(
            heat_df_long, x="topic_label", y="Posts", color="Sentiment",
            barmode="stack",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig_heat.update_layout(**LAYOUT, height=340,
                               xaxis=dict(tickangle=-30, title=""),
                               yaxis=dict(gridcolor="#313244"),
                               legend=dict(font=dict(color="#cdd6f4")))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Word clouds
    if WORDCLOUD_AVAILABLE:
        st.markdown("#### Word Clouds par sentiment")
        wc_sents = [s for s in ["Positif","Négatif"] if (df["sentiment"] == s).sum() > 5]
        if wc_sents:
            wc_cols = st.columns(len(wc_sents))
            cmaps = {"Positif":"Greens","Négatif":"Reds"}
            for col_w, s in zip(wc_cols, wc_sents):
                txt = " ".join(df[df["sentiment"] == s]["text"].dropna().astype(str))
                wc = WordCloud(width=550, height=280, background_color="#11111b",
                               colormap=cmaps[s], stopwords=STOPWORDS, max_words=70).generate(txt)
                fig_w, ax = plt.subplots(figsize=(7, 3.5))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                ax.set_title(f"Posts {s}", color="#cdd6f4", fontsize=12, pad=6)
                fig_w.patch.set_facecolor("#11111b")
                col_w.pyplot(fig_w); plt.close(fig_w)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TOPICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Topics":
    st.markdown("## Analyse des topics")
    real = df[df["topic_label"] != "Non classifié"].copy()

    if len(real) == 0:
        st.warning("Aucun topic trouvé avec les filtres actuels.")
        st.stop()

    tc = real.groupby("topic_label").size().sort_values(ascending=False)
    top_topic = tc.idxmax()
    avg_pp = round(tc.mean(), 1)

    c1,c2,c3 = st.columns(3)
    c1.markdown(kpi(str(n_topics),   "Topics présents",         "c-acc"), unsafe_allow_html=True)
    c2.markdown(kpi(top_topic,       "Topic le plus fréquent",  "c-neu"), unsafe_allow_html=True)
    c3.markdown(kpi(str(avg_pp),     "Moy. posts / topic",      "c-def"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top N slider
    top_n = st.slider("Nombre de topics à afficher", 5, min(30, n_topics), min(15, n_topics), key="top_n")

    col_a, col_b = st.columns([3,2])

    with col_a:
        st.markdown(f"#### Top {top_n} topics")
        topN = tc.head(top_n).reset_index()
        topN.columns = ["Topic","Posts"]
        fig_bar = px.bar(topN.iloc[::-1], x="Posts", y="Topic", orientation="h",
                         color="Posts", color_continuous_scale="Blues", text="Posts")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(**LAYOUT, height=max(350, top_n*28),
                              coloraxis_showscale=False,
                              xaxis=dict(gridcolor="#313244"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("#### Répartition (donut)")
        top10 = tc.head(10)
        others = tc.iloc[10:].sum()
        labels = list(top10.index) + (["Autres"] if others > 0 else [])
        values = list(top10.values) + ([others] if others > 0 else [])
        fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.45,
                                   textinfo="label+percent",
                                   hovertemplate="%{label}: %{value:,}<extra></extra>"))
        fig_pie.update_layout(**LAYOUT, height=max(350, top_n*28), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Treemap
    st.markdown("#### Treemap")
    tree_df = tc.head(25).reset_index()
    tree_df.columns = ["Topic","Posts"]
    fig_tree = px.treemap(tree_df, path=["Topic"], values="Posts",
                          color="Posts", color_continuous_scale="Blues")
    fig_tree.update_layout(**LAYOUT, height=360, font=dict(color="#1e1e2e"))
    st.plotly_chart(fig_tree, use_container_width=True)

    # Sentiment par topic — Radar
    st.markdown("#### Radar — Sentiments vs Topics (Top 8)")
    top8 = tc.head(8).index.tolist()
    radar = (
        real[real["topic_label"].isin(top8)]
        .groupby(["topic_label","sentiment"])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    for s in ["Positif","Neutre","Négatif"]:
        if s not in radar.columns:
            radar[s] = 0

    cats = list(radar["topic_label"])
    cats_c = cats + [cats[0]]
    fig_rad = go.Figure()
    for sent, color in SENTIMENT_COLORS.items():
        vals = list(radar[sent])
        vals_c = vals + [vals[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c, fill="toself", name=sent,
            line_color=color, fillcolor=hex_to_rgba(color, 0.15), opacity=0.9,
        ))
    fig_rad.update_layout(**LAYOUT, height=420,
                          polar=dict(
                              bgcolor="rgba(0,0,0,0)",
                              angularaxis=dict(tickcolor="#313244", linecolor="#313244"),
                              radialaxis=dict(gridcolor="#313244", linecolor="#313244"),
                          ),
                          showlegend=True, legend=dict(font=dict(color="#cdd6f4")))
    st.plotly_chart(fig_rad, use_container_width=True)

    # Word cloud per topic
    if WORDCLOUD_AVAILABLE:
        st.markdown("#### Word Cloud par topic")
        topic_sel = st.selectbox("Choisir un topic", sorted(real["topic_label"].unique()), key="topic_wc")
        txt = " ".join(real[real["topic_label"] == topic_sel]["text"].dropna().astype(str))
        if txt.strip():
            wc = WordCloud(width=950, height=300, background_color="#11111b",
                           colormap="plasma", stopwords=STOPWORDS, max_words=80).generate(txt)
            fig_wc, ax = plt.subplots(figsize=(13,4))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            fig_wc.patch.set_facecolor("#11111b")
            st.pyplot(fig_wc); plt.close(fig_wc)
    else:
        topic_sel = st.selectbox("Topic", sorted(real["topic_label"].unique()), key="topic_words")
        tw = top_words(real[real["topic_label"] == topic_sel]["text"], 15)
        wdf = pd.DataFrame(tw, columns=["Mot","Freq"])
        fig_wt = px.bar(wdf, x="Freq", y="Mot", orientation="h",
                        color_discrete_sequence=["#cba6f7"])
        fig_wt.update_layout(**LAYOUT, height=360, yaxis=dict(autorange="reversed"),
                             xaxis=dict(gridcolor="#313244"))
        st.plotly_chart(fig_wt, use_container_width=True)

    # Score distribution by topic
    st.markdown("#### Distribution des scores de confiance par topic")
    top_score = tc.head(12).index.tolist()
    fig_box = px.box(
        real[real["topic_label"].isin(top_score)],
        x="topic_score", y="topic_label",
        color="topic_label",
        orientation="h",
        labels={"topic_score":"Score de confiance","topic_label":"Topic"},
    )
    fig_box.update_layout(**LAYOUT, height=420, showlegend=False,
                          xaxis=dict(gridcolor="#313244"))
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PERFORMANCE DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performance du modèle":
    st.markdown("## Performance du modèle")
    st.caption("cardiffnlp/twitter-roberta-base-sentiment-latest · Fine-tuné sur les données Bluesky")

    if metrics:
        fm      = metrics["final_metrics"]
        acc     = round(fm.get("eval_accuracy", 0) * 100, 1)
        f1_mac  = round(fm.get("eval_f1_macro", 0) * 100, 1)
        f1_w    = round(fm.get("eval_f1_weighted", 0) * 100, 1)
        epochs  = metrics.get("training_epochs", "—")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(kpi(f"{acc}%",    "Accuracy",        "c-pos"), unsafe_allow_html=True)
        c2.markdown(kpi(f"{f1_mac}%", "F1 Macro",        "c-acc"), unsafe_allow_html=True)
        c3.markdown(kpi(f"{f1_w}%",   "F1 Weighted",     "c-neu"), unsafe_allow_html=True)
        c4.markdown(kpi(str(epochs),  "Epochs",          "c-def"), unsafe_allow_html=True)
        c5.markdown(kpi(f"{avg_conf}%","Confiance moy.", "c-pos"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Comparaison des métriques")
        names  = ["Accuracy","F1 Macro","F1 Weighted"]
        vals   = [acc, f1_mac, f1_w]
        colors = [SENTIMENT_COLORS["Positif"], "#cba6f7", SENTIMENT_COLORS["Neutre"]]
        fig_m = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker_color=colors,
            text=[f"{v}%" for v in vals], textposition="outside",
        ))
        fig_m.update_layout(**LAYOUT, height=210,
                            xaxis=dict(range=[0,105], ticksuffix="%", gridcolor="#313244"))
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.warning("Fichier metrics.json introuvable.")

    # Training plots
    st.markdown("#### Courbes d'entraînement")
    p_loss    = Path("models/sentiment/export/training_loss.png")
    p_metrics = Path("models/sentiment/export/training_metrics.png")
    ic1, ic2  = st.columns(2)
    if p_loss.exists():
        ic1.image(str(p_loss), caption="Courbe de Loss", use_container_width=True)
    else:
        ic1.info("training_loss.png non disponible.")
    if p_metrics.exists():
        ic2.image(str(p_metrics), caption="Métriques d'entraînement", use_container_width=True)
    else:
        ic2.info("training_metrics.png non disponible.")

    # Confidence distribution (uses filtered df)
    st.markdown("#### Distribution des scores de confiance (données filtrées)")
    conf_vals = df[df["topic_label"] != "Non classifié"]["topic_score"]
    if len(conf_vals) > 0:
        fig_hist = px.histogram(conf_vals, nbins=30,
                                color_discrete_sequence=["#cba6f7"],
                                labels={"value":"Score de confiance"})
        fig_hist.update_layout(**LAYOUT, height=280, showlegend=False,
                               xaxis=dict(gridcolor="#313244"),
                               yaxis=dict(gridcolor="#313244", title="Posts"))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Aucune donnée de confiance dans la sélection actuelle.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PREDICTION EN TEMPS REEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction en temps reel":
    st.markdown("## Prediction en temps reel")
    st.caption("Tape n'importe quelle phrase — le modele detecte automatiquement son sentiment.")

    API_URL = "http://localhost:5002"

    # Check API status
    api_ok = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        api_ok = r.status_code == 200
    except Exception:
        api_ok = False

    col_s1, col_s2 = st.columns(2)
    if api_ok:
        col_s1.markdown(
            "<span style='color:#a6e3a1;font-size:0.82rem'>Modele checkpoint-1236 — API connectee (port 5002)</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error("L'API n'est pas demarree. Lance : `python src/api_sentiment.py`")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    user_text = st.text_area(
        "Entrez votre texte ici",
        placeholder="Ex: This product is absolutely amazing and works perfectly!",
        height=130,
        max_chars=512,
    )

    col_btn, col_clear = st.columns([1, 5])
    analyze = col_btn.button("Analyser", type="primary")
    if col_clear.button("Effacer l'historique"):
        st.session_state.pop("pred_history", None)
        st.rerun()

    SENT_ICON  = {"Positif": "POSITIF", "Neutre": "NEUTRE", "Negatif": "NEGATIF",
                  "positive": "POSITIF", "neutral": "NEUTRE", "negative": "NEGATIF"}
    SENT_CLASS = {"Positif": "c-pos", "Neutre": "c-neu", "Negatif": "c-neg",
                  "positive": "c-pos", "neutral": "c-neu", "negative": "c-neg"}
    LABEL_MAP  = {"positive": "Positif", "neutral": "Neutre", "negative": "Negatif",
                  "Positif": "Positif", "Neutre": "Neutre", "Negatif": "Negatif",
                  "Négatif": "Negatif"}

    if analyze and user_text.strip():
        with st.spinner("Analyse en cours..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text.strip()},
                    timeout=15,
                )
                result = resp.json()
                result["source"] = "checkpoint-1236"

                if "error" in result:
                    st.error(f"Erreur API : {result.get('error', 'Inconnue')}")
                else:
                    raw_sent    = result.get("sentiment", "Neutre")
                    sentiment   = LABEL_MAP.get(raw_sent, raw_sent)
                    confidence  = result.get("confidence", 0.0)
                    all_scores  = result.get("all_scores", [])

                    # Store in session history
                    if "pred_history" not in st.session_state:
                        st.session_state["pred_history"] = []
                    st.session_state["pred_history"].insert(0, {
                        "text": user_text.strip()[:80] + ("..." if len(user_text) > 80 else ""),
                        "sentiment": sentiment,
                        "confidence": confidence,
                    })

                    # Result display
                    st.markdown("<br>", unsafe_allow_html=True)
                    source = result.get("source", "")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.markdown(
                        kpi(sentiment, "Sentiment detecte", SENT_CLASS.get(sentiment, "c-def")),
                        unsafe_allow_html=True,
                    )
                    r2.markdown(
                        kpi(f"{round(confidence * 100, 1)}%", "Confiance", "c-acc"),
                        unsafe_allow_html=True,
                    )
                    r3.markdown(
                        kpi(str(len(user_text.strip())), "Caracteres", "c-def"),
                        unsafe_allow_html=True,
                    )
                    r4.markdown(
                        kpi(source, "Moteur", "c-def"),
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Scores bar chart
                    if all_scores:
                        st.markdown("#### Scores par categorie")
                        scores_data = [
                            {
                                "Sentiment": LABEL_MAP.get(s["label"], s["label"]),
                                "Score": round(s["score"] * 100, 1),
                            }
                            for s in all_scores
                        ]
                        scores_df = pd.DataFrame(scores_data).sort_values("Score", ascending=True)
                        bar_colors = [
                            SENTIMENT_COLORS.get(row["Sentiment"], "#cdd6f4")
                            for _, row in scores_df.iterrows()
                        ]
                        fig_scores = go.Figure(go.Bar(
                            y=scores_df["Sentiment"],
                            x=scores_df["Score"],
                            orientation="h",
                            marker_color=bar_colors,
                            text=[f"{v}%" for v in scores_df["Score"]],
                            textposition="outside",
                        ))
                        fig_scores.update_layout(
                            **LAYOUT, height=200,
                            xaxis=dict(range=[0, 105], ticksuffix="%", gridcolor="#313244"),
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)

                    # Gauge de confiance
                    st.markdown("#### Jauge de confiance")
                    gauge_color = SENTIMENT_COLORS.get(sentiment, "#cdd6f4")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(confidence * 100, 1),
                        number={"suffix": "%", "font": {"color": "#cdd6f4", "size": 28}},
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#313244",
                                      tickfont=dict(color="#a6adc8")),
                            bar=dict(color=gauge_color),
                            bgcolor="#1e1e2e",
                            bordercolor="#313244",
                            steps=[
                                dict(range=[0, 50],  color="#181825"),
                                dict(range=[50, 75], color="#1e1e2e"),
                                dict(range=[75, 100], color="#181825"),
                            ],
                        ),
                    ))
                    fig_gauge.update_layout(
                        **LAYOUT, height=250,
                        font=dict(color="#cdd6f4"),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            except requests.exceptions.Timeout:
                st.error("L'API met trop de temps a repondre. Verifie qu'elle est bien lancee.")
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif analyze and not user_text.strip():
        st.warning("Veuillez entrer un texte avant d'analyser.")

    # Historique des predictions
    if st.session_state.get("pred_history"):
        st.markdown("---")
        st.markdown("#### Historique de la session")
        hist_df = pd.DataFrame(st.session_state["pred_history"])
        hist_df.columns = ["Texte", "Sentiment", "Confiance"]
        hist_df["Confiance"] = hist_df["Confiance"].apply(lambda x: f"{round(x*100,1)}%")

        row_tones = []
        tone_map = {"Positif": "success", "Neutre": "info", "Negatif": "danger"}
        for _, row in hist_df.iterrows():
            row_tones.append(tone_map.get(row["Sentiment"]))

        st.dataframe(
            hist_df,
            use_container_width=True,
            hide_index=True,
        )
