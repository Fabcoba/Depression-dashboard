# ============================================================
# mental_health_dashboard.py
# Data Story: Global Mental Health & Depression Rates
# Team Project — Data Visualization Course | Tec de Monterrey
# Run: python3 -m streamlit run mental_health_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Data Story",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — clean white design ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff !important;
}
.main .block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }

section[data-testid="stSidebar"] { background: #f8f9fa !important; border-right: 1px solid #e9ecef; }
section[data-testid="stSidebar"] * { color: #495057 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.7rem !important; text-transform: uppercase;
    letter-spacing: 1px; color: #6c757d !important;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-top: 3px solid #0096c7;
    border-radius: 10px;
    padding: 20px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 0.65rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #1a2540; line-height: 1; }
.kpi-sub   { font-size: 0.7rem; color: #adb5bd; margin-top: 6px; }

.section-title {
    font-size: 0.7rem; font-weight: 700; color: #0096c7;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 36px 0 6px 0; padding-bottom: 10px;
    border-bottom: 1px solid #e9ecef;
}
.section-story {
    font-size: 0.85rem; color: #495057; line-height: 1.75;
    margin-bottom: 18px; max-width: 900px;
}
.section-story b { color: #1a2540; }
.section-story .hl { color: #0096c7; font-weight: 600; }

.dash-header {
    background: linear-gradient(135deg, #f8fbff 0%, #e8f4ff 100%);
    border: 1px solid #d6e4f0;
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 28px;
}
.dash-header h1 { color: #1a2540; font-size: 1.6rem; font-weight: 700; margin: 4px 0; }
.dash-header p  { color: #6c757d; font-size: 0.72rem; margin: 0; text-transform: uppercase; letter-spacing: 1.2px; }
.dash-header .big-idea {
    background: white; border-left: 3px solid #0096c7;
    padding: 12px 16px; margin-top: 14px; border-radius: 0 6px 6px 0;
    font-size: 0.88rem; color: #495057; font-style: italic;
}

.story-block {
    background: #f8f9fa;
    border-left: 3px solid #0096c7;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 24px;
    font-size: 0.85rem; color: #495057; line-height: 1.75;
}
.story-block b { color: #1a2540; }
.story-block .hl { color: #0096c7; font-weight: 600; }

#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Colors ───────────────────────────────────────────────────
BG     = "#ffffff"
ACCENT = "#0096c7"
GRAY   = "#d0d5dd"
TICK   = "#6c757d"
TEXT   = "#1a2540"
GRID   = "#f0f2f5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#e9ecef", "axes.labelcolor": TICK,
    "xtick.color": TICK, "ytick.color": TICK, "text.color": TEXT,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.family": "sans-serif", "font.size": 10,
})

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("MentalhealthDepressiondisorderData.csv", low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Depression (%)"] = pd.to_numeric(df["Depression (%)"], errors="coerce")
    df["Anxiety disorders (%)"] = pd.to_numeric(df["Anxiety disorders (%)"], errors="coerce")
    df = df.dropna(subset=["Year", "Depression (%)"])
    df["Year"] = df["Year"].astype(int)
    return df

df = load_data()
countries_df = df[df["Code"].notna() & (df["Code"] != "") & (df["Code"].str.len() == 3)].copy()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-size:0.7rem;color:#0096c7;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;'>Data Story</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem;color:#1a2540;font-weight:600;margin-bottom:24px;'>Global Mental Health</p>", unsafe_allow_html=True)

    year_min, year_max = int(countries_df["Year"].min()), int(countries_df["Year"].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))

    top_n = st.slider("Top N Countries", 5, 20, 10)

    st.markdown("---")
    st.markdown("<p style='font-size:0.65rem;color:#adb5bd;'>Source: Our World in Data<br>1990–2017 · 200+ countries</p>", unsafe_allow_html=True)

dff = countries_df[(countries_df["Year"] >= year_range[0]) & (countries_df["Year"] <= year_range[1])].copy()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <p>Tec de Monterrey · Data Visualization Course · Team Project</p>
  <h1>🧠 Global Mental Health & Depression Rates</h1>
  <div class="big-idea">
    <b>Big Idea:</b> Depression is the world's most underestimated health crisis —
    and the countries that ignore it today will pay the highest cost tomorrow.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────
avg_dep = dff["Depression (%)"].mean()
max_dep = dff["Depression (%)"].max()
n_countries = dff["Entity"].nunique()
trend_change = (dff[dff["Year"]==dff["Year"].max()]["Depression (%)"].mean() -
                dff[dff["Year"]==dff["Year"].min()]["Depression (%)"].mean())

k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Global Avg Depression</div><div class="kpi-value">{avg_dep:.2f}%</div><div class="kpi-sub">Of population affected</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Highest Country Rate</div><div class="kpi-value">{max_dep:.2f}%</div><div class="kpi-sub">Maximum recorded</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Countries Analyzed</div><div class="kpi-value">{n_countries}</div><div class="kpi-sub">In selected period</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Change Over Period</div><div class="kpi-value">+{trend_change:.2f}%</div><div class="kpi-sub">Avg increase</div></div>', unsafe_allow_html=True)

# ── Chart 1 — Global trend ────────────────────────────────────
st.markdown('<div class="section-title">01 · The Rising Trend</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Depression rates have been <b>steadily climbing globally</b> since 1990. This isn't noise — it's a consistent upward trajectory
that demands attention from public health policy. The <span class="hl">peak year</span> reveals when global mental health
reached its most strained point.
</div>""", unsafe_allow_html=True)

global_trend = dff.groupby("Year")["Depression (%)"].mean().reset_index()
peak = global_trend.loc[global_trend["Depression (%)"].idxmax()]

fig1, ax1 = plt.subplots(figsize=(12, 4.5))
ax1.plot(global_trend["Year"], global_trend["Depression (%)"], color=ACCENT, linewidth=2.5, zorder=3)
ax1.fill_between(global_trend["Year"], global_trend["Depression (%)"], alpha=0.10, color=ACCENT)
ax1.annotate(f"Peak: {peak['Depression (%)']:.2f}%\n({int(peak['Year'])})",
    xy=(peak["Year"], peak["Depression (%)"]),
    xytext=(peak["Year"]-6, peak["Depression (%)"]+0.05),
    fontsize=9, color=ACCENT, fontweight="600",
    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
ax1.set_title("Global Average Depression Rate Over Time", fontsize=12, color=TEXT, pad=12, fontweight="bold")
ax1.set_xlabel("Year"); ax1.set_ylabel("Avg. Share of Population (%)")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax1.grid(axis="y")
for spine in ["top","right"]: ax1.spines[spine].set_visible(False)
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close()

# ── Chart 2 — Top countries ───────────────────────────────────
st.markdown('<div class="section-title">02 · Where the Burden Falls</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="section-story">
Not every country experiences this crisis equally. The <span class="hl">top country</span> in our analysis carries
a disproportionate share of the global burden — a clear signal of where targeted mental health investment is needed most.
</div>""", unsafe_allow_html=True)

latest_year = dff["Year"].max()
latest = dff[dff["Year"] == latest_year].copy()
top = latest.nlargest(top_n, "Depression (%)").sort_values("Depression (%)", ascending=True)

fig2, ax2 = plt.subplots(figsize=(11, max(4, top_n*0.4)))
bar_colors = [ACCENT if i == len(top)-1 else GRAY for i in range(len(top))]
bars = ax2.barh(top["Entity"], top["Depression (%)"], color=bar_colors, height=0.6, zorder=2)
for bar, val in zip(bars, top["Depression (%)"]):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9, color=TICK)
ax2.set_title(f"Top {top_n} Countries by Depression Rate ({latest_year})", fontsize=12, color=TEXT, pad=12, fontweight="bold")
ax2.set_xlabel("Share of Population (%)")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax2.set_xlim(0, top["Depression (%)"].max() * 1.2)
ax2.grid(axis="x")
for spine in ["top","right"]: ax2.spines[spine].set_visible(False)
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()

# ── Chart 3 — Multi-line ──────────────────────────────────────
st.markdown('<div class="section-title">03 · Country Trajectories</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
When we plot the trajectories of the highest-burden countries together, a pattern emerges:
<span class="hl">the leader keeps pulling away</span>. Countries that were already struggling in 1990 have seen their
situation worsen most dramatically.
</div>""", unsafe_allow_html=True)

top_countries = dff.groupby("Entity")["Depression (%)"].mean().nlargest(12).index.tolist()
highlight = top_countries[0]

fig3, ax3 = plt.subplots(figsize=(12, 5))
for country in top_countries:
    cdata = dff[dff["Entity"] == country].sort_values("Year")
    if country == highlight:
        ax3.plot(cdata["Year"], cdata["Depression (%)"], color=ACCENT, linewidth=2.8, zorder=3)
        ax3.text(cdata["Year"].max()+0.3, cdata["Depression (%)"].iloc[-1], country,
                 fontsize=8.5, color=ACCENT, va="center", fontweight="600")
    else:
        ax3.plot(cdata["Year"], cdata["Depression (%)"], color=GRAY, linewidth=1.1, alpha=0.85, zorder=2)
ax3.set_title(f"Trajectory Comparison — {highlight} Stands Out", fontsize=12, color=TEXT, pad=12, fontweight="bold")
ax3.set_xlabel("Year"); ax3.set_ylabel("Share of Population (%)")
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax3.grid(axis="y")
for spine in ["top","right"]: ax3.spines[spine].set_visible(False)
plt.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close()

# ── Chart 4 — Interactive scatter ─────────────────────────────
st.markdown('<div class="section-title">04 · Depression Meets Anxiety</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Mental health disorders rarely travel alone. This scatter plot reveals that countries with
<span class="hl">high depression rates often also have high anxiety rates</span> — they cluster together,
suggesting shared root causes that policy must address holistically.
</div>""", unsafe_allow_html=True)

latest2 = dff[dff["Year"] == dff["Year"].max()].dropna(subset=["Depression (%)","Anxiety disorders (%)"]).copy()
avg_d = latest2["Depression (%)"].mean()
latest2["hl"] = latest2["Depression (%)"] > avg_d

fig4 = go.Figure()
bg = latest2[~latest2["hl"]]
fig4.add_trace(go.Scatter(x=bg["Anxiety disorders (%)"], y=bg["Depression (%)"],
    mode="markers", name="Below avg",
    marker=dict(color=GRAY, size=8, opacity=0.7),
    text=bg["Entity"],
    hovertemplate="<b>%{text}</b><br>Anxiety: %{x:.2f}%<br>Depression: %{y:.2f}%<extra></extra>"))
hi = latest2[latest2["hl"]]
fig4.add_trace(go.Scatter(x=hi["Anxiety disorders (%)"], y=hi["Depression (%)"],
    mode="markers", name="Above avg depression",
    marker=dict(color=ACCENT, size=11, opacity=0.95, line=dict(color="#005f73", width=1)),
    text=hi["Entity"],
    hovertemplate="<b>%{text}</b><br>Anxiety: %{x:.2f}%<br>Depression: %{y:.2f}%<extra></extra>"))
fig4.add_hline(y=avg_d, line_dash="dot", line_color="#adb5bd",
               annotation_text=f"Avg depression {avg_d:.2f}%", annotation_font_color="#6c757d")
fig4.update_layout(
    title=dict(text=f"Depression vs Anxiety by Country ({dff['Year'].max()})", font=dict(size=12, color=TEXT)),
    plot_bgcolor=BG, paper_bgcolor=BG, font=dict(family="Inter", color=TICK),
    xaxis=dict(title="Anxiety Disorders (%)", showgrid=True, gridcolor=GRID, color=TICK),
    yaxis=dict(title="Depression (%)", showgrid=True, gridcolor=GRID, color=TICK),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TICK)), height=450
)
st.plotly_chart(fig4, use_container_width=True)

# ── Country deep dive ─────────────────────────────────────────
st.markdown('<div class="section-title">05 · Single Country Analysis</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-story">
Select any country below to see its depression trajectory in detail and benchmark it against the global average.
</div>""", unsafe_allow_html=True)

countries = sorted(dff["Entity"].unique().tolist())
default_idx = countries.index("United States") if "United States" in countries else 0
sel = st.selectbox("Select a country", countries, index=default_idx)

cd = dff[dff["Entity"] == sel].sort_values("Year")
global_avg = dff.groupby("Year")["Depression (%)"].mean().reset_index()

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=global_avg["Year"], y=global_avg["Depression (%)"],
    name="Global Average", mode="lines",
    line=dict(color=GRAY, width=2, dash="dot")))
fig5.add_trace(go.Scatter(x=cd["Year"], y=cd["Depression (%)"],
    name=sel, mode="lines", line=dict(color=ACCENT, width=2.8),
    fill="tozeroy", fillcolor="rgba(0,150,199,0.08)"))
fig5.update_layout(
    title=dict(text=f"{sel} — Depression Rate vs Global Average", font=dict(size=12, color=TEXT)),
    plot_bgcolor=BG, paper_bgcolor=BG, font=dict(family="Inter", color=TICK),
    xaxis=dict(showgrid=False, color=TICK),
    yaxis=dict(showgrid=True, gridcolor=GRID, color=TICK, tickformat=".2f", ticksuffix="%"),
    legend=dict(orientation="h", y=1.1, font=dict(color=TICK)), height=380
)
st.plotly_chart(fig5, use_container_width=True)

# ── Conclusion ────────────────────────────────────────────────
st.markdown('<div class="section-title">Conclusions</div>', unsafe_allow_html=True)
st.markdown("""<div class="story-block">
<b>What the data tells us.</b> Three findings stand out:
(1) <span class="hl">Depression is rising globally</span> — the trend line is unambiguous;
(2) <span class="hl">Burden is geographically concentrated</span> — some countries carry far more weight than others;
(3) <span class="hl">Depression and anxiety co-occur</span> — they share root causes and require integrated policy responses.<br><br>
<b>Recommendation.</b> Mental health funding must be reframed from a discretionary expense to a core public health priority,
with targeted investment in the countries showing the steepest deterioration.
</div>""", unsafe_allow_html=True)

st.markdown("<br><p style='font-size:0.65rem;color:#adb5bd;text-align:center;letter-spacing:2px;'>TEC DE MONTERREY · DATA VISUALIZATION COURSE · TEAM PROJECT · 2026</p>", unsafe_allow_html=True)
