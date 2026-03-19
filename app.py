"""
Power BI-Style Universal Analytics Dashboard — ULTRA EDITION v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW TABS & FEATURES ADDED:
  ┌─ Existing (enhanced) ──────────────────────────────────────────┐
  │  Time Series · Categories · Distributions · Correlations       │
  │  Advanced · Forecast · Segmentation · Anomaly · NL Query       │
  │  Data Cleaning · ML Report · Gemini AI                         │
  └────────────────────────────────────────────────────────────────┘
  ┌─ NEW TABS ──────────────────────────────────────────────────────┐
  │  📐 Statistical Deep Dive  — normality tests, skew, kurtosis,  │
  │     QQ-plot, ECDF, PDF fit, hypothesis testing                  │
  │  🧬 Feature Engineering   — create new columns on the fly,     │
  │     binning, log-transform, ratio cols, date-parts extraction   │
  │  🌐 Geo / Map Analysis    — choropleth, scatter-geo, bubble-map │
  │  📉 Cohort & Retention    — cohort heatmap, retention curves    │
  │  🔀 What-If Simulator     — Monte-Carlo simulation, scenario fan│
  │  📊 Comparative Analysis  — multi-metric bench, parallel coords │
  │  🧠 AutoML Lite           — train Ridge/RF/XGB, eval metrics,   │
  │     feature importance, SHAP-style bar, ROC/PR curves           │
  │  📋 Smart Report Builder  — drag-select charts → PDF/HTML report│
  └────────────────────────────────────────────────────────────────┘
  ENHANCED CHARTS (added to existing tabs):
    Lollipop · Step Chart · Area-Range · Density Contour
    Candlestick · OHLC · Hexbin · 3-D Scatter · Parallel Coords
    Sunburst · Icicle · Chord-like Sankey · Ridgeline (approx)
    Cumulative Distribution · QQ-Plot · Log-scale auto toggle
Run: streamlit run app.py
"""
import sys, os, io, json, warnings, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

def _load_env_key():
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key and "gemini_key" not in st.session_state:
        st.session_state["gemini_key"] = env_key
_load_env_key()

st.set_page_config(
    page_title="Analytics Dashboard Ultra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Ultra Dark Premium Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

/* ── KPI Cards ── */
.kpi-card{background:linear-gradient(135deg,#1A1D27 0%,#12141e 100%);
  border:1px solid rgba(255,255,255,0.07);border-radius:16px;
  padding:18px 20px;position:relative;overflow:hidden;margin-bottom:8px;
  transition:transform .2s,box-shadow .2s;}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,.4);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--accent,#FF6B35);border-radius:16px 16px 0 0;}
.kpi-card::after{content:'';position:absolute;top:-30px;right:-30px;width:80px;height:80px;
  border-radius:50%;background:var(--accent,#FF6B35);opacity:.04;}
.kpi-label{font-size:10px;color:#6B7280;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px;}
.kpi-value{font-size:24px;font-weight:700;color:#F9FAFB;line-height:1.1;font-family:'DM Mono',monospace;}
.kpi-sub{font-size:11px;margin-top:4px;color:#9CA3AF;}
.kpi-delta-pos{color:#34D399;font-size:11px;margin-top:4px;font-weight:600;}
.kpi-delta-neg{color:#EF4444;font-size:11px;margin-top:4px;font-weight:600;}
.kpi-icon{position:absolute;right:14px;top:14px;font-size:22px;opacity:.1;}

/* ── Insight & Section labels ── */
.sec-label{font-size:10px;font-weight:600;color:#6B7280;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:4px;margin-top:4px;}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,107,53,.25),transparent);margin:20px 0;}
.tab-header{font-size:20px;font-weight:700;color:#F9FAFB;margin-bottom:4px;font-family:'Syne',sans-serif;}
.tab-sub{font-size:13px;color:#6B7280;margin-bottom:16px;}
.insight-card{background:#1A1D27;border-left:3px solid var(--c,#FF6B35);
  border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;
  transition:border-width .15s;}
.insight-card:hover{border-left-width:5px;}
.insight-title{font-size:13px;font-weight:600;color:#F9FAFB;margin-bottom:3px;}
.insight-text{font-size:12px;color:#9CA3AF;}

/* ── ML Cards ── */
.ml-card{background:#1A1D27;border:1px solid rgba(255,255,255,0.07);
  border-radius:12px;padding:16px 18px;margin-bottom:10px;}
.ml-name{font-size:14px;font-weight:600;color:#F9FAFB;}
.ml-badge{display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:10px;font-weight:600;letter-spacing:.06em;margin-left:8px;}
.ml-desc{font-size:12px;color:#9CA3AF;margin-top:6px;}
.ml-why{font-size:11px;color:#6B7280;margin-top:4px;font-style:italic;}

/* ── Stat Table ── */
.stat-table{width:100%;border-collapse:collapse;font-size:12px;}
.stat-table td,.stat-table th{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.05);}
.stat-table th{color:#6B7280;font-size:10px;text-transform:uppercase;letter-spacing:.08em;}
.stat-table td{color:#E5E7EB;}
.stat-pass{color:#34D399;font-weight:600;}
.stat-fail{color:#EF4444;font-weight:600;}
.stat-warn{color:#FBBF24;font-weight:600;}

/* ── Feature Eng tags ── */
.feat-tag{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;
  font-weight:600;margin:2px;background:rgba(255,107,53,.12);color:#FF6B35;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:#080A12;border-right:1px solid rgba(255,255,255,0.04);}
#MainMenu,footer,header{visibility:hidden;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{gap:3px;background:rgba(255,255,255,0.025);
  border-radius:10px;padding:4px;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{border-radius:7px;padding:7px 12px;font-size:11.5px;
  font-weight:500;white-space:nowrap;}

/* ── Metric result card ── */
.result-card{background:#1A1D27;border:1px solid rgba(255,255,255,0.06);
  border-radius:12px;padding:16px 20px;margin:8px 0;}
.result-label{font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:.1em;}
.result-val{font-size:22px;font-weight:700;color:#F9FAFB;font-family:'DM Mono',monospace;}
.result-sub{font-size:11px;color:#9CA3AF;margin-top:2px;}

/* ── Progress bar override ── */
.stProgress > div > div {background:linear-gradient(90deg,#FF6B35,#2EC4B6);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════
ORANGE="#FF6B35"; TEAL="#2EC4B6"; DARK_BG="#0F1117"; CARD_BG="#1A1D27"
GRID="rgba(255,255,255,0.05)"; FONT="DM Sans"
PURPLE="#A78BFA"; GREEN="#34D399"; YELLOW="#FBBF24"; RED="#EF4444"; BLUE="#60A5FA"
PINK="#F472B6"; INDIGO="#818CF8"; LIME="#A3E635"; CYAN="#22D3EE"
PALETTE=[ORANGE,TEAL,PURPLE,GREEN,YELLOW,RED,BLUE,PINK,INDIGO,LIME,CYAN,"#FB923C","#E879F9"]

BASE=dict(font_family=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font_color="#C8CAD4", margin=dict(t=32,b=22,l=12,r=12),
          hoverlabel=dict(bgcolor=CARD_BG, font_color="white", font_family=FONT))
AX=dict(gridcolor=GRID, showline=False, zeroline=False, tickfont_size=11)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def fmt(v):
    try:
        v=float(v)
        if abs(v)>=1e9: return f"{v/1e9:.2f}B"
        if abs(v)>=1e6: return f"{v/1e6:.2f}M"
        if abs(v)>=1e3: return f"{v/1e3:.1f}K"
        return f"{v:,.2f}"
    except: return str(v)

def kpi(label, value, sub, icon, accent=ORANGE, delta=None):
    dcls=""
    if delta is not None:
        sign="▲" if delta>=0 else "▼"
        dcls=f'<div class="kpi-delta-{"pos" if delta>=0 else "neg"}">{sign} {abs(delta):.1f}%</div>'
    st.markdown(f"""<div class="kpi-card" style="--accent:{accent}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>{dcls}
    </div>""", unsafe_allow_html=True)

def sec(txt):
    st.markdown(f'<div class="sec-label">{txt}</div>', unsafe_allow_html=True)

def div():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def result_card(label, val, sub="", color=ORANGE):
    st.markdown(f"""<div class="result-card">
      <div class="result-label">{label}</div>
      <div class="result-val" style="color:{color}">{val}</div>
      <div class="result-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# COLUMN DETECTOR  (enhanced — smarter date parsing)
# ══════════════════════════════════════════════════════════════════
def detect_columns(df):
    date_col=None; num_cols=[]; cat_cols=[]
    for col in df.columns:
        cl=col.lower()
        if date_col is None:
            if df[col].dtype in ["datetime64[ns]","datetime64[ns, UTC]"]:
                date_col=col; continue
            if any(k in cl for k in ["date","time","day","month","year","dt","period","timestamp","week"]):
                try:
                    df[col]=pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                    date_col=col; continue
                except: pass
            if df[col].dtype==object:
                try:
                    p=pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if p.notna().mean()>0.7: df[col]=p; date_col=col; continue
                except: pass
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            nuniq=df[col].nunique()
            if nuniq<=max(50, len(df)*0.05) and nuniq>1:
                cat_cols.append(col)
    return {"date_col":date_col, "numeric_cols":num_cols, "category_cols":cat_cols}

# ══════════════════════════════════════════════════════════════════
# ─── CHART LIBRARY (original + NEW charts) ───────────────────────
# ══════════════════════════════════════════════════════════════════

# ── Trend / Time Series ───────────────────────────────────────────
def ch_trend(df,dc,mc,freq="auto"):
    tmp=df[[dc,mc]].dropna()
    if tmp.empty: return go.Figure()
    n=(tmp[dc].max()-tmp[dc].min()).days
    p="W" if n>60 else "D"
    if freq!="auto": p=freq
    grp=tmp.groupby(tmp[dc].dt.to_period(p).dt.start_time)[mc].sum().reset_index()
    grp.columns=["Period",mc]
    fig=go.Figure(go.Scatter(x=grp["Period"],y=grp[mc],mode="lines",
        line=dict(color=ORANGE,width=2.5),fill="tozeroy",
        fillcolor="rgba(255,107,53,0.08)",
        hovertemplate=f"<b>%{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,hovermode="x unified",xaxis=dict(**AX),yaxis=dict(**AX,title=mc))
    return fig

def ch_mom(df,dc,mc):
    tmp=df[[dc,mc]].dropna()
    m=tmp.groupby(tmp[dc].dt.to_period("M").dt.to_timestamp())[mc].sum()
    g=(m.pct_change()*100).dropna().reset_index(); g.columns=["Month","Growth"]
    if g.empty: return go.Figure()
    colors=[GREEN if v>=0 else RED for v in g["Growth"]]
    fig=go.Figure(go.Bar(x=g["Month"],y=g["Growth"],marker_color=colors,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=0,line_color="rgba(255,255,255,0.1)",line_width=1)
    fig.update_layout(**BASE,bargap=0.25,xaxis=dict(**AX),
                      yaxis=dict(**AX,ticksuffix="%",title="Growth %"))
    return fig

def ch_yoy(df,dc,mc):
    tmp=df[[dc,mc]].dropna().copy()
    tmp["Year"]=tmp[dc].dt.year; tmp["Month"]=tmp[dc].dt.month
    pivot=tmp.groupby(["Year","Month"])[mc].sum().reset_index()
    fig=go.Figure()
    for i,yr in enumerate(sorted(pivot["Year"].unique())):
        d=pivot[pivot["Year"]==yr]
        fig.add_trace(go.Scatter(x=d["Month"],y=d[mc],mode="lines+markers",name=str(yr),
            line=dict(color=PALETTE[i%len(PALETTE)],width=2),
            hovertemplate=f"<b>{yr} Month %{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,
        xaxis=dict(**AX,tickvals=list(range(1,13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
        yaxis=dict(**AX,title=mc),legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)"))
    return fig

def ch_heatmap(df,dc,mc):
    df2=df[[dc,mc]].dropna().copy()
    df2["DayOfWeek"]=df2[dc].dt.day_name()
    df2["Month"]=df2[dc].dt.strftime("%b %Y")
    pivot=df2.groupby(["DayOfWeek","Month"])[mc].sum().unstack(fill_value=0)
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot=pivot.reindex([d for d in days if d in pivot.index])
    if pivot.empty: return go.Figure()
    fig=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),
        colorscale=[[0,DARK_BG],[0.5,TEAL],[1,ORANGE]],showscale=True,
        hovertemplate=f"<b>%{{y}} · %{{x}}</b><br>{mc}: %{{z:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,height=280,
        xaxis=dict(tickfont_size=10,tickangle=-45,gridcolor=GRID),
        yaxis=dict(tickfont_size=11,gridcolor=GRID))
    return fig

def ch_waterfall(df,dc,mc):
    tmp=df[[dc,mc]].dropna()
    m=tmp.groupby(tmp[dc].dt.to_period("M").dt.to_timestamp())[mc].sum()
    vals=m.values; labels=[str(x.strftime("%b %Y")) for x in m.index]
    changes=np.diff(vals,prepend=0); changes[0]=vals[0]
    fig=go.Figure(go.Waterfall(x=labels,y=changes,measure=["relative"]*len(changes),
        increasing=dict(marker=dict(color=GREEN)),decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=TEAL)),
        connector=dict(line=dict(color="rgba(255,255,255,0.1)",width=1)),
        hovertemplate="<b>%{x}</b><br>Change: %{y:,.2f}<extra></extra>"))
    fig.update_layout(**BASE,height=320,xaxis=dict(**AX,tickangle=-30),yaxis=dict(**AX))
    return fig

def ch_rolling(df,dc,mc,window=7):
    tmp=df[[dc,mc]].dropna().sort_values(dc).copy()
    tmp["Roll"]=tmp[mc].rolling(window).mean()
    fig=go.Figure()
    fig.add_trace(go.Bar(x=tmp[dc],y=tmp[mc],
        marker_color="rgba(255,107,53,0.25)",name=mc,
        hovertemplate=f"<b>%{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tmp[dc],y=tmp["Roll"],mode="lines",
        line=dict(color=ORANGE,width=2.5),name=f"{window}-period MA"))
    fig.update_layout(**BASE,hovermode="x unified",bargap=0,
        xaxis=dict(**AX),yaxis=dict(**AX,title=mc),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

def ch_stacked(df,dc,cc,mc):
    tmp=df[[dc,cc,mc]].dropna()
    grp=tmp.groupby([tmp[dc].dt.to_period("M").dt.to_timestamp(),cc])[mc].sum().reset_index()
    grp.columns=["Period",cc,mc]
    fig=px.bar(grp,x="Period",y=mc,color=cc,color_discrete_sequence=PALETTE,barmode="stack")
    fig.update_layout(**BASE,bargap=0.1,xaxis=dict(**AX),yaxis=dict(**AX),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

# NEW ── Step Chart
def ch_step(df,dc,mc):
    tmp=df[[dc,mc]].dropna().sort_values(dc)
    grp=tmp.groupby(tmp[dc].dt.to_period("W").dt.start_time)[mc].sum().reset_index()
    grp.columns=["Period",mc]
    fig=go.Figure(go.Scatter(x=grp["Period"],y=grp[mc],mode="lines",
        line=dict(color=CYAN,width=2,shape="hv"),
        fill="tozeroy",fillcolor="rgba(34,211,238,0.06)",
        hovertemplate=f"<b>%{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX,title=mc),
                      title=dict(text="Step Chart (Weekly)",font_size=13))
    return fig

# NEW ── Area Range (min–max band)
def ch_area_range(df,dc,mc):
    tmp=df[[dc,mc]].dropna().sort_values(dc)
    grp=tmp.groupby(tmp[dc].dt.to_period("M").dt.to_timestamp())[mc].agg(["mean","min","max"]).reset_index()
    grp.columns=["Period","Mean","Min","Max"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=list(grp["Period"])+list(grp["Period"][::-1]),
        y=list(grp["Max"])+list(grp["Min"][::-1]),
        fill="toself",fillcolor="rgba(255,107,53,0.12)",
        line=dict(width=0),name="Min–Max Range",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grp["Period"],y=grp["Mean"],mode="lines",
        line=dict(color=ORANGE,width=2.5),name="Mean",
        hovertemplate="<b>%{x}</b><br>Mean: %{y:,.2f}<extra></extra>"))
    fig.update_layout(**BASE,hovermode="x unified",xaxis=dict(**AX),
                      yaxis=dict(**AX,title=mc),
                      legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

# NEW ── Candlestick (works on any numeric col grouped by month)
def ch_candlestick(df,dc,mc):
    tmp=df[[dc,mc]].dropna().sort_values(dc)
    grp=tmp.groupby(tmp[dc].dt.to_period("M").dt.to_timestamp())[mc].agg(
        ["first","max","min","last"]).reset_index()
    grp.columns=["Period","Open","High","Low","Close"]
    fig=go.Figure(go.Candlestick(
        x=grp["Period"],open=grp["Open"],high=grp["High"],
        low=grp["Low"],close=grp["Close"],
        increasing_line_color=GREEN,decreasing_line_color=RED))
    fig.update_layout(**BASE,xaxis=dict(**AX,rangeslider_visible=False),
                      yaxis=dict(**AX,title=mc))
    return fig

# ── Categories ────────────────────────────────────────────────────
def ch_donut(df,cc,mc):
    grp=df.groupby(cc)[mc].sum().reset_index()
    fig=go.Figure(go.Pie(labels=grp[cc],values=grp[mc],hole=0.55,
        marker=dict(colors=PALETTE[:len(grp)],line=dict(color=DARK_BG,width=2)),
        textinfo="percent",textfont=dict(size=11,family=FONT),
        hovertemplate=f"<b>%{{label}}</b><br>{mc}: %{{value:,.2f}}<extra></extra>"))
    fig.add_annotation(text=fmt(grp[mc].sum()),x=0.5,y=0.5,
        font=dict(size=16,color="white",family=FONT),showarrow=False)
    fig.update_layout(**BASE,showlegend=True,legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

def ch_bar_h(df,cc,mc,n=10,color=ORANGE):
    grp=df.groupby(cc)[mc].sum().nlargest(n).reset_index().sort_values(mc)
    fig=go.Figure(go.Bar(x=grp[mc],y=grp[cc],orientation="h",
        marker=dict(color=grp[mc],colorscale=[[0,CARD_BG],[1,color]],line=dict(width=0)),
        hovertemplate=f"<b>%{{y}}</b><br>{mc}: %{{x:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,bargap=0.2,xaxis=dict(**AX,title=mc),yaxis=dict(**AX))
    return fig

def ch_bar_v(df,cc,mc):
    grp=df.groupby(cc)[mc].sum().reset_index().sort_values(mc,ascending=False)
    fig=go.Figure(go.Bar(x=grp[cc],y=grp[mc],marker_color=PALETTE[:len(grp)],
        hovertemplate=f"<b>%{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,bargap=0.3,xaxis=dict(**AX),yaxis=dict(**AX,title=mc))
    return fig

def ch_funnel(df,cc,mc):
    grp=df.groupby(cc)[mc].sum().reset_index().sort_values(mc,ascending=False)
    fig=go.Figure(go.Funnel(y=grp[cc],x=grp[mc],marker=dict(color=PALETTE[:len(grp)]),
        textposition="inside",textinfo="value+percent initial",
        hovertemplate=f"<b>%{{y}}</b><br>{mc}: %{{x:,.2f}}<extra></extra>"))
    fig.update_layout(**BASE,height=360)
    return fig

def ch_treemap(df,cc,mc,cc2=None):
    try:
        path=[cc] if not cc2 else [cc,cc2]
        fig=px.treemap(df.dropna(subset=[mc]),path=path,values=mc,color=mc,
            color_continuous_scale=[[0,CARD_BG],[0.5,TEAL],[1,ORANGE]])
        fig.update_layout(**BASE,height=380,coloraxis_colorbar=dict(
            title=dict(text=mc,font=dict(size=10,color="#C8CAD4")),
            tickfont=dict(size=10,color="#C8CAD4"),bgcolor="rgba(0,0,0,0)",outlinewidth=0))
        return fig
    except: return go.Figure()

def ch_pareto(df,cc,mc):
    grp=df.groupby(cc)[mc].sum().sort_values(ascending=False).reset_index()
    grp["cum_pct"]=(grp[mc].cumsum()/grp[mc].sum()*100)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=grp[cc],y=grp[mc],marker_color=ORANGE,name=mc,
        hovertemplate=f"<b>%{{x}}</b><br>{mc}: %{{y:,.2f}}<extra></extra>"),secondary_y=False)
    fig.add_trace(go.Scatter(x=grp[cc],y=grp["cum_pct"],mode="lines+markers",
        line=dict(color=TEAL,width=2),name="Cumulative %",
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>"),secondary_y=True)
    fig.add_hline(y=80,line_dash="dash",line_color="rgba(255,255,255,0.2)",secondary_y=True)
    fig.update_layout(**BASE,height=320,bargap=0.15,legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(dict(**AX),secondary_y=False)
    fig.update_yaxes(dict(**AX,ticksuffix="%",title="Cumulative %"),secondary_y=True)
    return fig

# NEW ── Lollipop chart
def ch_lollipop(df,cc,mc,n=15):
    grp=df.groupby(cc)[mc].sum().nlargest(n).reset_index().sort_values(mc)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=grp[mc],y=grp[cc],mode="markers",
        marker=dict(color=ORANGE,size=10,line=dict(color="white",width=1)),
        hovertemplate=f"<b>%{{y}}</b><br>{mc}: %{{x:,.2f}}<extra></extra>",name=mc))
    for _,row in grp.iterrows():
        fig.add_shape(type="line",x0=0,x1=row[mc],y0=row[cc],y1=row[cc],
            line=dict(color="rgba(255,107,53,0.3)",width=2))
    fig.update_layout(**BASE,bargap=0.2,xaxis=dict(**AX,title=mc),yaxis=dict(**AX),height=max(300,n*28))
    return fig

# NEW ── Sunburst
def ch_sunburst(df,cc,mc,cc2=None):
    try:
        path=[cc] if not cc2 else [cc,cc2]
        fig=px.sunburst(df.dropna(subset=[mc]),path=path,values=mc,
            color=mc,color_continuous_scale=[[0,DARK_BG],[0.5,PURPLE],[1,ORANGE]])
        fig.update_layout(**BASE,height=420)
        return fig
    except: return go.Figure()

# NEW ── Icicle (reverse sunburst)
def ch_icicle(df,cc,mc,cc2=None):
    try:
        path=[cc] if not cc2 else [cc,cc2]
        fig=px.icicle(df.dropna(subset=[mc]),path=path,values=mc,
            color=mc,color_continuous_scale=[[0,DARK_BG],[0.5,TEAL],[1,ORANGE]])
        fig.update_layout(**BASE,height=420)
        return fig
    except: return go.Figure()

# ── Distributions ─────────────────────────────────────────────────
def ch_dist(df,mc):
    vals=df[mc].dropna()
    fig=go.Figure(go.Histogram(x=vals,nbinsx=50,marker_color=TEAL,marker_opacity=0.75,
        hovertemplate=f"{mc}: %{{x}}<br>Count: %{{y}}<extra></extra>"))
    fig.add_vline(x=vals.mean(),line_dash="dash",line_color=ORANGE,line_width=1.5,
        annotation_text=f"μ={vals.mean():,.2f}",annotation_font_color=ORANGE,annotation_font_size=10)
    fig.add_vline(x=vals.median(),line_dash="dot",line_color=TEAL,line_width=1.5,
        annotation_text=f"M={vals.median():,.2f}",annotation_font_color=TEAL,annotation_font_size=10)
    fig.update_layout(**BASE,height=270,xaxis=dict(**AX,title=mc),
        yaxis=dict(**AX,title="Count"),showlegend=False)
    return fig

def ch_box(df,mc,cc=None):
    if cc and cc in df.columns:
        fig=go.Figure()
        for i,c in enumerate(df[cc].dropna().unique()):
            vals=df[df[cc]==c][mc].dropna()
            fig.add_trace(go.Box(y=vals,name=str(c),marker_color=PALETTE[i%len(PALETTE)],
                line_color=PALETTE[i%len(PALETTE)],boxmean=True))
    else:
        fig=go.Figure(go.Box(y=df[mc].dropna(),marker_color=ORANGE,
                              line_color=ORANGE,name=mc,boxmean=True))
    fig.update_layout(**BASE,height=300,yaxis=dict(**AX,title=mc),
        showlegend=cc is not None,legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

def ch_violin(df,mc,cc=None):
    if cc and cc in df.columns:
        fig=go.Figure()
        for i,c in enumerate(df[cc].dropna().unique()):
            vals=df[df[cc]==c][mc].dropna()
            fig.add_trace(go.Violin(y=vals,name=str(c),fillcolor=PALETTE[i%len(PALETTE)],
                line_color=PALETTE[i%len(PALETTE)],opacity=0.7,box_visible=True,meanline_visible=True))
    else:
        fig=go.Figure(go.Violin(y=df[mc].dropna(),fillcolor=TEAL,line_color=TEAL,
            opacity=0.7,box_visible=True,meanline_visible=True))
    fig.update_layout(**BASE,height=300,yaxis=dict(**AX,title=mc),showlegend=cc is not None)
    return fig

# NEW ── ECDF
def ch_ecdf(df,mc):
    vals=df[mc].dropna().sort_values()
    ecdf=np.arange(1,len(vals)+1)/len(vals)
    fig=go.Figure(go.Scatter(x=vals,y=ecdf,mode="lines",line=dict(color=TEAL,width=2),
        fill="tozeroy",fillcolor="rgba(46,196,182,0.07)",
        hovertemplate=f"{mc}: %{{x:,.2f}}<br>P(X≤x): %{{y:.3f}}<extra></extra>"))
    fig.update_layout(**BASE,height=270,xaxis=dict(**AX,title=mc),
        yaxis=dict(**AX,title="Cumulative Probability",tickformat=".0%"))
    return fig

# NEW ── QQ Plot
def ch_qq(df,mc):
    vals=df[mc].dropna()
    if len(vals)<4: return go.Figure()
    osm,osr=scipy_stats.probplot(vals,dist="norm")
    theory,sample=osm
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=theory,y=sample,mode="markers",
        marker=dict(color=ORANGE,size=5,opacity=0.7),name="Data",
        hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"))
    mn,mx=theory[0],theory[-1]
    slope,intercept,_,_,_=scipy_stats.linregress(theory,sample)
    fig.add_trace(go.Scatter(x=[mn,mx],y=[slope*mn+intercept,slope*mx+intercept],
        mode="lines",line=dict(color=TEAL,width=1.5,dash="dash"),name="Normal line"))
    fig.update_layout(**BASE,height=290,xaxis=dict(**AX,title="Theoretical Quantiles"),
        yaxis=dict(**AX,title="Sample Quantiles"),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

# NEW ── Density Contour
def ch_density_contour(df,xc,yc):
    tmp=df[[xc,yc]].dropna()
    fig=px.density_contour(tmp,x=xc,y=yc,color_discrete_sequence=[ORANGE])
    fig.update_traces(contours_coloring="fill",
        colorscale=[[0,"rgba(0,0,0,0)"],[0.5,f"rgba(255,107,53,0.3)"],[1,ORANGE]])
    fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX))
    return fig

# NEW ── Hexbin (2D histogram)
def ch_hexbin(df,xc,yc):
    tmp=df[[xc,yc]].dropna()
    fig=go.Figure(go.Histogram2dContour(x=tmp[xc],y=tmp[yc],
        colorscale=[[0,DARK_BG],[0.3,PURPLE],[0.7,ORANGE],[1,"#fff"]],
        contours=dict(coloring="heatmap"),showscale=True,
        hovertemplate=f"{xc}: %{{x:.2f}}<br>{yc}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(**BASE,height=320,xaxis=dict(**AX,title=xc),yaxis=dict(**AX,title=yc))
    return fig

# ── Correlations ──────────────────────────────────────────────────
def ch_corr(df,cols):
    corr=df[cols].corr().round(2)
    fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
        colorscale=[[0,RED],[0.5,CARD_BG],[1,TEAL]],zmin=-1,zmax=1,
        text=corr.values,texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z}<extra></extra>"))
    fig.update_layout(**BASE,height=370,
        xaxis=dict(tickfont_size=10,tickangle=-30),yaxis=dict(tickfont_size=10))
    return fig

def ch_scatter(df,xc,yc,cc=None,sc=None):
    cols=[xc,yc]
    if cc and cc in df.columns: cols.append(cc)
    if sc and sc in df.columns and pd.api.types.is_numeric_dtype(df[sc]): cols.append(sc)
    tmp=df[list(dict.fromkeys(cols))].dropna(subset=[xc,yc])
    kw=dict(data_frame=tmp,x=xc,y=yc,opacity=0.65,color_discrete_sequence=PALETTE)
    if cc and cc in tmp.columns: kw["color"]=cc
    if sc and sc in tmp.columns and pd.api.types.is_numeric_dtype(tmp[sc]):
        kw["size"]=sc; kw["size_max"]=30
    fig=px.scatter(**kw)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

def ch_bubble(df,xc,yc,sc,cc=None):
    cols=[xc,yc,sc]
    if cc and cc in df.columns: cols.append(cc)
    tmp=df[cols].dropna()
    kw=dict(data_frame=tmp,x=xc,y=yc,size=sc,size_max=50,opacity=0.7,
            color_discrete_sequence=PALETTE)
    if cc and cc in tmp.columns: kw["color"]=cc
    fig=px.scatter(**kw)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

# NEW ── 3-D Scatter
def ch_scatter3d(df,xc,yc,zc,cc=None):
    cols=[xc,yc,zc];
    if cc and cc in df.columns: cols.append(cc)
    tmp=df[list(dict.fromkeys(cols))].dropna()
    kw=dict(data_frame=tmp,x=xc,y=yc,z=zc,opacity=0.7,color_discrete_sequence=PALETTE)
    if cc and cc in tmp.columns: kw["color"]=cc
    fig=px.scatter_3d(**kw)
    fig.update_traces(marker=dict(size=4,line=dict(width=0)))
    fig.update_layout(**BASE,height=500,scene=dict(
        xaxis=dict(backgroundcolor=DARK_BG,gridcolor=GRID,title=xc),
        yaxis=dict(backgroundcolor=DARK_BG,gridcolor=GRID,title=yc),
        zaxis=dict(backgroundcolor=DARK_BG,gridcolor=GRID,title=zc)),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    return fig

# NEW ── Parallel Coordinates
def ch_parallel(df,num_cols,cc=None):
    tmp=df[num_cols].dropna().head(2000)
    dims=[dict(range=[tmp[c].min(),tmp[c].max()],label=c,values=tmp[c]) for c in num_cols]
    color=tmp[num_cols[0]]
    fig=go.Figure(go.Parcoords(line=dict(color=color,
        colorscale=[[0,TEAL],[0.5,PURPLE],[1,ORANGE]],showscale=True),
        dimensions=dims))
    fig.update_layout(**BASE,height=400)
    return fig

# ── Radar / Sankey ────────────────────────────────────────────────
def ch_radar(df,cc,num_cols,mc):
    grp=df.groupby(cc)[num_cols].mean().reset_index()
    cats=grp[cc].tolist()
    metrics=[c for c in num_cols if c!=mc][:6]
    if not metrics: return go.Figure()
    normed=grp[metrics].copy()
    for col in metrics:
        mn,mx=normed[col].min(),normed[col].max()
        normed[col]=(normed[col]-mn)/(mx-mn+1e-9)
    fig=go.Figure()
    for i,row in normed.iterrows():
        vals=row[metrics].tolist(); vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(r=vals,theta=metrics+[metrics[0]],
            fill="toself",name=str(cats[i]),
            line=dict(color=PALETTE[i%len(PALETTE)],width=2),opacity=0.7))
    fig.update_layout(**BASE,polar=dict(
        bgcolor=CARD_BG,
        radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID,tickfont_size=9),
        angularaxis=dict(gridcolor=GRID,tickfont_size=10)),
        showlegend=True,legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"),height=400)
    return fig

def ch_sankey(df,source_col,target_col,mc):
    grp=df.groupby([source_col,target_col])[mc].sum().reset_index().nlargest(30,mc)
    all_nodes=list(pd.unique(grp[[source_col,target_col]].values.ravel("K")))
    node_idx={n:i for i,n in enumerate(all_nodes)}
    fig=go.Figure(go.Sankey(
        node=dict(label=all_nodes,color=PALETTE[:len(all_nodes)],
            pad=15,thickness=20,line=dict(color="rgba(0,0,0,0)",width=0)),
        link=dict(source=[node_idx[s] for s in grp[source_col]],
                  target=[node_idx[t] for t in grp[target_col]],
                  value=grp[mc].tolist(),color="rgba(255,107,53,0.22)")))
    fig.update_layout(**BASE,height=430,font_size=12)
    return fig

# ── Gauge / KPI charts ────────────────────────────────────────────
def ch_gauge(value,ref,label,color=ORANGE):
    fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=value,
        delta={"reference":ref,"relative":True,"valueformat":".1%"},
        title={"text":label,"font":{"size":12,"color":"#C8CAD4"}},
        number={"font":{"size":20,"color":"#F9FAFB"}},
        gauge={"axis":{"range":[0,ref*2],"tickfont":{"size":9,"color":"#6B7280"}},
               "bar":{"color":color},"bgcolor":CARD_BG,
               "steps":[{"range":[0,ref*0.5],"color":"rgba(239,68,68,0.15)"},
                        {"range":[ref*0.5,ref],"color":"rgba(251,191,36,0.15)"},
                        {"range":[ref,ref*2],"color":"rgba(52,211,153,0.15)"}],
               "threshold":{"line":{"color":"white","width":2},"value":ref}}))
    gl={k:v for k,v in BASE.items() if k!="margin"}
    fig.update_layout(**gl,height=220,margin=dict(t=40,b=10,l=30,r=30))
    return fig

# ── Forecast ──────────────────────────────────────────────────────
def ch_forecast(df,dc,mc,periods=30):
    tmp=df[[dc,mc]].dropna().sort_values(dc)
    tmp=tmp.groupby(dc)[mc].sum().reset_index()
    if len(tmp)<10: return go.Figure(),pd.DataFrame()
    x=np.arange(len(tmp)); y=tmp[mc].values
    coeffs=np.polyfit(x,y,1); trend=np.poly1d(coeffs)
    last_date=tmp[dc].max()
    freq=pd.infer_freq(tmp[dc]) or "D"
    try:
        future_dates=pd.date_range(start=last_date,periods=periods+1,freq=freq)[1:]
    except:
        future_dates=pd.date_range(start=last_date,periods=periods+1,freq="D")[1:]
    future_x=np.arange(len(tmp),len(tmp)+periods)
    forecast_vals=trend(future_x)
    residuals=y-trend(x); std=residuals.std()
    upper=forecast_vals+1.96*std; lower=forecast_vals-1.96*std
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=tmp[dc],y=y,mode="lines",name="Historical",
        line=dict(color=ORANGE,width=2),
        hovertemplate="<b>%{x}</b><br>Actual: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=future_dates,y=forecast_vals,mode="lines",name="Forecast",
        line=dict(color=TEAL,width=2,dash="dash"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates[::-1]),
        y=list(upper)+list(lower[::-1]),
        fill="toself",fillcolor="rgba(46,196,182,0.09)",
        line=dict(width=0),name="95% CI",hoverinfo="skip"))
    fig.update_layout(**BASE,hovermode="x unified",
        xaxis=dict(**AX),yaxis=dict(**AX,title=mc),
        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
    fcast_df=pd.DataFrame({"Date":future_dates,
        "Forecast":forecast_vals.round(2),
        "Lower":lower.round(2),"Upper":upper.round(2)})
    return fig,fcast_df

# ── ML helpers ────────────────────────────────────────────────────
def run_kmeans(df,num_cols,k=3):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    tmp=df[num_cols].dropna()
    if len(tmp)<k: return df.copy(),"Not enough rows"
    scaler=StandardScaler(); scaled=scaler.fit_transform(tmp)
    km=KMeans(n_clusters=k,random_state=42,n_init=10); labels=km.fit_predict(scaled)
    result=df.copy()
    result.loc[tmp.index,"Cluster"]=labels
    result["Cluster"]=result["Cluster"].fillna(-1).astype(int).astype(str)
    return result,"ok"

def run_anomaly(df,num_cols,contamination=0.05):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    tmp=df[num_cols].dropna()
    if len(tmp)<10: return df.copy(),"Not enough rows"
    scaler=StandardScaler(); scaled=scaler.fit_transform(tmp)
    iso=IsolationForest(contamination=contamination,random_state=42)
    preds=iso.fit_predict(scaled)
    result=df.copy()
    result.loc[tmp.index,"Anomaly"]=np.where(preds==-1,"Anomaly","Normal")
    result["Anomaly"]=result["Anomaly"].fillna("Normal")
    return result,"ok"

# ══════════════════════════════════════════════════════════════════
# AUTO INSIGHTS ENGINE
# ══════════════════════════════════════════════════════════════════
def auto_insights(df,cols_info,metric_col):
    insights=[]; num_cols=cols_info.get("numeric_cols",[]); cat_cols=cols_info.get("category_cols",[])
    date_col=cols_info.get("date_col"); mc=metric_col
    vals=df[mc].dropna()
    if len(vals):
        q75,q25=vals.quantile(0.75),vals.quantile(0.25); iqr=q75-q25
        outliers=((vals<q25-1.5*iqr)|(vals>q75+1.5*iqr)).sum()
        if outliers>0:
            insights.append({"icon":"⚠️","color":YELLOW,"title":f"{outliers} outliers in {mc}",
                "text":f"Values outside IQR [{q25-1.5*iqr:,.2f}–{q75+1.5*iqr:,.2f}]."})
        skew=float(vals.skew())
        if abs(skew)>1:
            d="right (positive)" if skew>0 else "left (negative)"
            insights.append({"icon":"📐","color":BLUE,"title":f"{mc} skewed {d}",
                "text":f"Skewness={skew:.2f}. Consider log-transform before ML."})
        cv=vals.std()/vals.mean()*100 if vals.mean()!=0 else 0
        if cv>50:
            insights.append({"icon":"📊","color":PINK,"title":f"High variability in {mc}",
                "text":f"Coefficient of variation = {cv:.1f}%. Data is highly dispersed."})
    if cat_cols:
        cc=cat_cols[0]; grp=df.groupby(cc)[mc].sum()
        top=grp.idxmax(); top_pct=grp.max()/grp.sum()*100
        insights.append({"icon":"🏆","color":GREEN,"title":f"Top {cc}: {top}",
            "text":f"Accounts for {top_pct:.1f}% of total {mc}."})
        if top_pct>50:
            insights.append({"icon":"🎯","color":ORANGE,"title":"Concentration risk",
                "text":f"Over 50% of {mc} from single {cc}."})
    if date_col:
        tmp=df[[date_col,mc]].dropna()
        if len(tmp)>1:
            monthly=tmp.groupby(tmp[date_col].dt.to_period("M"))[mc].sum()
            if len(monthly)>1:
                trend=np.polyfit(range(len(monthly)),monthly.values,1)[0]
                d="upward 📈" if trend>0 else "downward 📉"
                insights.append({"icon":"📅","color":TEAL if trend>0 else RED,
                    "title":f"Overall trend is {d}",
                    "text":f"Avg monthly change: {trend:+,.2f} in {mc}."})
    if len(num_cols)>=2:
        try:
            corr=df[num_cols].corr()[mc].drop(mc).abs()
            if not corr.empty:
                best=corr.idxmax()
                insights.append({"icon":"🔗","color":PURPLE,"title":f"Strongest correlation: {best}",
                    "text":f"|r|={corr.max():.2f} with {mc}."})
        except: pass
    null_pct=(df.isnull().sum()/len(df)*100); bad=null_pct[null_pct>20]
    if not bad.empty:
        insights.append({"icon":"🕳️","color":RED,"title":f"{len(bad)} cols >20% missing",
            "text":f"Cols: {', '.join(bad.index.tolist()[:3])}. Impute before modelling."})
    return insights

# ══════════════════════════════════════════════════════════════════
# ML RECOMMENDER
# ══════════════════════════════════════════════════════════════════
def recommend_ml(df,cols_info,metric_col):
    num_cols=cols_info.get("numeric_cols",[]); cat_cols=cols_info.get("category_cols",[])
    date_col=cols_info.get("date_col"); n_num=len(num_cols); n_cat=len(cat_cols)
    target=df[metric_col].dropna(); is_binary=target.nunique()==2; is_cont=target.nunique()>10
    recs=[]
    if date_col and is_cont:
        recs.append({"name":"Prophet / ARIMA","badge":"Time Series","badge_color":BLUE,
            "desc":"Forecast future values using historical time patterns.",
            "why":f"Date col ({date_col}) + continuous target = ideal for forecasting.",
            "libs":"from prophet import Prophet  |  from statsmodels.tsa.arima.model import ARIMA",
            "difficulty":"⭐⭐","accuracy":"High for seasonal data"})
        recs.append({"name":"XGBoost Regressor","badge":"Regression","badge_color":ORANGE,
            "desc":"Gradient boosted trees — handles mixed features excellently.",
            "why":"Best when you have date + categories + numeric columns.",
            "libs":"from xgboost import XGBRegressor","difficulty":"⭐⭐","accuracy":"Very High"})
    if is_cont and n_num>=2:
        recs.append({"name":"Random Forest","badge":"Regression","badge_color":GREEN,
            "desc":"Ensemble of trees. Robust with interpretable feature importances.",
            "why":f"{n_num} numeric cols — RF ranks which drive {metric_col} most.",
            "libs":"from sklearn.ensemble import RandomForestRegressor","difficulty":"⭐","accuracy":"High"})
        recs.append({"name":"Ridge Regression","badge":"Regression","badge_color":TEAL,
            "desc":"Fast, interpretable baseline.",
            "why":"Always start here before complex models.",
            "libs":"from sklearn.linear_model import Ridge","difficulty":"⭐","accuracy":"Medium"})
    if is_binary:
        recs.append({"name":"XGBoost Classifier","badge":"Classification","badge_color":ORANGE,
            "desc":"Best-in-class gradient boosting for classification.",
            "why":"Handles imbalanced classes and mixed features well.",
            "libs":"from xgboost import XGBClassifier","difficulty":"⭐⭐","accuracy":"Very High"})
    if n_num>=3:
        recs.append({"name":"K-Means Clustering","badge":"Clustering","badge_color":YELLOW,
            "desc":"Segment data into natural groups.",
            "why":f"{n_num} numeric cols — clustering reveals hidden segments.",
            "libs":"from sklearn.cluster import KMeans","difficulty":"⭐","accuracy":"N/A"})
        recs.append({"name":"Isolation Forest","badge":"Anomaly Detection","badge_color":RED,
            "desc":"Detect outliers automatically.",
            "why":"Useful for fraud detection, quality control.",
            "libs":"from sklearn.ensemble import IsolationForest","difficulty":"⭐","accuracy":"High"})
    if n_num>=2:
        recs.append({"name":"PCA + t-SNE","badge":"Dim Reduction","badge_color":TEAL,
            "desc":"Reduce dimensions for visualization or preprocessing.",
            "why":f"{n_num} numeric cols — PCA reveals structure.",
            "libs":"from sklearn.decomposition import PCA","difficulty":"⭐⭐","accuracy":"N/A"})
    return recs

# ══════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ══════════════════════════════════════════════════════════════════
def gen_excel(df,cols_info,metric_col):
    buf=io.BytesIO()
    num_cols=cols_info.get("numeric_cols",[]); cat_cols=cols_info.get("category_cols",[])
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="Raw Data",index=False)
        if num_cols: df[num_cols].describe().round(3).to_excel(w,sheet_name="Summary Stats")
        for cc in cat_cols[:3]:
            grp=df.groupby(cc)[metric_col].agg(["sum","mean","count","std"]).round(2)
            grp.columns=["Total","Average","Count","Std Dev"]
            grp.to_excel(w,sheet_name=f"By {cc}"[:31])
        info=[{"Column":c,"Type":str(df[c].dtype),"Non-null":int(df[c].notna().sum()),
               "Nulls":int(df[c].isna().sum()),"Unique":int(df[c].nunique())} for c in df.columns]
        pd.DataFrame(info).to_excel(w,sheet_name="Column Info",index=False)
    buf.seek(0); return buf.read()

def gen_csv(df): return df.to_csv(index=False).encode("utf-8")

def gen_json(df,cols_info,metric_col):
    num_cols=cols_info.get("numeric_cols",[])
    r={"summary":{"rows":len(df),"columns":len(df.columns),"metric":metric_col,
                  "total":float(df[metric_col].sum()),"mean":float(df[metric_col].mean())},
       "stats":df[num_cols].describe().round(3).to_dict() if num_cols else {},
       "sample":df.head(10).to_dict(orient="records")}
    return json.dumps(r,indent=2,default=str).encode("utf-8")

def gen_txt(df,cols_info,metric_col,insights,recs):
    lines=["ANALYTICS REPORT ULTRA v3","="*50,
           f"Rows: {len(df):,}  |  Columns: {len(df.columns)}",
           f"Primary Metric: {metric_col}",
           f"Total: {fmt(df[metric_col].sum())}",
           f"Average: {fmt(df[metric_col].mean())}",
           f"Max: {fmt(df[metric_col].max())}",
           f"Min: {fmt(df[metric_col].min())}","","AUTO INSIGHTS","-"*30]
    for ins in insights: lines.append(f"{ins['icon']} {ins['title']}: {ins['text']}")
    lines+=["","ML RECOMMENDATIONS","-"*30]
    for r in recs: lines.append(f"• {r['name']} [{r['badge']}] — {r['desc']}")
    return "\n".join(lines).encode("utf-8")

# ══════════════════════════════════════════════════════════════════
# GEMINI AI ENGINE
# ══════════════════════════════════════════════════════════════════
def init_gemini(api_key):
    if not GEMINI_AVAILABLE: return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def build_data_context(df,cols_info,metric_col,max_rows=30):
    num_cols=cols_info.get("numeric_cols",[]); cat_cols=cols_info.get("category_cols",[])
    date_col=cols_info.get("date_col","")
    stats=df[num_cols].describe().round(2).to_string() if num_cols else "N/A"
    sample=df.head(max_rows).to_string(index=False)
    return f"""DATASET SUMMARY
Rows: {len(df):,} | Columns: {len(df.columns)}
Primary metric: {metric_col}
Date column: {date_col or "None"}
Numeric columns: {num_cols}
Category columns: {cat_cols}

STATISTICS:
{stats}

SAMPLE DATA (first {min(max_rows,len(df))} rows):
{sample}""".strip()

def gemini_ask(model,prompt):
    try:
        resp=model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"

# ══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <span style="font-size:36px;">📊</span>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
         background:linear-gradient(90deg,#FF6B35,#2EC4B6);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      Analytics Dashboard Ultra
    </div>
    <div style="color:#6B7280;font-size:13px;margin-top:2px;">
      Connect · Explore · Statistical Analysis · Forecast · ML · Gemini AI
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
div()

# ══════════════════════════════════════════════════════════════════
# DATA SOURCE CONNECTORS
# ══════════════════════════════════════════════════════════════════
st.markdown("### 🔌 Connect Data Source")
src1,src2,src3,src4,src5=st.tabs(["📄 CSV","📊 Excel","🔷 JSON","🌐 URL","📋 Paste"])

if "df_raw" not in st.session_state:
    st.session_state.df_raw=None; st.session_state.source_name=""

with src1:
    st.markdown("Upload a **CSV** file.")
    f=st.file_uploader("CSV",type=["csv"],key="csv_up",label_visibility="collapsed")
    if f:
        try:
            st.session_state.df_raw=pd.read_csv(f); st.session_state.source_name=f.name
            st.success(f"✅ {f.name} — {len(st.session_state.df_raw):,} rows")
        except Exception as e: st.error(str(e))

with src2:
    st.markdown("Upload an **Excel** file.")
    f=st.file_uploader("Excel",type=["xlsx","xls"],key="xl_up",label_visibility="collapsed")
    if f:
        try:
            xf=pd.ExcelFile(f); sel=st.selectbox("Sheet",xf.sheet_names,key="xl_sheet")
            st.session_state.df_raw=pd.read_excel(f,sheet_name=sel)
            st.session_state.source_name=f"{f.name} [{sel}]"
            st.success(f"✅ Sheet: {sel} — {len(st.session_state.df_raw):,} rows")
        except Exception as e: st.error(str(e))

with src3:
    st.markdown("Upload a **JSON** file.")
    f=st.file_uploader("JSON",type=["json"],key="json_up",label_visibility="collapsed")
    if f:
        try:
            st.session_state.df_raw=pd.read_json(f); st.session_state.source_name=f.name
            st.success(f"✅ {f.name} — {len(st.session_state.df_raw):,} rows")
        except Exception as e: st.error(str(e))

with src4:
    st.markdown("Fetch from a **public URL**.")
    url=st.text_input("URL",placeholder="https://raw.githubusercontent.com/.../data.csv",key="url_in")
    if st.button("🔗 Fetch",key="fetch_btn") and url:
        try:
            st.session_state.df_raw=pd.read_csv(url)
            st.session_state.source_name=url.split("/")[-1]
            st.success(f"✅ {len(st.session_state.df_raw):,} rows fetched")
        except Exception as e: st.error(f"Could not fetch: {e}")

with src5:
    st.markdown("**Paste** CSV data (first row = headers).")
    pasted=st.text_area("Paste",height=120,
        placeholder="date,revenue,category\n2024-01-01,1200,Electronics",key="paste_area")
    if st.button("📥 Load",key="paste_btn") and pasted.strip():
        try:
            st.session_state.df_raw=pd.read_csv(io.StringIO(pasted))
            st.session_state.source_name="Pasted data"
            st.success(f"✅ {len(st.session_state.df_raw):,} rows loaded")
        except Exception as e: st.error(f"Parse error: {e}")

df_raw=st.session_state.df_raw; source_name=st.session_state.source_name

if df_raw is None:
    st.markdown("""<div style="background:#1A1D27;border:2px dashed rgba(255,107,53,0.3);
        border-radius:14px;padding:40px;text-align:center;margin-top:20px;">
      <div style="font-size:36px;margin-bottom:10px">☝️</div>
      <div style="font-size:17px;font-weight:600;color:#F9FAFB;margin-bottom:8px;">
        Select a data source above to begin</div>
      <div style="font-size:13px;color:#6B7280;">CSV · Excel · JSON · URL · Paste — all supported</div>
    </div>""",unsafe_allow_html=True)
    st.stop()

df_raw.columns=df_raw.columns.str.strip()
cols_info=detect_columns(df_raw)
num_cols_detected=cols_info.get("numeric_cols",[])
if not num_cols_detected:
    st.error("No numeric columns found. Please check your data.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown(f"**Source:** `{source_name}`")
    st.caption(f"{len(df_raw):,} rows · {len(df_raw.columns)} cols")
    if st.button("🗑️ Clear data",key="clear_btn"):
        for k in ["df_raw","source_name","df_seg","df_anom","df_eng"]:
            st.session_state.pop(k,None)
        st.rerun()
    st.markdown("---")
    st.markdown("### 🤖 Gemini AI")
    if not GEMINI_AVAILABLE:
        st.warning("Run: `pip install google-generativeai`")
    _current_key=st.session_state.get("gemini_key","")
    _typed_key=st.text_input("Gemini API key",type="password",value=_current_key,
        placeholder="AIza...  (or set in .env)",key="gemini_key_input",
        help="Paste key here OR add GEMINI_API_KEY=... to .env")
    if _typed_key.strip(): st.session_state["gemini_key"]=_typed_key.strip()
    if st.session_state.get("gemini_key",""):
        st.success("✅ Gemini ready")
    else:
        st.caption("No key — add to .env or paste above")
    st.markdown("---")
    st.markdown("### 🔧 Column Mapping")
    all_cols=df_raw.columns.tolist(); auto_date=cols_info.get("date_col")
    cat_opts=cols_info.get("category_cols",[])
    date_col=st.selectbox("📅 Date column",["(none)"]+all_cols,
        index=(all_cols.index(auto_date)+1) if auto_date and auto_date in all_cols else 0)
    date_col=None if date_col=="(none)" else date_col
    metric_col=st.selectbox("📊 Primary metric",num_cols_detected,index=0)
    cat_default=[c for c in cat_opts if c!=metric_col][:2]
    cat_cols=st.multiselect("🏷️ Category columns",
        [c for c in cat_opts if c!=metric_col],default=cat_default)
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    df=df_raw.copy()
    if date_col:
        try: df[date_col]=pd.to_datetime(df[date_col],infer_datetime_format=True,errors="coerce")
        except: pass
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        vd=df[date_col].dropna()
        if not vd.empty:
            mn,mx=vd.min().date(),vd.max().date()
            dr=st.date_input("Date range",value=(mn,mx),min_value=mn,max_value=mx)
            if isinstance(dr,(list,tuple)) and len(dr)==2:
                df=df[(df[date_col]>=pd.Timestamp(dr[0]))&(df[date_col]<=pd.Timestamp(dr[1]))]
    for cc in cat_cols:
        uq=sorted(df[cc].dropna().unique().tolist())
        if 2<=len(uq)<=30:
            sel=st.multiselect(cc,uq,default=uq)
            if sel: df=df[df[cc].isin(sel)]
    st.markdown("---")
    st.caption(f"Showing {len(df):,} / {len(df_raw):,} rows")

if df.empty:
    st.warning("No data after filters."); st.stop()

div()
st.success(f"✅ **{source_name}** · **{len(df):,} rows** · **{len(df.columns)} columns** · Metric: **{metric_col}**")
div()

# KPI Cards
kpi_metrics=[metric_col]+[c for c in num_cols_detected if c!=metric_col][:4]
kpi_icons=["💰","📦","📊","🔢","📐"]; kpi_accents=[ORANGE,TEAL,PURPLE,GREEN,YELLOW]
kcols=st.columns(len(kpi_metrics))
for i,km in enumerate(kpi_metrics):
    vals=df[km].dropna(); half=len(vals)//2
    prev=vals.iloc[:half].sum() if half>0 else None
    curr=vals.iloc[half:].sum() if half>0 else None
    delta=((curr-prev)/prev*100) if prev and prev!=0 else None
    with kcols[i]:
        kpi(label=km,value=fmt(vals.sum()),
            sub=f"Avg {fmt(vals.mean())} · σ {fmt(vals.std())}",
            icon=kpi_icons[i%5],accent=kpi_accents[i%5],delta=delta)
div()

# Gauges
sec("Performance gauges — filtered vs full dataset")
gcols=st.columns(min(len(kpi_metrics),4))
for i,km in enumerate(kpi_metrics[:4]):
    with gcols[i]:
        st.plotly_chart(ch_gauge(float(df[km].sum()),float(df_raw[km].sum()),km,kpi_accents[i%5]),
                        use_container_width=True,key=f"gauge_{i}")
div()

# Auto insights
insights=auto_insights(df,cols_info,metric_col)
if insights:
    st.markdown("### 💡 Auto Insights")
    icols=st.columns(min(len(insights),3))
    for i,ins in enumerate(insights):
        with icols[i%3]:
            st.markdown(f"""<div class="insight-card" style="--c:{ins['color']}">
              <div class="insight-title">{ins['icon']} {ins['title']}</div>
              <div class="insight-text">{ins['text']}</div></div>""",unsafe_allow_html=True)
    div()

# ══════════════════════════════════════════════════════════════════
# ALL TABS  (20 tabs total)
# ══════════════════════════════════════════════════════════════════
tabs=st.tabs([
    "📅 Time Series","🏷️ Categories","📊 Distributions","🔗 Correlations",
    "🧩 Advanced","🔮 Forecast","🎯 Segmentation","🚨 Anomaly",
    "💬 NL Query","🧹 Data Cleaning","🤖 ML · Report","✨ Gemini AI",
    # ── NEW TABS ────────────────────────────────────────────────
    "📐 Statistics","🧬 Feature Eng","🌐 Geo Map",
    "📉 Cohort","🔀 Monte-Carlo","📊 Comparative",
    "🧠 AutoML Lite","📋 Report Builder"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: TIME SERIES  (enhanced)
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="tab-header">Time Series Analysis</div>',unsafe_allow_html=True)
    if not date_col or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.info("No date column detected. Select one in the sidebar ⚙️.")
    else:
        freq=st.radio("Aggregation",["Auto","Daily","Weekly","Monthly","Quarterly"],
                      horizontal=True,key="ts_freq")
        fmap={"Auto":"auto","Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q"}
        c1,c2=st.columns([3,2])
        with c1:
            sec(f"{metric_col} over time")
            st.plotly_chart(ch_trend(df,date_col,metric_col,fmap[freq]),use_container_width=True,key="trend_chart")
        with c2:
            sec("Month-over-month growth %")
            st.plotly_chart(ch_mom(df,date_col,metric_col),use_container_width=True,key="mom_chart")
        sec("Year-over-year comparison")
        st.plotly_chart(ch_yoy(df,date_col,metric_col),use_container_width=True,key="yoy_chart")
        c3,c4=st.columns(2)
        with c3:
            sec("Day-of-week heatmap")
            st.plotly_chart(ch_heatmap(df,date_col,metric_col),use_container_width=True,key="heatmap_chart")
        with c4:
            win=st.slider("Rolling window",3,30,7,key="roll_win")
            sec(f"{win}-period rolling average")
            st.plotly_chart(ch_rolling(df,date_col,metric_col,win),use_container_width=True,key="rolling_chart")
        sec("Waterfall — monthly change")
        st.plotly_chart(ch_waterfall(df,date_col,metric_col),use_container_width=True,key="waterfall_chart")
        c5,c6=st.columns(2)
        with c5:
            sec("Step chart (weekly)")
            st.plotly_chart(ch_step(df,date_col,metric_col),use_container_width=True,key="step_chart")
        with c6:
            sec("Area range — min/mean/max band")
            st.plotly_chart(ch_area_range(df,date_col,metric_col),use_container_width=True,key="area_range_chart")
        sec("Candlestick (monthly OHLC)")
        st.plotly_chart(ch_candlestick(df,date_col,metric_col),use_container_width=True,key="candle_chart")
        if cat_cols:
            sec(f"Stacked by {cat_cols[0]}")
            st.plotly_chart(ch_stacked(df,date_col,cat_cols[0],metric_col),use_container_width=True,key="stacked_chart")

# ═══════════════════════════════════════════════════════════════
# TAB 2: CATEGORIES  (enhanced + Sunburst + Icicle + Lollipop)
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="tab-header">Category Breakdown</div>',unsafe_allow_html=True)
    if not cat_cols:
        st.info("No category columns selected. Choose some in the sidebar ⚙️.")
    else:
        for cc in cat_cols:
            cc_key=cc.replace(" ","_").replace("/","_")
            st.markdown(f"#### {metric_col} by **{cc}**")
            ca,cb,cc3=st.columns(3)
            with ca:
                sec("Donut")
                st.plotly_chart(ch_donut(df,cc,metric_col),use_container_width=True,key=f"donut_{cc_key}")
            with cb:
                sec("Top bar")
                st.plotly_chart(ch_bar_h(df,cc,metric_col,10),use_container_width=True,key=f"barh_{cc_key}")
            with cc3:
                sec("Lollipop")
                st.plotly_chart(ch_lollipop(df,cc,metric_col),use_container_width=True,key=f"lollipop_{cc_key}")
            c2a,c2b=st.columns(2)
            with c2a:
                sec("Funnel")
                st.plotly_chart(ch_funnel(df,cc,metric_col),use_container_width=True,key=f"funnel_{cc_key}")
            with c2b:
                sec("Pareto 80/20")
                st.plotly_chart(ch_pareto(df,cc,metric_col),use_container_width=True,key=f"pareto_{cc_key}")
            c3a,c3b=st.columns(2)
            cc2_opt=[c for c in cat_cols if c!=cc]
            with c3a:
                sec("Treemap")
                st.plotly_chart(ch_treemap(df,cc,metric_col,cc2_opt[0] if cc2_opt else None),
                    use_container_width=True,key=f"treemap_{cc_key}")
            with c3b:
                sec("Sunburst")
                st.plotly_chart(ch_sunburst(df,cc,metric_col,cc2_opt[0] if cc2_opt else None),
                    use_container_width=True,key=f"sunburst_{cc_key}")
            sec("Icicle chart")
            st.plotly_chart(ch_icicle(df,cc,metric_col,cc2_opt[0] if cc2_opt else None),
                use_container_width=True,key=f"icicle_{cc_key}")
            div()

        if len(cat_cols)>=2:
            div(); sec(f"Sankey — {cat_cols[0]} → {cat_cols[1]}")
            sk1=cat_cols[0].replace(" ","_"); sk2=cat_cols[1].replace(" ","_")
            st.plotly_chart(ch_sankey(df,cat_cols[0],cat_cols[1],metric_col),
                use_container_width=True,key=f"sankey_{sk1}_{sk2}")

        num_avail=[c for c in num_cols_detected if c in df.columns]
        if cat_cols and len(num_avail)>=3:
            div(); st.markdown("#### Radar / Spider Chart")
            radar_cat=st.selectbox("Category for radar",cat_cols,key="radar_cat")
            rc_key=radar_cat.replace(" ","_").replace("/","_")
            sec(f"Normalised metrics by {radar_cat}")
            st.plotly_chart(ch_radar(df,radar_cat,num_avail,metric_col),
                use_container_width=True,key=f"radar_{rc_key}")

# ═══════════════════════════════════════════════════════════════
# TAB 3: DISTRIBUTIONS  (enhanced + ECDF + QQ)
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="tab-header">Distribution Analysis</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    if not num_avail:
        st.info("No numeric columns available.")
    else:
        for nc in num_avail[:5]:
            st.markdown(f"#### **{nc}**")
            d1,d2,d3=st.columns(3)
            with d1:
                sec("Histogram")
                st.plotly_chart(ch_dist(df,nc),use_container_width=True,key=f"dist_{nc.replace(' ','_')}")
            with d2:
                sec("Box plot")
                st.plotly_chart(ch_box(df,nc,cat_cols[0] if cat_cols else None),
                    use_container_width=True,key=f"box_{nc.replace(' ','_')}")
            with d3:
                sec("Violin")
                st.plotly_chart(ch_violin(df,nc,cat_cols[0] if cat_cols else None),
                    use_container_width=True,key=f"violin_{nc.replace(' ','_')}")
            d4,d5=st.columns(2)
            with d4:
                sec("ECDF (Empirical Cumulative Distribution)")
                st.plotly_chart(ch_ecdf(df,nc),use_container_width=True,key=f"ecdf_{nc.replace(' ','_')}")
            with d5:
                sec("QQ Plot (Normality check)")
                st.plotly_chart(ch_qq(df,nc),use_container_width=True,key=f"qq_{nc.replace(' ','_')}")
            div()

# ═══════════════════════════════════════════════════════════════
# TAB 4: CORRELATIONS  (enhanced + 3D + Hexbin + Parallel Coords)
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="tab-header">Correlation & Relationships</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    if len(num_avail)<2:
        st.info("Need at least 2 numeric columns.")
    else:
        sec("Correlation matrix")
        st.plotly_chart(ch_corr(df,num_avail),use_container_width=True,key="corr_matrix")
        others=[c for c in num_avail if c!=metric_col]
        if others:
            st.markdown("#### Scatter Plots")
            for i in range(0,min(len(others),4),2):
                pair=others[i:i+2]; scols=st.columns(len(pair))
                for j,oc in enumerate(pair):
                    with scols[j]:
                        sec(f"{metric_col} vs {oc}")
                        st.plotly_chart(ch_scatter(df,metric_col,oc,cat_cols[0] if cat_cols else None),
                            use_container_width=True,key=f"scatter_{oc.replace(' ','_')}")
            if len(others)>=1:
                div(); st.markdown("#### Density Contour & Hexbin")
                hx1,hx2=st.columns(2)
                with hx1:
                    sec(f"Density Contour: {metric_col} vs {others[0]}")
                    st.plotly_chart(ch_density_contour(df,metric_col,others[0]),
                        use_container_width=True,key="density_cont")
                with hx2:
                    sec(f"Hexbin: {metric_col} vs {others[0]}")
                    st.plotly_chart(ch_hexbin(df,metric_col,others[0]),
                        use_container_width=True,key="hexbin_chart")
            if len(num_avail)>=3:
                div(); st.markdown("#### 3-D Scatter")
                s3d_cols=st.multiselect("Pick 3 columns for 3-D scatter",num_avail,
                    default=num_avail[:3],key="s3d_cols",max_selections=3)
                if len(s3d_cols)==3:
                    st.plotly_chart(ch_scatter3d(df,s3d_cols[0],s3d_cols[1],s3d_cols[2],
                        cat_cols[0] if cat_cols else None),use_container_width=True,key="scatter3d")
            if len(num_avail)>=3:
                div(); sec("Parallel Coordinates (all numeric cols)")
                st.plotly_chart(ch_parallel(df,num_avail[:8],cat_cols[0] if cat_cols else None),
                    use_container_width=True,key="parallel_coords")
            if len(num_avail)>=3 and others:
                div(); st.markdown("#### Bubble Chart")
                sc_col=others[1] if len(others)>1 else others[0]
                sec(f"{metric_col} vs {others[0]} (size={sc_col})")
                st.plotly_chart(ch_bubble(df,metric_col,others[0],sc_col,cat_cols[0] if cat_cols else None),
                    use_container_width=True,key="bubble_chart")

# ═══════════════════════════════════════════════════════════════
# TAB 5: ADVANCED  (original + What-If)
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="tab-header">Advanced Analysis</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]

    if cat_cols:
        st.markdown("#### 🗂️ Pivot Table")
        row_col=st.selectbox("Row",cat_cols,key="piv_row")
        val_col=st.selectbox("Value",num_avail,key="piv_val")
        agg_fn=st.selectbox("Aggregation",["sum","mean","count","max","min"],key="piv_agg")
        col_col_opts=["(none)"]+[c for c in cat_cols if c!=row_col]
        col_col=st.selectbox("Column dim",col_col_opts,key="piv_col")
        col_col=None if col_col=="(none)" else col_col
        try:
            pivot=pd.pivot_table(df,values=val_col,index=row_col,columns=col_col,
                aggfunc=agg_fn,fill_value=0).round(2)
            st.dataframe(pivot,use_container_width=True)
        except Exception as e: st.error(str(e))
        div()

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.markdown("#### 📈 Cumulative Growth")
        tmp=df[[date_col,metric_col]].dropna().sort_values(date_col).copy()
        tmp["Cumulative"]=tmp[metric_col].cumsum()
        fig=go.Figure(go.Scatter(x=tmp[date_col],y=tmp["Cumulative"],mode="lines",
            line=dict(color=GREEN,width=2.5),fill="tozeroy",
            fillcolor="rgba(52,211,153,0.08)",
            hovertemplate=f"<b>%{{x}}</b><br>Cumulative: %{{y:,.2f}}<extra></extra>"))
        fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX,title=f"Cumulative {metric_col}"))
        st.plotly_chart(fig,use_container_width=True,key="piv_cum_chart")
        div()

    st.markdown("#### 📊 Percentile Breakdown")
    pct_col=st.selectbox("Column",num_avail,key="pct_col")
    vals=df[pct_col].dropna()
    pcts=[5,10,25,50,75,90,95,99]
    pct_df=pd.DataFrame({"Percentile":[f"P{p}" for p in pcts],
        "Value":[round(float(vals.quantile(p/100)),3) for p in pcts]})
    p1,p2=st.columns(2)
    with p1: st.dataframe(pct_df,use_container_width=True,hide_index=True)
    with p2:
        fig=go.Figure(go.Bar(x=pct_df["Percentile"],y=pct_df["Value"],
            marker_color=PALETTE[:len(pct_df)],
            hovertemplate="<b>%{x}</b><br>%{y:,.3f}<extra></extra>"))
        fig.update_layout(**BASE,height=260,xaxis=dict(**AX),yaxis=dict(**AX))
        st.plotly_chart(fig,use_container_width=True,key="pct_bar_chart")
    div()

    st.markdown("#### 🔍 Z-Score Outlier Detection")
    out_col=st.selectbox("Column",num_avail,key="out_col")
    z_thresh=st.slider("Z-score threshold",1.5,4.0,2.5,0.1,key="z_thr")
    tmp2=df[[out_col]].dropna().copy()
    tmp2["z"]=(tmp2[out_col]-tmp2[out_col].mean())/tmp2[out_col].std()
    tmp2["is_out"]=tmp2["z"].abs()>z_thresh
    n_out=int(tmp2["is_out"].sum())
    st.info(f"**{n_out} outliers** (|z|>{z_thresh}) out of {len(tmp2):,} rows")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=tmp2[~tmp2["is_out"]].index,y=tmp2[~tmp2["is_out"]][out_col],
        mode="markers",name="Normal",marker=dict(color=TEAL,size=4,opacity=0.6)))
    fig.add_trace(go.Scatter(x=tmp2[tmp2["is_out"]].index,y=tmp2[tmp2["is_out"]][out_col],
        mode="markers",name="Outlier",marker=dict(color=RED,size=8,symbol="x")))
    fig.update_layout(**BASE,height=280,xaxis=dict(**AX),yaxis=dict(**AX,title=out_col),
        legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig,use_container_width=True,key="zscore_chart")

    div()
    st.markdown("#### 🧪 What-If Scenario Builder")
    st.caption("Adjust numeric columns and see estimated impact on primary metric.")
    num_avail2=[c for c in num_avail if c!=metric_col]
    if len(num_avail2)>=1:
        baseline_total=float(df[metric_col].sum())
        scenario_vals={}
        wcols=st.columns(min(len(num_avail2),3))
        for i,nc in enumerate(num_avail2[:3]):
            with wcols[i]:
                pct_change=st.slider(f"{nc} change %",-50,100,0,5,key=f"wif_{nc}")
                scenario_vals[nc]=pct_change
        try:
            corr_with_target=df[num_avail].corr()[metric_col]
            total_effect=sum(
                scenario_vals[nc]/100*abs(float(corr_with_target.get(nc,0)))*baseline_total
                for nc in scenario_vals)
            estimated=baseline_total+total_effect
            delta_pct=(total_effect/baseline_total*100) if baseline_total!=0 else 0
        except: estimated=baseline_total; delta_pct=0
        w1,w2,w3=st.columns(3)
        with w1: kpi("Baseline",fmt(baseline_total),"Current total","📊",TEAL)
        with w2: kpi("Estimated",fmt(estimated),"After scenario","🔮",GREEN if estimated>baseline_total else RED)
        with w3: kpi("Change",f"{delta_pct:+.1f}%","Projected delta","📈",GREEN if delta_pct>=0 else RED)

# ═══════════════════════════════════════════════════════════════
# TAB 6: FORECAST
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="tab-header">Forecasting</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Linear trend + 95% confidence interval forecast.</div>',unsafe_allow_html=True)
    if not date_col or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.info("Forecasting requires a date column. Select one in the sidebar ⚙️.")
    else:
        periods=st.slider("Forecast periods ahead",7,180,30,key="fcast_periods")
        fcast_metric=st.selectbox("Metric to forecast",
            [metric_col]+[c for c in num_cols_detected if c!=metric_col],key="fcast_met")
        fig_f,fcast_df=ch_forecast(df,date_col,fcast_metric,periods)
        if fcast_df.empty:
            st.warning("Not enough data points (need ≥ 10).")
        else:
            sec(f"{fcast_metric} — {periods}-period forecast with 95% CI")
            st.plotly_chart(fig_f,use_container_width=True,key="forecast_chart")
            div()
            f1,f2,f3=st.columns(3)
            with f1: kpi("Forecast end value",fmt(float(fcast_df["Forecast"].iloc[-1])),"Predicted last period","🔮",TEAL)
            with f2: kpi("Forecast total",fmt(float(fcast_df["Forecast"].sum())),f"Over {periods} periods","📈",GREEN)
            with f3: kpi("Avg CI width",fmt(float((fcast_df["Upper"]-fcast_df["Lower"]).mean())),"Uncertainty range","📏",YELLOW)
            div()
            st.markdown("#### Forecast table")
            st.dataframe(fcast_df.round(2),use_container_width=True,height=280)
            st.download_button("⬇️ Download forecast CSV",
                data=fcast_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast.csv",mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# TAB 7: SEGMENTATION
# ═══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="tab-header">Customer / Data Segmentation</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">K-Means clustering — find natural groups in your data.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    if len(num_avail)<2:
        st.info("Need at least 2 numeric columns.")
    else:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            sklearn_ok=True
        except:
            sklearn_ok=False
            st.warning("Install scikit-learn: `pip install scikit-learn`")

        if sklearn_ok:
            k=st.slider("Number of segments (K)",2,8,3,key="km_k")
            seg_cols=st.multiselect("Columns to segment on",num_avail,default=num_avail[:3])
            if len(seg_cols)>=2 and st.button("🎯 Run Segmentation",key="km_run"):
                with st.spinner("Running K-Means..."):
                    df_seg,status=run_kmeans(df,seg_cols,k)
                if status=="ok":
                    st.session_state["df_seg"]=df_seg
                    st.success(f"✅ {k} segments found!")
                else: st.error(status)

            if "df_seg" in st.session_state:
                df_seg=st.session_state["df_seg"]
                seg_counts=df_seg["Cluster"].value_counts().sort_index()
                s1,s2=st.columns(2)
                with s1:
                    sec("Segment sizes")
                    fig=go.Figure(go.Bar(
                        x=["Seg "+str(c) for c in seg_counts.index],
                        y=seg_counts.values,marker_color=PALETTE[:len(seg_counts)],
                        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"))
                    fig.update_layout(**BASE,height=300,xaxis=dict(**AX),yaxis=dict(**AX))
                    st.plotly_chart(fig,use_container_width=True,key="seg_size_chart")
                with s2:
                    if metric_col in df_seg.columns:
                        sec(f"Segment mean {metric_col}")
                        seg_mean=df_seg.groupby("Cluster")[metric_col].mean().sort_index()
                        fig=go.Figure(go.Bar(
                            x=["Seg "+str(c) for c in seg_mean.index],
                            y=seg_mean.values,marker_color=PALETTE[:len(seg_mean)],
                            hovertemplate="<b>%{x}</b><br>Mean: %{y:,.2f}<extra></extra>"))
                        fig.update_layout(**BASE,height=300,xaxis=dict(**AX),yaxis=dict(**AX))
                        st.plotly_chart(fig,use_container_width=True,key="seg_mean_chart")
                if len(seg_cols)>=2:
                    sec(f"Scatter: {seg_cols[0]} vs {seg_cols[1]}")
                    fig=px.scatter(df_seg.dropna(subset=seg_cols[:2]),
                        x=seg_cols[0],y=seg_cols[1],color="Cluster",
                        color_discrete_sequence=PALETTE,opacity=0.7)
                    fig.update_traces(marker=dict(size=7,line=dict(width=0)))
                    fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX),
                        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig,use_container_width=True,key="seg_scatter_chart")
                div(); sec("Segment profile (mean values)")
                profile=df_seg.groupby("Cluster")[num_avail].mean().round(2)
                st.dataframe(profile,use_container_width=True)
                st.download_button("⬇️ Download segmented data",
                    data=df_seg.to_csv(index=False).encode("utf-8"),
                    file_name="segmented_data.csv",mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# TAB 8: ANOMALY
# ═══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="tab-header">Anomaly Detection</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Isolation Forest detects unusual records.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    try:
        from sklearn.ensemble import IsolationForest
        iso_ok=True
    except:
        iso_ok=False
        st.warning("Install scikit-learn: `pip install scikit-learn`")

    if iso_ok and num_avail:
        cont=st.slider("Contamination (expected anomaly %)",1,20,5,key="iso_cont")/100
        anom_cols=st.multiselect("Columns to analyse",num_avail,default=num_avail[:3])
        if len(anom_cols)>=1 and st.button("🚨 Detect Anomalies",key="iso_run"):
            with st.spinner("Running Isolation Forest..."):
                df_anom,status=run_anomaly(df,anom_cols,cont)
            if status=="ok":
                st.session_state["df_anom"]=df_anom
                n_anom=(df_anom["Anomaly"]=="Anomaly").sum()
                st.success(f"✅ {n_anom} anomalies ({n_anom/len(df_anom)*100:.1f}%)")
            else: st.error(status)

        if "df_anom" in st.session_state:
            df_anom=st.session_state["df_anom"]
            n_anom=(df_anom["Anomaly"]=="Anomaly").sum()
            n_norm=(df_anom["Anomaly"]=="Normal").sum()
            a1,a2,a3=st.columns(3)
            with a1: kpi("Total rows",f"{len(df_anom):,}","Analysed","📊",TEAL)
            with a2: kpi("Anomalies",f"{n_anom:,}",f"{n_anom/len(df_anom)*100:.1f}% flagged","🚨",RED)
            with a3: kpi("Normal",f"{n_norm:,}","Clean records","✅",GREEN)
            div()
            if len(anom_cols)>=2:
                sec(f"Scatter: {anom_cols[0]} vs {anom_cols[1]}")
                fig=px.scatter(df_anom.dropna(subset=anom_cols[:2]),
                    x=anom_cols[0],y=anom_cols[1],color="Anomaly",
                    color_discrete_map={"Normal":TEAL,"Anomaly":RED},opacity=0.7)
                fig.update_traces(marker=dict(size=6,line=dict(width=0)))
                fig.update_layout(**BASE,xaxis=dict(**AX),yaxis=dict(**AX),
                    legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig,use_container_width=True,key="anom_scatter_chart")
            div(); st.markdown("#### Anomalous records")
            anom_rows=df_anom[df_anom["Anomaly"]=="Anomaly"].reset_index(drop=True)
            st.dataframe(anom_rows,use_container_width=True,height=300)
            st.download_button("⬇️ Download anomalies",
                data=anom_rows.to_csv(index=False).encode("utf-8"),
                file_name="anomalies.csv",mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# TAB 9: NL QUERY
# ═══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="tab-header">Natural Language Query</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Ask questions in plain English — rule-based pandas engine.</div>',unsafe_allow_html=True)
    examples=["What is the total?","Top 5 by revenue","Show top 10 rows",
              "How many unique categories?","What is the average?","Rows above 1000",
              "Which has the highest sales?","How many missing values?"]
    ex_cols=st.columns(4)
    for i,ex in enumerate(examples):
        with ex_cols[i%4]:
            if st.button(ex,key=f"ex_{i}",use_container_width=True):
                st.session_state["nl_q"]=ex

    nl_q=st.text_input("Ask a question about your data",
        value=st.session_state.get("nl_q",""),
        placeholder="e.g. What is the total revenue by region?",key="nl_input")

    if nl_q:
        q=nl_q.lower().strip(); result_txt=""; result_df=None
        try:
            if any(w in q for w in ["total","sum"]) and metric_col:
                matched_cat=next((cc for cc in cat_cols if cc.lower() in q),None)
                if matched_cat:
                    grp=df.groupby(matched_cat)[metric_col].sum().sort_values(ascending=False).reset_index()
                    result_txt=f"Total **{metric_col}** by **{matched_cat}**:"; result_df=grp
                else:
                    result_txt=f"Total **{metric_col}** = **{fmt(df[metric_col].sum())}**"
            elif any(w in q for w in ["average","mean","avg"]):
                matched_cat=next((cc for cc in cat_cols if cc.lower() in q),None)
                if matched_cat:
                    grp=df.groupby(matched_cat)[metric_col].mean().sort_values(ascending=False).reset_index()
                    result_txt=f"Average **{metric_col}** by **{matched_cat}**:"; result_df=grp
                else:
                    result_txt=f"Average **{metric_col}** = **{fmt(df[metric_col].mean())}**"
            elif any(w in q for w in ["highest","top","maximum","max","best","largest"]):
                n=5; nums=re.findall(r'\d+',q)
                if nums: n=int(nums[0])
                matched_cat=next((cc for cc in cat_cols if cc.lower() in q),None)
                if matched_cat:
                    grp=df.groupby(matched_cat)[metric_col].sum().nlargest(n).reset_index()
                    result_txt=f"Top {n} **{matched_cat}** by **{metric_col}**:"; result_df=grp
                else:
                    result_txt=f"Top {n} rows:"; result_df=df.nlargest(n,metric_col).reset_index(drop=True)
            elif any(w in q for w in ["lowest","bottom","minimum","min","worst"]):
                n=5; nums=re.findall(r'\d+',q)
                if nums: n=int(nums[0])
                matched_cat=next((cc for cc in cat_cols if cc.lower() in q),None)
                if matched_cat:
                    grp=df.groupby(matched_cat)[metric_col].sum().nsmallest(n).reset_index()
                    result_txt=f"Bottom {n} **{matched_cat}**:"; result_df=grp
                else:
                    result_txt=f"Bottom {n} rows:"; result_df=df.nsmallest(n,metric_col).reset_index(drop=True)
            elif any(w in q for w in ["unique","distinct","how many"]):
                cc_match=next((cc for cc in cat_cols if cc.lower() in q),None)
                if cc_match:
                    result_txt=f"**{df[cc_match].nunique()}** unique **{cc_match}** values."
                else:
                    result_txt=f"Dataset: **{len(df):,}** rows · **{df.shape[1]}** columns"
            elif any(w in q for w in ["show","display","rows"]):
                n=10; nums=re.findall(r'\d+',q)
                if nums: n=int(nums[0])
                result_txt=f"First {n} rows:"; result_df=df.head(n).reset_index(drop=True)
            elif any(w in q for w in ["above","greater","over","more than"]):
                nums=re.findall(r'[\d,]+\.?\d*',q)
                if nums:
                    thr=float(nums[0].replace(",",""))
                    result_df=df[df[metric_col]>thr].reset_index(drop=True)
                    result_txt=f"**{len(result_df)}** rows where {metric_col} > {fmt(thr)}:"
                else: result_txt="Please specify a value."
            elif any(w in q for w in ["below","less than","under"]):
                nums=re.findall(r'[\d,]+\.?\d*',q)
                if nums:
                    thr=float(nums[0].replace(",",""))
                    result_df=df[df[metric_col]<thr].reset_index(drop=True)
                    result_txt=f"**{len(result_df)}** rows where {metric_col} < {fmt(thr)}:"
                else: result_txt="Please specify a value."
            elif any(w in q for w in ["missing","null","empty","nan"]):
                null_info=df.isnull().sum(); null_info=null_info[null_info>0]
                if null_info.empty: result_txt="✅ No missing values!"
                else:
                    result_txt="Columns with missing values:"
                    result_df=null_info.reset_index(); result_df.columns=["Column","Missing Count"]
            elif any(w in q for w in ["std","standard deviation","variance"]):
                result_txt=f"Std Dev of **{metric_col}** = **{fmt(df[metric_col].std())}**  |  Variance = **{fmt(df[metric_col].var())}**"
            elif "median" in q:
                result_txt=f"Median **{metric_col}** = **{fmt(df[metric_col].median())}**"
            elif "describe" in q or "summary" in q or "stats" in q:
                result_df=df[[metric_col]].describe().round(3).reset_index()
                result_txt=f"Statistical summary of **{metric_col}**:"
            else:
                result_txt="Try: 'total', 'average', 'top 10', 'show rows', 'median', 'std', 'missing'."
        except Exception as e:
            result_txt=f"Query error: {e}"

        if result_txt: st.markdown(f"**Answer:** {result_txt}")
        if result_df is not None and not result_df.empty:
            st.dataframe(result_df,use_container_width=True,height=320)
            st.download_button("⬇️ Download",data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# TAB 10: DATA CLEANING
# ═══════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="tab-header">Auto Data Cleaning</div>',unsafe_allow_html=True)
    quality_rows=[]
    for col in df_raw.columns:
        null_n=int(df_raw[col].isna().sum()); null_p=null_n/len(df_raw)*100
        dtype=str(df_raw[col].dtype); issues=[]
        if null_p>0: issues.append(f"{null_p:.1f}% missing")
        if df_raw[col].nunique()==1: issues.append("constant")
        if pd.api.types.is_numeric_dtype(df_raw[col]):
            q1,q3=df_raw[col].quantile(0.25),df_raw[col].quantile(0.75); iqr=q3-q1
            n_out=int(((df_raw[col]<q1-1.5*iqr)|(df_raw[col]>q3+1.5*iqr)).sum())
            if n_out>0: issues.append(f"{n_out} outliers")
        quality_rows.append({"Column":col,"Type":dtype,"Missing":f"{null_p:.1f}%",
            "Unique":int(df_raw[col].nunique()),
            "Issues":"; ".join(issues) if issues else "✅ Clean"})
    st.dataframe(pd.DataFrame(quality_rows),use_container_width=True,hide_index=True)
    div()

    df_clean=df_raw.copy()
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Missing value handling**")
        fill_method=st.selectbox("Fill missing with",
            ["(no action)","Mean","Median","Mode","Zero","Forward fill","Drop rows"],key="fill_meth")
        fill_cols=st.multiselect("Apply to columns",
            [c for c in num_cols_detected if c in df_clean.columns],key="fill_cols")
        if fill_method!="(no action)" and fill_cols:
            for col in fill_cols:
                if fill_method=="Mean": df_clean[col].fillna(df_clean[col].mean(),inplace=True)
                elif fill_method=="Median": df_clean[col].fillna(df_clean[col].median(),inplace=True)
                elif fill_method=="Mode": df_clean[col].fillna(df_clean[col].mode()[0],inplace=True)
                elif fill_method=="Zero": df_clean[col].fillna(0,inplace=True)
                elif fill_method=="Forward fill": df_clean[col].ffill(inplace=True)
                elif fill_method=="Drop rows": df_clean.dropna(subset=[col],inplace=True)
    with c2:
        st.markdown("**Duplicate & type fixes**")
        if st.checkbox("Remove duplicate rows",key="rm_dup"):
            before=len(df_clean); df_clean.drop_duplicates(inplace=True)
            st.caption(f"Removed {before-len(df_clean)} duplicates")
        if st.checkbox("Strip whitespace from text",key="strip_ws"):
            for col in df_clean.select_dtypes("object").columns:
                df_clean[col]=df_clean[col].str.strip()
        if st.checkbox("Convert text numbers to numeric",key="conv_num"):
            for col in df_clean.select_dtypes("object").columns:
                converted=pd.to_numeric(df_clean[col],errors="coerce")
                if converted.notna().mean()>0.8: df_clean[col]=converted

    div()
    cl1,cl2,cl3=st.columns(3)
    with cl1: kpi("Before",f"{len(df_raw):,}","Original rows","📊",ORANGE)
    with cl2: kpi("After",f"{len(df_clean):,}","Cleaned rows","✅",GREEN)
    with cl3: kpi("Removed",f"{len(df_raw)-len(df_clean):,}","Rows out","🗑️",RED if len(df_raw)!=len(df_clean) else TEAL)
    div()
    st.dataframe(df_clean.head(50).reset_index(drop=True),use_container_width=True,height=280)
    st.download_button("⬇️ Download cleaned CSV",data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",mime="text/csv")
    if st.button("✅ Use cleaned data for dashboard",key="use_clean"):
        st.session_state.df_raw=df_clean
        st.success("Cleaned data applied!"); st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 11: ML + REPORT
# ═══════════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown('<div class="tab-header">ML Recommender · Report Download</div>',unsafe_allow_html=True)
    recs=recommend_ml(df,cols_info,metric_col)
    badge_bg={"Regression":"rgba(255,107,53,0.15)","Classification":"rgba(167,139,250,0.15)",
              "Clustering":"rgba(251,191,36,0.15)","Time Series":"rgba(96,165,250,0.15)",
              "Anomaly Detection":"rgba(239,68,68,0.15)","Dim Reduction":"rgba(46,196,182,0.15)"}
    badge_tc={"Regression":ORANGE,"Classification":PURPLE,"Clustering":YELLOW,
              "Time Series":BLUE,"Anomaly Detection":RED,"Dim Reduction":TEAL}
    for r in recs:
        bg=badge_bg.get(r["badge"],"rgba(255,255,255,0.1)"); tc=badge_tc.get(r["badge"],ORANGE)
        st.markdown(f"""<div class="ml-card">
          <span class="ml-name">{r['name']}</span>
          <span class="ml-badge" style="background:{bg};color:{tc};">{r['badge']}</span>
          <span style="font-size:11px;color:#6B7280;margin-left:12px;">
            Difficulty:{r['difficulty']} · Accuracy:{r['accuracy']}</span>
          <div class="ml-desc">{r['desc']}</div>
          <div class="ml-why">💡 Why? {r['why']}</div>
          <div style="margin-top:8px;background:#12141E;border-radius:6px;padding:8px 12px;
               font-family:'DM Mono',monospace;font-size:11px;color:#9CA3AF;">{r['libs']}</div>
        </div>""",unsafe_allow_html=True)

    div(); st.markdown("#### 📐 Data Readiness for ML")
    num_avail=[c for c in num_cols_detected if c in df.columns]
    null_pct=df[num_avail].isnull().sum()/len(df)*100 if num_avail else pd.Series()
    avg_null=float(null_pct.mean()) if not null_pct.empty else 0
    r1,r2,r3,r4=st.columns(4)
    with r1: kpi("Rows",f"{len(df):,}","Training samples","📦",TEAL)
    with r2: kpi("Numeric Features",str(len(num_cols_detected)),"Input features","🔢",PURPLE)
    with r3: kpi("Category Features",str(len(cols_info.get("category_cols",[]))),"Need encoding","🏷️",YELLOW)
    with r4: kpi("Missing Data",f"{avg_null:.1f}%","Avg across cols","🕳️",RED if avg_null>10 else GREEN)

    if len(num_avail)>=2:
        div(); sec("Feature variance")
        var_df=df[num_avail].var().sort_values(ascending=False).reset_index()
        var_df.columns=["Feature","Variance"]
        var_df["Normalized"]=var_df["Variance"]/var_df["Variance"].sum()*100
        fig=go.Figure(go.Bar(x=var_df["Feature"],y=var_df["Normalized"],
            marker_color=PALETTE[:len(var_df)],
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"))
        fig.update_layout(**BASE,height=260,bargap=0.3,
            xaxis=dict(**AX),yaxis=dict(**AX,ticksuffix="%",title="% Variance"))
        st.plotly_chart(fig,use_container_width=True,key="var_chart")

    div(); st.markdown("### 📥 Download Report")
    dl1,dl2,dl3,dl4=st.columns(4)
    with dl1:
        try:
            st.download_button("📊 Excel",data=gen_excel(df,cols_info,metric_col),
                file_name="report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        except: st.error("pip install openpyxl")
    with dl2:
        st.download_button("📄 CSV",data=gen_csv(df),
            file_name="filtered_data.csv",mime="text/csv",use_container_width=True)
    with dl3:
        st.download_button("🔷 JSON",data=gen_json(df,cols_info,metric_col),
            file_name="report.json",mime="application/json",use_container_width=True)
    with dl4:
        recs_txt=recommend_ml(df,cols_info,metric_col)
        st.download_button("📋 TXT",
            data=gen_txt(df,cols_info,metric_col,insights,recs_txt),
            file_name="summary.txt",mime="text/plain",use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 12: GEMINI AI
# ═══════════════════════════════════════════════════════════════
with tabs[11]:
    st.markdown('<div class="tab-header">✨ Gemini AI Assistant</div>',unsafe_allow_html=True)
    g_key=st.session_state.get("gemini_key","").strip()
    if not g_key:
        st.warning("⚠️ Enter your Gemini API key in the sidebar.")
        st.markdown("""<div style="background:#1A1D27;border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:20px;margin-top:12px;">
          <div style="font-size:14px;font-weight:600;color:#F9FAFB;margin-bottom:10px;">
            Get a free Gemini API key:</div>
          <div style="font-size:13px;color:#9CA3AF;line-height:1.8;">
            1. Go to <b style="color:#FF6B35">aistudio.google.com</b><br>
            2. Sign in with Google → <b style="color:#FF6B35">Get API Key</b><br>
            3. Paste in the sidebar<br>
            <span style="color:#34D399">✅ Free: 15 req/min · 1M tokens/day</span>
          </div></div>""",unsafe_allow_html=True)
    else:
        if not GEMINI_AVAILABLE:
            st.error("Install: `pip install google-generativeai`")
        else:
            gem_model=init_gemini(g_key)
            data_ctx=build_data_context(df,cols_info,metric_col)
            ai1,ai2,ai3=st.tabs(["📋 Executive Summary","💡 Smart Insights","💬 Data Chat"])
            ai4,ai5,ai6=st.tabs(["🐍 Code Generator","📖 Story Writer","🎯 Prediction Advice"])

            with ai1:
                tone=st.selectbox("Tone",["Professional","Casual","Technical","C-Suite"],key="sum_tone")
                length=st.selectbox("Length",["Brief","Standard (1 paragraph)","Detailed (3 paragraphs)"],key="sum_len")
                if st.button("✨ Generate Summary",key="gen_summary",type="primary"):
                    with st.spinner("Writing..."):
                        result=gemini_ask(gem_model,f"""Senior business analyst. Write executive summary.
Tone:{tone} Length:{length}\n{data_ctx}\nStart directly with insights.""")
                    st.markdown(f"""<div style="background:#1A1D27;border-left:3px solid {TEAL};
                        border-radius:0 12px 12px 0;padding:20px;margin-top:12px;
                        font-size:14px;line-height:1.8;color:#E5E7EB;">
                        {result.replace(chr(10),"<br>")}</div>""",unsafe_allow_html=True)
                    st.download_button("⬇️ Download",result.encode(),"summary.txt","text/plain",key="dl_summary")

            with ai2:
                focus=st.multiselect("Focus",["Revenue trends","Customer behaviour","Seasonal patterns",
                    "Risk factors","Growth opportunities","Anomalies"],
                    default=["Revenue trends","Growth opportunities"],key="ins_focus")
                n_ins=st.slider("Number of insights",3,10,5,key="ins_n")
                if st.button("✨ Generate Insights",key="gen_insights",type="primary"):
                    with st.spinner("Analysing..."):
                        result=gemini_ask(gem_model,f"""Data scientist & strategist.
Generate {n_ins} actionable insights. Focus:{', '.join(focus)}\n{data_ctx}
Numbered list. Each: finding + why it matters + one action.""")
                    st.markdown(result)
                    st.download_button("⬇️ Download",result.encode(),"insights.txt","text/plain",key="dl_insights")

            with ai3:
                if "chat_history" not in st.session_state: st.session_state.chat_history=[]
                for msg in st.session_state.chat_history:
                    rc=ORANGE if msg["role"]=="user" else TEAL
                    rl="You" if msg["role"]=="user" else "Gemini"
                    st.markdown(f"""<div style="background:#1A1D27;border-left:3px solid {rc};
                        border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;">
                      <div style="font-size:10px;color:#6B7280;text-transform:uppercase;">{rl}</div>
                      <div style="font-size:13px;color:#E5E7EB;line-height:1.7;">
                        {msg['content'].replace(chr(10),'<br>')}</div></div>""",unsafe_allow_html=True)
                qcols=st.columns(3)
                for qi,qtext in enumerate(["Top 3 trends?","Biggest risk?","3 recommendations"]):
                    with qcols[qi]:
                        if st.button(qtext,key=f"qq_{qi}",use_container_width=True):
                            st.session_state["chat_input_val"]=qtext
                user_q=st.text_input("Ask anything...",value=st.session_state.get("chat_input_val",""),key="chat_input")
                cs,cc2=st.columns([3,1])
                with cs: send=st.button("Send →",key="chat_send",type="primary",use_container_width=True)
                with cc2:
                    if st.button("Clear",key="chat_clear",use_container_width=True):
                        st.session_state.chat_history=[]; st.rerun()
                if send and user_q.strip():
                    st.session_state.chat_history.append({"role":"user","content":user_q})
                    hist="\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history[-6:]])
                    with st.spinner("Thinking..."):
                        answer=gemini_ask(gem_model,f"Expert data analyst. Answer concisely with numbers.\n{data_ctx}\n\n{hist}")
                    st.session_state.chat_history.append({"role":"assistant","content":answer})
                    st.session_state["chat_input_val"]=""; st.rerun()

            with ai4:
                code_type=st.selectbox("Code type",["pandas analysis","plotly visualisation",
                    "scikit-learn ML model","SQL query","data cleaning"],key="code_type")
                code_task=st.text_area("Describe task",height=80,key="code_task")
                if st.button("✨ Generate Code",key="gen_code",type="primary") and code_task:
                    with st.spinner("Writing code..."):
                        result=gemini_ask(gem_model,f"""Python expert. Write {code_type} code.
Cols:{df.columns.tolist()} Metric:{metric_col} Date:{cols_info.get('date_col','None')}
Sample:\n{df.head(3).to_string(index=False)}\nTask:{code_task}
df already loaded. Standard libs. Comments. Return ONLY code.""")
                    clean=result.strip()
                    if clean.startswith("```"): clean="\n".join(clean.split("\n")[1:])
                    if clean.endswith("```"): clean="\n".join(clean.split("\n")[:-1])
                    st.code(clean.strip(),language="python")
                    st.download_button("⬇️ Download",clean.strip().encode(),"code.py","text/plain",key="dl_code")

            with ai5:
                audience=st.selectbox("Audience",["CEO / Board","Marketing","Sales","Investors","General public"],key="story_audience")
                style=st.selectbox("Style",["Business report","Blog post","Newsletter","Press release"],key="story_style")
                if st.button("✨ Write Story",key="gen_story",type="primary"):
                    with st.spinner("Crafting story..."):
                        result=gemini_ask(gem_model,f"""Business storyteller. Write {style} for {audience}.
{data_ctx}\nLead with impact. Use numbers. Under 400 words.""")
                    st.markdown(f"""<div style="background:#1A1D27;border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:24px;font-size:14px;line-height:1.9;color:#E5E7EB;">
                        {result.replace(chr(10),'<br>')}</div>""",unsafe_allow_html=True)
                    st.download_button("⬇️ Download",result.encode(),"story.txt","text/plain",key="dl_story")

            with ai6:
                horizon=st.selectbox("Horizon",["Next week","Next month","Next quarter","Next year"],key="pred_horizon")
                pred_focus_q=st.text_input("Specific question",placeholder="Will revenue exceed 500K?",key="pred_focus")
                if st.button("✨ Get Predictions",key="gen_pred",type="primary"):
                    with st.spinner("Analysing trends..."):
                        result=gemini_ask(gem_model,f"""Senior forecaster. Predictions for {horizon}.
{f'Address: {pred_focus_q}' if pred_focus_q else ''}\n{data_ctx}
Format: 1.KEY PREDICTION + confidence 2.EVIDENCE (3 bullets) 3.RISKS (2 bullets) 4.ACTIONS (3 steps)""")
                    st.markdown(f"""<div style="background:#1A1D27;border-left:3px solid {PURPLE};
                        border-radius:0 12px 12px 0;padding:20px;font-size:14px;line-height:1.8;color:#E5E7EB;">
                        {result.replace(chr(10),'<br>')}</div>""",unsafe_allow_html=True)
                    st.download_button("⬇️ Download",result.encode(),"predictions.txt","text/plain",key="dl_pred")

# ═══════════════════════════════════════════════════════════════
# TAB 13 — NEW: STATISTICAL DEEP DIVE
# ═══════════════════════════════════════════════════════════════
with tabs[12]:
    st.markdown('<div class="tab-header">📐 Statistical Deep Dive</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Normality tests · Hypothesis testing · Advanced descriptive stats</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    if not num_avail:
        st.info("No numeric columns available.")
    else:
        stat_col=st.selectbox("Select column",num_avail,key="stat_col")
        vals=df[stat_col].dropna()

        # ── Full Descriptive Stats ────────────────────────────────
        st.markdown("#### 📊 Full Descriptive Statistics")
        desc_rows={
            "Count":len(vals),"Mean":vals.mean(),"Median":vals.median(),
            "Mode":float(vals.mode()[0]) if not vals.mode().empty else None,
            "Std Dev":vals.std(),"Variance":vals.var(),
            "Min":vals.min(),"Max":vals.max(),"Range":vals.max()-vals.min(),
            "Skewness":vals.skew(),"Kurtosis":vals.kurtosis(),
            "CV (%)":vals.std()/vals.mean()*100 if vals.mean()!=0 else None,
            "IQR":vals.quantile(0.75)-vals.quantile(0.25),
            "P5":vals.quantile(0.05),"P25":vals.quantile(0.25),
            "P50":vals.quantile(0.50),"P75":vals.quantile(0.75),
            "P95":vals.quantile(0.95),"P99":vals.quantile(0.99),
        }
        desc_df=pd.DataFrame({"Statistic":list(desc_rows.keys()),
                               "Value":[round(float(v),4) if v is not None else "N/A" for v in desc_rows.values()]})
        d1,d2=st.columns([1,2])
        with d1: st.dataframe(desc_df,use_container_width=True,hide_index=True,height=460)
        with d2:
            fig=go.Figure()
            fig.add_trace(go.Histogram(x=vals,nbinsx=40,marker_color=TEAL,marker_opacity=0.7,name="Data"))
            mu,sigma=vals.mean(),vals.std()
            x_range=np.linspace(vals.min(),vals.max(),200)
            pdf=scipy_stats.norm.pdf(x_range,mu,sigma)
            scale_factor=len(vals)*(vals.max()-vals.min())/40
            fig.add_trace(go.Scatter(x=x_range,y=pdf*scale_factor,mode="lines",
                line=dict(color=ORANGE,width=2.5),name="Normal fit"))
            fig.update_layout(**BASE,height=460,xaxis=dict(**AX),yaxis=dict(**AX),
                legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig,use_container_width=True,key="stat_hist_fit")
        div()

        # ── Normality Tests ───────────────────────────────────────
        st.markdown("#### 🧪 Normality Tests")
        n_sample=min(len(vals),5000)
        vsample=vals.sample(n_sample,random_state=42) if len(vals)>5000 else vals
        tests=[]
        try:
            stat,p=scipy_stats.shapiro(vsample[:5000])
            tests.append({"Test":"Shapiro-Wilk","Statistic":round(stat,4),"p-value":round(p,6),
                "Result":"✅ Normal" if p>0.05 else "❌ Not Normal","Note":"Best for n<5000"})
        except: pass
        try:
            stat,p=scipy_stats.kstest(vals,"norm",args=(vals.mean(),vals.std()))
            tests.append({"Test":"Kolmogorov-Smirnov","Statistic":round(stat,4),"p-value":round(p,6),
                "Result":"✅ Normal" if p>0.05 else "❌ Not Normal","Note":"Goodness of fit"})
        except: pass
        try:
            stat,p=scipy_stats.normaltest(vals)
            tests.append({"Test":"D'Agostino-Pearson","Statistic":round(stat,4),"p-value":round(p,6),
                "Result":"✅ Normal" if p>0.05 else "❌ Not Normal","Note":"Skew+Kurtosis combined"})
        except: pass
        try:
            stat,p=scipy_stats.anderson(vsample)
            tests.append({"Test":"Anderson-Darling","Statistic":round(float(stat),4),"p-value":"—",
                "Result":"See critical values","Note":"Multiple significance levels"})
        except: pass
        if tests:
            tdf=pd.DataFrame(tests)
            st.dataframe(tdf,use_container_width=True,hide_index=True)
        div()

        # ── QQ Plot ───────────────────────────────────────────────
        st.markdown("#### 📉 QQ Plot & ECDF")
        qq1,qq2=st.columns(2)
        with qq1:
            sec("QQ Plot (vs Normal)")
            st.plotly_chart(ch_qq(df,stat_col),use_container_width=True,key="stat_qq")
        with qq2:
            sec("ECDF")
            st.plotly_chart(ch_ecdf(df,stat_col),use_container_width=True,key="stat_ecdf")
        div()

        # ── Hypothesis Testing ────────────────────────────────────
        st.markdown("#### ⚖️ Hypothesis Testing")
        ht1,ht2=st.tabs(["One-Sample t-test","Two-Sample t-test"])
        with ht1:
            mu0=st.number_input("Hypothesised mean (μ₀)",value=float(vals.mean()),key="mu0")
            alpha=st.selectbox("Significance level",["0.05","0.01","0.10"],key="alpha_1")
            if st.button("Run t-test",key="ttest_1"):
                stat,p=scipy_stats.ttest_1samp(vals,mu0)
                a=float(alpha)
                t1,t2,t3=st.columns(3)
                with t1: result_card("t-statistic",f"{stat:.4f}","",ORANGE)
                with t2: result_card("p-value",f"{p:.6f}","",TEAL if p>a else RED)
                with t3: result_card("Decision","Reject H₀" if p<a else "Fail to reject H₀",
                    f"at α={alpha}",RED if p<a else GREEN)
        with ht2:
            if cat_cols:
                grp_col=st.selectbox("Group column",cat_cols,key="grp_col_tt")
                groups=df[grp_col].dropna().unique().tolist()
                if len(groups)>=2:
                    g1=st.selectbox("Group 1",groups,key="tt_g1")
                    g2=st.selectbox("Group 2",groups,index=min(1,len(groups)-1),key="tt_g2")
                    if st.button("Run 2-sample t-test",key="ttest_2"):
                        v1=df[df[grp_col]==g1][stat_col].dropna()
                        v2=df[df[grp_col]==g2][stat_col].dropna()
                        stat,p=scipy_stats.ttest_ind(v1,v2)
                        st.metric("t-statistic",f"{stat:.4f}")
                        st.metric("p-value",f"{p:.6f}")
                        st.info(f"{'Reject H₀ — means are significantly different' if p<0.05 else 'Fail to reject H₀ — no significant difference'} (α=0.05)")
            else:
                st.info("Need at least one category column for 2-sample test.")
        div()

        # ── ANOVA ─────────────────────────────────────────────────
        if cat_cols:
            st.markdown("#### 🔬 One-Way ANOVA")
            anova_col=st.selectbox("Group by",cat_cols,key="anova_col")
            if st.button("Run ANOVA",key="run_anova"):
                groups_data=[df[df[anova_col]==g][stat_col].dropna().values
                             for g in df[anova_col].dropna().unique() if len(df[df[anova_col]==g][stat_col].dropna())>=2]
                if len(groups_data)>=2:
                    f_stat,p_val=scipy_stats.f_oneway(*groups_data)
                    a1,a2,a3=st.columns(3)
                    with a1: result_card("F-statistic",f"{f_stat:.4f}","",ORANGE)
                    with a2: result_card("p-value",f"{p_val:.6f}","",TEAL if p_val>0.05 else RED)
                    with a3: result_card("Decision",
                        "Significant difference" if p_val<0.05 else "No significant difference",
                        "between groups at α=0.05",RED if p_val<0.05 else GREEN)
                else:
                    st.warning("Need at least 2 groups with ≥2 observations each.")

# ═══════════════════════════════════════════════════════════════
# TAB 14 — NEW: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
with tabs[13]:
    st.markdown('<div class="tab-header">🧬 Feature Engineering</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Create new columns, transform data, extract date parts — on the fly.</div>',unsafe_allow_html=True)
    df_eng=st.session_state.get("df_eng",df.copy())
    num_avail=[c for c in df_eng.columns if pd.api.types.is_numeric_dtype(df_eng[c])]

    eng1,eng2,eng3,eng4=st.tabs(["➕ New Column","📊 Binning","📐 Transforms","📅 Date Parts"])

    with eng1:
        st.markdown("#### Create a calculated column")
        new_col_name=st.text_input("New column name",value="new_col",key="eng_name")
        if len(num_avail)>=2:
            col_a=st.selectbox("Column A",num_avail,key="eng_a")
            op=st.selectbox("Operation",["+","-","×","÷","% of total","ratio A/B","log(A)","sqrt(A)","A²"],key="eng_op")
            col_b=st.selectbox("Column B",num_avail,key="eng_b")
            if st.button("✅ Create Column",key="eng_create"):
                try:
                    if op=="+":       df_eng[new_col_name]=df_eng[col_a]+df_eng[col_b]
                    elif op=="-":     df_eng[new_col_name]=df_eng[col_a]-df_eng[col_b]
                    elif op=="×":     df_eng[new_col_name]=df_eng[col_a]*df_eng[col_b]
                    elif op=="÷":     df_eng[new_col_name]=df_eng[col_a]/df_eng[col_b].replace(0,np.nan)
                    elif op=="% of total": df_eng[new_col_name]=df_eng[col_a]/df_eng[col_a].sum()*100
                    elif op=="ratio A/B": df_eng[new_col_name]=df_eng[col_a]/df_eng[col_b].replace(0,np.nan)
                    elif op=="log(A)": df_eng[new_col_name]=np.log1p(df_eng[col_a].clip(lower=0))
                    elif op=="sqrt(A)": df_eng[new_col_name]=np.sqrt(df_eng[col_a].clip(lower=0))
                    elif op=="A²":    df_eng[new_col_name]=df_eng[col_a]**2
                    st.session_state["df_eng"]=df_eng
                    st.success(f"✅ Column **{new_col_name}** created!")
                    st.markdown(f'<span class="feat-tag">{new_col_name}</span>',unsafe_allow_html=True)
                except Exception as e: st.error(str(e))

    with eng2:
        st.markdown("#### Bin a numeric column")
        bin_col=st.selectbox("Column to bin",num_avail,key="bin_col")
        n_bins=st.slider("Number of bins",2,20,5,key="n_bins")
        bin_method=st.radio("Method",["Equal width","Quantile (equal freq)"],horizontal=True,key="bin_method")
        bin_label_name=st.text_input("Output column name",value=f"{bin_col}_bin",key="bin_label_name")
        if st.button("✅ Create Bins",key="bin_create"):
            try:
                if bin_method=="Equal width":
                    df_eng[bin_label_name]=pd.cut(df_eng[bin_col],bins=n_bins,labels=False)
                else:
                    df_eng[bin_label_name]=pd.qcut(df_eng[bin_col],q=n_bins,labels=False,duplicates="drop")
                st.session_state["df_eng"]=df_eng
                st.success(f"✅ Bins created: **{bin_label_name}**")
                st.bar_chart(df_eng[bin_label_name].value_counts().sort_index())
            except Exception as e: st.error(str(e))

    with eng3:
        st.markdown("#### Apply transformations")
        trans_col=st.selectbox("Column",num_avail,key="trans_col")
        transforms=st.multiselect("Transformations to apply",
            ["Log (log1p)","Square root","Normalize (0–1)","Standardize (Z-score)",
             "Rank","Absolute value","Inverse (1/x)"],key="trans_sel")
        if st.button("✅ Apply Transforms",key="trans_apply") and transforms:
            for t in transforms:
                tname=f"{trans_col}_{t.split()[0].lower()}"
                try:
                    if t=="Log (log1p)": df_eng[tname]=np.log1p(df_eng[trans_col].clip(lower=0))
                    elif t=="Square root": df_eng[tname]=np.sqrt(df_eng[trans_col].clip(lower=0))
                    elif t=="Normalize (0–1)":
                        mn,mx=df_eng[trans_col].min(),df_eng[trans_col].max()
                        df_eng[tname]=(df_eng[trans_col]-mn)/(mx-mn+1e-9)
                    elif t=="Standardize (Z-score)":
                        df_eng[tname]=(df_eng[trans_col]-df_eng[trans_col].mean())/df_eng[trans_col].std()
                    elif t=="Rank": df_eng[tname]=df_eng[trans_col].rank()
                    elif t=="Absolute value": df_eng[tname]=df_eng[trans_col].abs()
                    elif t=="Inverse (1/x)": df_eng[tname]=1/df_eng[trans_col].replace(0,np.nan)
                    st.markdown(f'<span class="feat-tag">{tname}</span>',unsafe_allow_html=True)
                except Exception as e: st.error(f"{t}: {e}")
            st.session_state["df_eng"]=df_eng
            st.success(f"✅ {len(transforms)} transforms applied!")

    with eng4:
        if date_col and pd.api.types.is_datetime64_any_dtype(df_eng.get(date_col,pd.Series())):
            st.markdown(f"#### Extract parts from **{date_col}**")
            parts=st.multiselect("Parts to extract",
                ["Year","Month","Day","DayOfWeek","Quarter","WeekOfYear","Hour","IsWeekend"],
                default=["Year","Month","Quarter"],key="date_parts")
            if st.button("✅ Extract Date Parts",key="date_extract") and parts:
                for p in parts:
                    try:
                        if p=="Year": df_eng[f"{date_col}_year"]=df_eng[date_col].dt.year
                        elif p=="Month": df_eng[f"{date_col}_month"]=df_eng[date_col].dt.month
                        elif p=="Day": df_eng[f"{date_col}_day"]=df_eng[date_col].dt.day
                        elif p=="DayOfWeek": df_eng[f"{date_col}_dow"]=df_eng[date_col].dt.dayofweek
                        elif p=="Quarter": df_eng[f"{date_col}_quarter"]=df_eng[date_col].dt.quarter
                        elif p=="WeekOfYear": df_eng[f"{date_col}_week"]=df_eng[date_col].dt.isocalendar().week.astype(int)
                        elif p=="Hour": df_eng[f"{date_col}_hour"]=df_eng[date_col].dt.hour
                        elif p=="IsWeekend": df_eng[f"{date_col}_weekend"]=(df_eng[date_col].dt.dayofweek>=5).astype(int)
                    except Exception as e: st.error(f"{p}: {e}")
                st.session_state["df_eng"]=df_eng
                st.success(f"✅ {len(parts)} date parts extracted!")
        else:
            st.info("No date column detected. Select one in the sidebar.")

    div()
    new_cols=[c for c in df_eng.columns if c not in df.columns]
    if new_cols:
        st.markdown(f"**{len(new_cols)} new columns created:**")
        for c in new_cols:
            st.markdown(f'<span class="feat-tag">{c}</span>',unsafe_allow_html=True)
        st.dataframe(df_eng[new_cols].head(20),use_container_width=True)
        st.download_button("⬇️ Download engineered dataset",
            data=df_eng.to_csv(index=False).encode("utf-8"),
            file_name="engineered_data.csv",mime="text/csv")
        if st.button("✅ Use as main data",key="use_eng"):
            st.session_state.df_raw=df_eng
            st.success("Applied!"); st.rerun()
    else:
        st.info("No new columns created yet. Use the tabs above to add features.")

# ═══════════════════════════════════════════════════════════════
# TAB 15 — NEW: GEO / MAP
# ═══════════════════════════════════════════════════════════════
with tabs[14]:
    st.markdown('<div class="tab-header">🌐 Geo / Map Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Choropleth maps, geo scatter, and location-based analysis.</div>',unsafe_allow_html=True)

    num_avail=[c for c in num_cols_detected if c in df.columns]
    all_cols_list=df.columns.tolist()

    geo_mode=st.radio("Map type",["Choropleth (country)","Choropleth (US state)","Scatter Geo","Bubble Map"],
                      horizontal=True,key="geo_mode")

    if geo_mode in ["Choropleth (country)","Choropleth (US state)"]:
        st.info("Select a column that contains country names or ISO codes (for world) or US state codes.")
        loc_col=st.selectbox("Location column",all_cols_list,key="geo_loc")
        val_col=st.selectbox("Value column",num_avail,key="geo_val")
        if st.button("🗺️ Draw Map",key="draw_choropleth"):
            try:
                scope="world" if geo_mode=="Choropleth (country)" else "usa"
                lmode="country names" if geo_mode=="Choropleth (country)" else "USA-states"
                grp=df.groupby(loc_col)[val_col].sum().reset_index()
                fig=px.choropleth(grp,locations=loc_col,locationmode=lmode,color=val_col,
                    color_continuous_scale=[[0,DARK_BG],[0.5,TEAL],[1,ORANGE]],scope=scope)
                fig.update_layout(**BASE,height=500,geo=dict(bgcolor=DARK_BG,
                    landcolor="#1A1D27",showland=True,showcoastlines=True,
                    coastlinecolor="rgba(255,255,255,0.1)"))
                st.plotly_chart(fig,use_container_width=True,key="choropleth_map")
            except Exception as e:
                st.error(f"Map error: {e}. Ensure location column has valid country/state names.")

    elif geo_mode=="Scatter Geo":
        st.info("Need latitude and longitude columns.")
        lat_col=st.selectbox("Latitude column",all_cols_list,key="lat_col")
        lon_col=st.selectbox("Longitude column",all_cols_list,key="lon_col")
        color_col=st.selectbox("Color by",[None]+all_cols_list,key="geo_color")
        if st.button("🗺️ Draw Map",key="draw_scatter_geo"):
            try:
                kw=dict(data_frame=df.dropna(subset=[lat_col,lon_col]),
                    lat=lat_col,lon=lon_col,opacity=0.6,color_discrete_sequence=PALETTE)
                if color_col and color_col in df.columns: kw["color"]=color_col
                fig=px.scatter_geo(**kw)
                fig.update_layout(**BASE,height=500,
                    geo=dict(bgcolor=DARK_BG,landcolor="#1A1D27",
                             showcoastlines=True,coastlinecolor="rgba(255,255,255,0.1)"))
                st.plotly_chart(fig,use_container_width=True,key="scatter_geo_map")
            except Exception as e: st.error(str(e))

    elif geo_mode=="Bubble Map":
        st.info("Need latitude, longitude, and a size column.")
        lat_col=st.selectbox("Latitude",all_cols_list,key="bm_lat")
        lon_col=st.selectbox("Longitude",all_cols_list,key="bm_lon")
        size_col=st.selectbox("Bubble size",num_avail,key="bm_size")
        if st.button("🗺️ Draw Bubble Map",key="draw_bubble_geo"):
            try:
                fig=px.scatter_geo(df.dropna(subset=[lat_col,lon_col,size_col]),
                    lat=lat_col,lon=lon_col,size=size_col,
                    color=size_col,size_max=40,opacity=0.7,
                    color_continuous_scale=[[0,TEAL],[1,ORANGE]])
                fig.update_layout(**BASE,height=500,
                    geo=dict(bgcolor=DARK_BG,landcolor="#1A1D27",
                             showcoastlines=True,coastlinecolor="rgba(255,255,255,0.1)"))
                st.plotly_chart(fig,use_container_width=True,key="bubble_geo_map")
            except Exception as e: st.error(str(e))

    div()
    if cat_cols:
        st.markdown("#### 📊 Location-Based Summary")
        loc_sum_col=st.selectbox("Group by",cat_cols,key="loc_sum_col")
        loc_met=st.selectbox("Metric",num_avail,key="loc_met")
        grp=df.groupby(loc_sum_col)[loc_met].agg(["sum","mean","count"]).round(2).reset_index()
        grp.columns=[loc_sum_col,"Total","Average","Count"]
        st.dataframe(grp.sort_values("Total",ascending=False),use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 16 — NEW: COHORT ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tabs[15]:
    st.markdown('<div class="tab-header">📉 Cohort & Retention Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Understand user retention and cohort behaviour over time.</div>',unsafe_allow_html=True)

    if not date_col or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.info("Cohort analysis requires a date column.")
    else:
        st.markdown("#### Cohort Heatmap")
        st.caption("Groups records by first-seen month and tracks activity in subsequent months.")
        id_col_opts=["(use row index)"]+[c for c in df.columns if df[c].nunique()==len(df) or df[c].nunique()>len(df)*0.7]
        id_col=st.selectbox("Unique ID column (optional)",id_col_opts,key="cohort_id")
        if st.button("📊 Build Cohort",key="build_cohort"):
            try:
                tmp=df[[date_col,metric_col]].dropna().copy()
                tmp["CohortMonth"]=tmp[date_col].dt.to_period("M")
                tmp["OrderMonth"]=tmp[date_col].dt.to_period("M")
                cohort_grp=tmp.groupby(["CohortMonth","OrderMonth"])[metric_col].sum().reset_index()
                cohort_grp["CohortIndex"]=(cohort_grp["OrderMonth"]-cohort_grp["CohortMonth"]).apply(lambda x: x.n)
                cohort_pivot=cohort_grp.pivot_table(index="CohortMonth",columns="CohortIndex",
                    values=metric_col,aggfunc="sum")
                cohort_pct=cohort_pivot.divide(cohort_pivot.iloc[:,0],axis=0)*100
                fig=go.Figure(go.Heatmap(
                    z=cohort_pct.values,
                    x=[f"Month +{i}" for i in cohort_pct.columns],
                    y=cohort_pct.index.astype(str).tolist(),
                    colorscale=[[0,DARK_BG],[0.3,PURPLE],[0.7,ORANGE],[1,YELLOW]],
                    showscale=True,
                    text=cohort_pct.round(1).values,
                    texttemplate="%{text}%",
                    hovertemplate="Cohort: %{y}<br>%{x}<br>Retention: %{z:.1f}%<extra></extra>"))
                fig.update_layout(**BASE,height=500,
                    xaxis=dict(tickfont_size=10,gridcolor=GRID),
                    yaxis=dict(tickfont_size=10,gridcolor=GRID,autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True,key="cohort_heatmap")
                div()
                st.markdown("#### Retention Curve (avg across cohorts)")
                avg_ret=cohort_pct.mean(axis=0).reset_index()
                avg_ret.columns=["Month","Avg Retention %"]
                fig2=go.Figure(go.Scatter(x=avg_ret["Month"],y=avg_ret["Avg Retention %"],
                    mode="lines+markers",line=dict(color=ORANGE,width=2.5),
                    fill="tozeroy",fillcolor="rgba(255,107,53,0.08)",
                    hovertemplate="Month +%{x}<br>Retention: %{y:.1f}%<extra></extra>"))
                fig2.update_layout(**BASE,xaxis=dict(**AX,title="Months since first period"),
                    yaxis=dict(**AX,title="Avg Retention %",ticksuffix="%"))
                st.plotly_chart(fig2,use_container_width=True,key="retention_curve")
            except Exception as e:
                st.error(f"Cohort error: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 17 — NEW: MONTE-CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════
with tabs[16]:
    st.markdown('<div class="tab-header">🔀 Monte-Carlo Simulation</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Run probabilistic scenarios — forecast with uncertainty using simulation.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]
    if not num_avail:
        st.info("Need numeric columns.")
    else:
        mc_col=st.selectbox("Simulate based on",num_avail,key="mc_col")
        vals=df[mc_col].dropna()
        mc1,mc2=st.columns(2)
        with mc1:
            n_sims=st.slider("Number of simulations",500,10000,2000,500,key="mc_n")
            n_periods=st.slider("Forecast periods",10,120,30,key="mc_periods")
        with mc2:
            mc_mean=st.number_input("Mean growth rate (%)",value=float(vals.pct_change().mean()*100),
                format="%.3f",key="mc_mean")
            mc_std=st.number_input("Std dev of growth (%)",value=float(vals.pct_change().std()*100),
                format="%.3f",key="mc_std")

        if st.button("🔀 Run Monte-Carlo",key="run_mc"):
            with st.spinner(f"Running {n_sims:,} simulations..."):
                start_val=float(vals.iloc[-1])
                mu_d=mc_mean/100; sigma_d=mc_std/100
                all_paths=np.zeros((n_sims,n_periods+1))
                all_paths[:,0]=start_val
                for t in range(1,n_periods+1):
                    shocks=np.random.normal(mu_d,sigma_d,n_sims)
                    all_paths[:,t]=all_paths[:,t-1]*(1+shocks)
                final_vals=all_paths[:,-1]
                p5,p25,p50,p75,p95=np.percentile(final_vals,[5,25,50,75,95])

            fig=go.Figure()
            for i in range(min(200,n_sims)):
                fig.add_trace(go.Scatter(y=all_paths[i],mode="lines",
                    line=dict(color="rgba(255,107,53,0.04)",width=1),showlegend=False))
            for pct,color,name in [(5,RED,"P5"),(25,YELLOW,"P25"),(50,ORANGE,"Median"),(75,YELLOW,"P75"),(95,GREEN,"P95")]:
                idx_val=all_paths[np.argsort(all_paths[:,-1])[int(pct/100*n_sims)]]
                fig.add_trace(go.Scatter(y=idx_val,mode="lines",
                    line=dict(color=color,width=2),name=name))
            fig.update_layout(**BASE,height=420,
                xaxis=dict(**AX,title="Period"),yaxis=dict(**AX,title=mc_col),
                legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig,use_container_width=True,key="mc_paths")
            div()
            r1,r2,r3,r4,r5=st.columns(5)
            with r1: kpi("P5 (Bear)",fmt(p5),f"5th percentile","📉",RED)
            with r2: kpi("P25",fmt(p25),f"25th percentile","📊",YELLOW)
            with r3: kpi("Median",fmt(p50),f"50th percentile","📊",ORANGE)
            with r4: kpi("P75",fmt(p75),f"75th percentile","📈",YELLOW)
            with r5: kpi("P95 (Bull)",fmt(p95),f"95th percentile","🚀",GREEN)
            div()
            fig2=go.Figure(go.Histogram(x=final_vals,nbinsx=60,marker_color=ORANGE,
                marker_opacity=0.7,hovertemplate="Value: %{x:,.2f}<br>Count: %{y}<extra></extra>"))
            for p_val,color in [(p5,RED),(p25,YELLOW),(p50,ORANGE),(p75,YELLOW),(p95,GREEN)]:
                fig2.add_vline(x=p_val,line_color=color,line_dash="dash",line_width=1.5)
            fig2.update_layout(**BASE,height=300,xaxis=dict(**AX,title=f"Final {mc_col}"),
                yaxis=dict(**AX,title="Count"))
            st.plotly_chart(fig2,use_container_width=True,key="mc_dist")
            prob_above=float((final_vals>start_val).mean()*100)
            st.info(f"📊 Probability of growth from start: **{prob_above:.1f}%** across {n_sims:,} simulations")

# ═══════════════════════════════════════════════════════════════
# TAB 18 — NEW: COMPARATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tabs[17]:
    st.markdown('<div class="tab-header">📊 Comparative Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Compare multiple metrics, groups, and time periods side-by-side.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]

    comp1,comp2,comp3=st.tabs(["Multi-Metric","Group Comparison","Period Comparison"])

    with comp1:
        st.markdown("#### Compare multiple metrics normalized")
        sel_metrics=st.multiselect("Select metrics",num_avail,default=num_avail[:4],key="comp_metrics")
        if sel_metrics and len(sel_metrics)>=2:
            normed=df[sel_metrics].copy()
            for c in sel_metrics:
                mn,mx=normed[c].min(),normed[c].max()
                normed[c]=(normed[c]-mn)/(mx-mn+1e-9)
            if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                normed[date_col]=df[date_col]
                grp=normed.groupby(normed[date_col].dt.to_period("M").dt.to_timestamp())[sel_metrics].mean().reset_index()
                fig=go.Figure()
                for i,m in enumerate(sel_metrics):
                    fig.add_trace(go.Scatter(x=grp[date_col],y=grp[m],mode="lines",name=m,
                        line=dict(color=PALETTE[i],width=2)))
                fig.update_layout(**BASE,hovermode="x unified",xaxis=dict(**AX),
                    yaxis=dict(**AX,title="Normalised Value"),
                    legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig,use_container_width=True,key="comp_multi")
            else:
                st.dataframe(df[sel_metrics].describe().round(3),use_container_width=True)

    with comp2:
        if cat_cols:
            st.markdown("#### Compare groups on multiple metrics")
            grp_col=st.selectbox("Group by",cat_cols,key="comp_grp_col")
            grp_metrics=st.multiselect("Metrics",num_avail,default=num_avail[:3],key="comp_grp_met")
            if grp_metrics:
                grp_df=df.groupby(grp_col)[grp_metrics].mean().round(2).reset_index()
                fig=px.bar(grp_df.melt(id_vars=grp_col,value_vars=grp_metrics),
                    x=grp_col,y="value",color="variable",barmode="group",
                    color_discrete_sequence=PALETTE)
                fig.update_layout(**BASE,bargap=0.15,
                    xaxis=dict(**AX),yaxis=dict(**AX,title="Mean value"),
                    legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig,use_container_width=True,key="comp_group")
                div(); st.dataframe(grp_df,use_container_width=True,hide_index=True)
        else:
            st.info("Need category columns for group comparison.")

    with comp3:
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            st.markdown("#### Compare two time periods")
            col_p1,col_p2=st.columns(2)
            periods_available=df[date_col].dt.to_period("M").unique()
            periods_str=[str(p) for p in sorted(periods_available)]
            if len(periods_str)>=2:
                with col_p1:
                    p1=st.selectbox("Period 1",periods_str,index=0,key="comp_p1")
                with col_p2:
                    p2=st.selectbox("Period 2",periods_str,index=len(periods_str)-1,key="comp_p2")
                if st.button("Compare Periods",key="comp_periods_btn"):
                    d1=df[df[date_col].dt.to_period("M").astype(str)==p1]
                    d2=df[df[date_col].dt.to_period("M").astype(str)==p2]
                    comp_rows=[]
                    for m in num_avail[:6]:
                        v1=float(d1[m].sum()); v2=float(d2[m].sum())
                        chg=((v2-v1)/v1*100) if v1!=0 else None
                        comp_rows.append({"Metric":m,f"{p1}":round(v1,2),f"{p2}":round(v2,2),
                            "Change %":f"{chg:+.1f}%" if chg is not None else "N/A"})
                    cdf=pd.DataFrame(comp_rows)
                    st.dataframe(cdf,use_container_width=True,hide_index=True)
        else:
            st.info("Need date column for period comparison.")

# ═══════════════════════════════════════════════════════════════
# TAB 19 — NEW: AutoML LITE
# ═══════════════════════════════════════════════════════════════
with tabs[18]:
    st.markdown('<div class="tab-header">🧠 AutoML Lite</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Train Ridge · Random Forest · XGBoost — evaluate, compare, visualise feature importance.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]

    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import (mean_squared_error,mean_absolute_error,r2_score,
                                     accuracy_score,classification_report,roc_auc_score,roc_curve)
        sklearn_aml=True
    except:
        sklearn_aml=False
        st.warning("Install scikit-learn: `pip install scikit-learn`")

    if sklearn_aml and len(num_avail)>=2:
        target_col=st.selectbox("Target (Y) column",num_avail,key="aml_target")
        feature_cols=st.multiselect("Feature (X) columns",
            [c for c in num_avail if c!=target_col],
            default=[c for c in num_avail if c!=target_col][:5],key="aml_features")
        test_size=st.slider("Test set size %",10,40,20,key="aml_test")
        aml_models=st.multiselect("Models to train",
            ["Ridge Regression","Random Forest","XGBoost (if installed)"],
            default=["Ridge Regression","Random Forest"],key="aml_models")

        if len(feature_cols)>=1 and st.button("🧠 Train & Evaluate",key="aml_train"):
            tmp=df[feature_cols+[target_col]].dropna()
            X=tmp[feature_cols].values; y=tmp[target_col].values
            is_reg=len(np.unique(y))>10
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=test_size/100,random_state=42)
            scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr); X_te_s=scaler.transform(X_te)
            results=[]; feat_imps={}

            for model_name in aml_models:
                try:
                    if model_name=="Ridge Regression":
                        m=Ridge(); m.fit(X_tr_s,y_tr)
                        preds=m.predict(X_te_s)
                        if is_reg:
                            results.append({"Model":"Ridge","R²":round(r2_score(y_te,preds),4),
                                "RMSE":round(mean_squared_error(y_te,preds)**0.5,4),
                                "MAE":round(mean_absolute_error(y_te,preds),4)})
                        feat_imps["Ridge"]=dict(zip(feature_cols,m.coef_ if len(m.coef_.shape)==1 else m.coef_[0]))

                    elif model_name=="Random Forest":
                        m=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1) if is_reg \
                          else RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
                        m.fit(X_tr_s,y_tr); preds=m.predict(X_te_s)
                        if is_reg:
                            results.append({"Model":"Random Forest","R²":round(r2_score(y_te,preds),4),
                                "RMSE":round(mean_squared_error(y_te,preds)**0.5,4),
                                "MAE":round(mean_absolute_error(y_te,preds),4)})
                        feat_imps["RF"]=dict(zip(feature_cols,m.feature_importances_))

                    elif model_name=="XGBoost (if installed)":
                        try:
                            from xgboost import XGBRegressor,XGBClassifier
                            m=XGBRegressor(n_estimators=100,random_state=42,verbosity=0) if is_reg \
                              else XGBClassifier(n_estimators=100,random_state=42,verbosity=0)
                            m.fit(X_tr_s,y_tr); preds=m.predict(X_te_s)
                            if is_reg:
                                results.append({"Model":"XGBoost","R²":round(r2_score(y_te,preds),4),
                                    "RMSE":round(mean_squared_error(y_te,preds)**0.5,4),
                                    "MAE":round(mean_absolute_error(y_te,preds),4)})
                            feat_imps["XGB"]=dict(zip(feature_cols,m.feature_importances_))
                        except: st.warning("XGBoost not installed: `pip install xgboost`")
                except Exception as e: st.error(f"{model_name}: {e}")

            if results:
                st.markdown("#### 📊 Model Comparison")
                rdf=pd.DataFrame(results)
                st.dataframe(rdf,use_container_width=True,hide_index=True)
                fig=go.Figure(go.Bar(x=rdf["Model"],y=rdf["R²"],
                    marker_color=[ORANGE,TEAL,PURPLE][:len(rdf)],
                    hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"))
                fig.update_layout(**BASE,height=300,bargap=0.3,
                    xaxis=dict(**AX),yaxis=dict(**AX,title="R² Score",range=[0,1]))
                st.plotly_chart(fig,use_container_width=True,key="aml_r2_chart")

            if feat_imps:
                st.markdown("#### 🔍 Feature Importance")
                for model_nm,imps in feat_imps.items():
                    fidf=pd.DataFrame({"Feature":list(imps.keys()),
                                       "Importance":[abs(v) for v in imps.values()]})
                    fidf=fidf.sort_values("Importance",ascending=True)
                    fig=go.Figure(go.Bar(x=fidf["Importance"],y=fidf["Feature"],
                        orientation="h",
                        marker=dict(color=fidf["Importance"],colorscale=[[0,CARD_BG],[1,ORANGE]]),
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
                    fig.update_layout(**BASE,title=dict(text=f"{model_nm} Feature Importance",
                        font_size=13),height=max(250,len(feature_cols)*35),
                        xaxis=dict(**AX),yaxis=dict(**AX))
                    st.plotly_chart(fig,use_container_width=True,key=f"aml_fi_{model_nm}")

            if is_reg and len(aml_models)>0:
                try:
                    m_rf=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
                    m_rf.fit(X_tr_s,y_tr); preds_rf=m_rf.predict(X_te_s)
                    st.markdown("#### 📈 Actual vs Predicted (Random Forest)")
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=y_te,y=preds_rf,mode="markers",
                        marker=dict(color=ORANGE,size=5,opacity=0.6),name="Predicted vs Actual",
                        hovertemplate="Actual: %{x:,.2f}<br>Predicted: %{y:,.2f}<extra></extra>"))
                    mn_v,mx_v=min(y_te.min(),preds_rf.min()),max(y_te.max(),preds_rf.max())
                    fig.add_trace(go.Scatter(x=[mn_v,mx_v],y=[mn_v,mx_v],
                        mode="lines",line=dict(color=TEAL,width=1.5,dash="dash"),name="Perfect fit"))
                    fig.update_layout(**BASE,height=360,xaxis=dict(**AX,title="Actual"),
                        yaxis=dict(**AX,title="Predicted"),
                        legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig,use_container_width=True,key="aml_actual_pred")
                except: pass

# ═══════════════════════════════════════════════════════════════
# TAB 20 — NEW: REPORT BUILDER
# ═══════════════════════════════════════════════════════════════
with tabs[19]:
    st.markdown('<div class="tab-header">📋 Smart Report Builder</div>',unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Select sections → generate a complete HTML report you can save or print.</div>',unsafe_allow_html=True)
    num_avail=[c for c in num_cols_detected if c in df.columns]

    st.markdown("#### 🗂️ Select report sections")
    rb1,rb2=st.columns(2)
    with rb1:
        inc_summary=st.checkbox("Executive Summary",value=True,key="rb_summary")
        inc_kpi=st.checkbox("KPI Overview",value=True,key="rb_kpi")
        inc_stats=st.checkbox("Statistical Summary",value=True,key="rb_stats")
        inc_top=st.checkbox("Top / Bottom Analysis",value=True,key="rb_top")
    with rb2:
        inc_corr=st.checkbox("Correlation Table",value=True,key="rb_corr")
        inc_missing=st.checkbox("Data Quality",value=True,key="rb_missing")
        inc_ml=st.checkbox("ML Recommendations",value=True,key="rb_ml")
        inc_sample=st.checkbox("Sample Data (top 20 rows)",value=False,key="rb_sample")

    report_title=st.text_input("Report title","Analytics Report",key="rb_title")
    report_author=st.text_input("Author / Organisation","",key="rb_author")

    if st.button("📋 Build HTML Report",key="build_report",type="primary"):
        sections=[]
        ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        css=f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'DM Sans',sans-serif;background:#0F1117;color:#E5E7EB;padding:32px;}}
h1{{font-size:28px;font-weight:700;background:linear-gradient(90deg,#FF6B35,#2EC4B6);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;}}
h2{{font-size:18px;font-weight:600;color:#F9FAFB;margin:24px 0 12px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:6px;}}
h3{{font-size:14px;font-weight:600;color:#9CA3AF;margin:16px 0 8px;text-transform:uppercase;letter-spacing:.08em;}}
.meta{{font-size:12px;color:#6B7280;margin-bottom:28px;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin:12px 0;}}
.kpi{{background:#1A1D27;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
      padding:16px;border-top:3px solid #FF6B35;}}
.kpi-l{{font-size:10px;color:#6B7280;text-transform:uppercase;letter-spacing:.1em;}}
.kpi-v{{font-size:22px;font-weight:700;color:#F9FAFB;font-family:'DM Mono',monospace;margin:4px 0;}}
.kpi-s{{font-size:11px;color:#9CA3AF;}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px;}}
th{{background:#1A1D27;color:#9CA3AF;font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;padding:8px 12px;text-align:left;}}
td{{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.04);color:#E5E7EB;}}
tr:nth-child(even){{background:rgba(255,255,255,0.02);}}
.insight{{background:#1A1D27;border-left:3px solid #FF6B35;border-radius:0 8px 8px 0;
          padding:10px 14px;margin:6px 0;}}
.insight-t{{font-size:13px;font-weight:600;color:#F9FAFB;}}
.insight-d{{font-size:12px;color:#9CA3AF;margin-top:2px;}}
.footer{{margin-top:40px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.08);
         font-size:11px;color:#4B5563;text-align:center;}}
</style>"""
        html=[f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{report_title}</title>{css}</head><body>"]
        html.append(f"<h1>{report_title}</h1>")
        html.append(f"<div class='meta'>Generated: {ts}{' · ' + report_author if report_author else ''} · Source: {source_name} · {len(df):,} rows</div>")

        if inc_summary:
            html.append("<h2>Executive Summary</h2>")
            ins_html="".join([f"<div class='insight'><div class='insight-t'>{i['icon']} {i['title']}</div><div class='insight-d'>{i['text']}</div></div>" for i in insights])
            html.append(ins_html)

        if inc_kpi:
            html.append("<h2>KPI Overview</h2><div class='kpi-grid'>")
            for km in kpi_metrics[:5]:
                v=df[km].dropna()
                html.append(f"<div class='kpi'><div class='kpi-l'>{km}</div><div class='kpi-v'>{fmt(v.sum())}</div><div class='kpi-s'>Avg {fmt(v.mean())} · σ {fmt(v.std())}</div></div>")
            html.append("</div>")

        if inc_stats and num_avail:
            html.append("<h2>Statistical Summary</h2>")
            desc=df[num_avail[:6]].describe().round(3)
            html.append("<table><thead><tr><th>Statistic</th>"+
                "".join([f"<th>{c}</th>" for c in desc.columns])+"</tr></thead><tbody>")
            for idx,row in desc.iterrows():
                html.append("<tr><td>"+str(idx)+"</td>"+"".join([f"<td>{v}</td>" for v in row.values])+"</tr>")
            html.append("</tbody></table>")

        if inc_top and cat_cols:
            cc=cat_cols[0]
            html.append(f"<h2>Top 10 by {metric_col} ({cc})</h2>")
            top10=df.groupby(cc)[metric_col].sum().nlargest(10).reset_index()
            html.append("<table><thead><tr>"+"".join([f"<th>{c}</th>" for c in top10.columns])+"</tr></thead><tbody>")
            for _,row in top10.iterrows():
                html.append("<tr>"+"".join([f"<td>{v}</td>" for v in row.values])+"</tr>")
            html.append("</tbody></table>")

        if inc_corr and len(num_avail)>=2:
            html.append("<h2>Correlation Matrix</h2>")
            corr=df[num_avail[:6]].corr().round(3)
            html.append("<table><thead><tr><th></th>"+"".join([f"<th>{c}</th>" for c in corr.columns])+"</tr></thead><tbody>")
            for idx,row in corr.iterrows():
                html.append(f"<tr><td><b>{idx}</b></td>"+"".join([f"<td style='color:{'#34D399' if v>0.5 else '#EF4444' if v<-0.5 else '#E5E7EB'}'>{v}</td>" for v in row.values])+"</tr>")
            html.append("</tbody></table>")

        if inc_missing:
            html.append("<h2>Data Quality</h2>")
            null_info=pd.DataFrame({"Column":df.columns,
                "Missing":df.isnull().sum().values,
                "Missing %":(df.isnull().sum()/len(df)*100).round(1).values,
                "Type":df.dtypes.values})
            html.append("<table><thead><tr>"+"".join([f"<th>{c}</th>" for c in null_info.columns])+"</tr></thead><tbody>")
            for _,row in null_info.iterrows():
                html.append("<tr>"+"".join([f"<td>{v}</td>" for v in row.values])+"</tr>")
            html.append("</tbody></table>")

        if inc_ml:
            html.append("<h2>ML Recommendations</h2>")
            for r in recommend_ml(df,cols_info,metric_col):
                html.append(f"<div class='insight'><div class='insight-t'>🤖 {r['name']} [{r['badge']}]</div><div class='insight-d'>{r['desc']} — {r['why']}</div></div>")

        if inc_sample:
            html.append("<h2>Sample Data (first 20 rows)</h2>")
            sample=df.head(20)
            html.append("<table><thead><tr>"+"".join([f"<th>{c}</th>" for c in sample.columns])+"</tr></thead><tbody>")
            for _,row in sample.iterrows():
                html.append("<tr>"+"".join([f"<td>{v}</td>" for v in row.values])+"</tr>")
            html.append("</tbody></table>")

        html.append(f"<div class='footer'>Analytics Dashboard Ultra v3.0 · {ts}</div></body></html>")
        final_html="".join(html)
        st.download_button("⬇️ Download HTML Report",data=final_html.encode("utf-8"),
            file_name="analytics_report.html",mime="text/html",use_container_width=True)
        st.success("✅ Report ready — click Download HTML Report above!")
        with st.expander("Preview report"):
            st.components.v1.html(final_html,height=600,scrolling=True)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
div()
st.markdown("""
<div style="text-align:center;font-size:11px;color:#4B5563;padding:8px 0;">
  Analytics Dashboard Ultra v3.0 · Built with Streamlit · Plotly · pandas · numpy · scipy · scikit-learn · Google Gemini
  <br>20 Tabs · 50+ Chart Types · Statistical Tests · Monte-Carlo · AutoML · Geo Maps · Cohort Analysis · Report Builder
</div>
""", unsafe_allow_html=True)
