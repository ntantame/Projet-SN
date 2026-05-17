import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
import itertools

st.set_page_config(page_title="Orange Analytics", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

# styles 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── PALETTE : orange = accent uniquement ── */
:root {
  /* Accent orange — utilisé avec parcimonie */
  --accent:    #FF6B00;
  --accent-lt: #FF9A3C;

  /* Neutres light mode */
  --surf:      #F8F9FA;
  --card-bg:   #FFFFFF;
  --card-txt:  #1C1C1E;
  --card-sub:  #6B7280;
  --card-ref:  #9CA3AF;
  --card-sep:  #E5E7EB;
  --bdr:       rgba(0,0,0,0.08);
  --corr-bg:   #FFFFFF;
  --corr-track:#E5E7EB;

  /* Titres & textes */
  --sec-c:    #111827;
  --secs-c:   #6B7280;
  --stat-c:   #6B7280;
  --interp-c: #374151;

  /* Sémantique */
  --green:    #16A34A;
  --red:      #DC2626;
  --blue:     #2563EB;
  --amber:    #D97706;

  /* Insight cards — fond très pâle, bordure colorée */
  --ins-gn-bg:#F0FDF4; --ins-gn-bd:rgba(22,163,74,.3);  --ins-gn-t:#15803D;
  --ins-am-bg:#FFFBEB; --ins-am-bd:rgba(217,119,6,.3);  --ins-am-t:#B45309;
  --ins-rd-bg:#FEF2F2; --ins-rd-bd:rgba(220,38,38,.3);  --ins-rd-t:#B91C1C;
  --ins-bl-bg:#EFF6FF; --ins-bl-bd:rgba(37,99,235,.3);  --ins-bl-t:#1D4ED8;
  --ins-tx:   #374151;

  /* Badges */
  --bdg-s-bg:#DCFCE7; --bdg-s-c:#15803D;
  --bdg-n-bg:#F1F5F9; --bdg-n-c:#64748B;
  --bdg-g-bg:#FEF9C3; --bdg-g-c:#A16207;

  /* Plotly dark charts */
  --plt-bg:   #1E1E1E;
  --plt-paper:#252525;
}

/* ── DARK MODE ── */
@media (prefers-color-scheme: dark) {
  :root {
    --surf:      #0F0F0F;
    --card-bg:   #1A1A1A;
    --card-txt:  #F3F4F6;
    --card-sub:  #9CA3AF;
    --card-ref:  #6B7280;
    --card-sep:  #2D2D2D;
    --bdr:       rgba(255,255,255,0.08);
    --corr-bg:   #1A1A1A;
    --corr-track:#2D2D2D;
    --sec-c:     #F3F4F6;
    --secs-c:    #9CA3AF;
    --stat-c:    #9CA3AF;
    --interp-c:  #D1D5DB;
    --ins-gn-bg:#052E16; --ins-gn-bd:rgba(34,197,94,.25); --ins-gn-t:#4ADE80;
    --ins-am-bg:#1C1007; --ins-am-bd:rgba(251,191,36,.25); --ins-am-t:#FCD34D;
    --ins-rd-bg:#1C0808; --ins-rd-bd:rgba(248,113,113,.25);--ins-rd-t:#FCA5A5;
    --ins-bl-bg:#0C1628; --ins-bl-bd:rgba(96,165,250,.25); --ins-bl-t:#93C5FD;
    --ins-tx:    #D1D5DB;
    --bdg-s-bg:#052E16; --bdg-s-c:#4ADE80;
    --bdg-n-bg:#1F2937; --bdg-n-c:#9CA3AF;
    --bdg-g-bg:#1C1007; --bdg-g-c:#FCD34D;
  }
  .stApp { background:#0F0F0F !important; background-image:none !important; }
  .sec   { color:var(--sec-c) !important; }
  .sec-s { color:var(--secs-c) !important; }
}

/* ── BASE ── */
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
.stApp{ background:var(--surf); }

/* ── SIDEBAR — seul endroit vraiment orange ── */
[data-testid="stSidebar"]{
  background:#111827 !important;
  border-right:1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] *{ color:rgba(255,255,255,.82)!important; }
[data-testid="stSidebar"] .stRadio label{
  padding:8px 12px; border-radius:8px; transition:.2s;
}
[data-testid="stSidebar"] .stRadio label:hover{
  background:rgba(255,107,0,.15)!important;
}

/* ── HERO — gradient neutre, touche orange en fin ── */
.hero{
  background:linear-gradient(135deg,#111827 0%,#1F2937 55%,#374151 100%);
  border-radius:16px; padding:40px 48px; margin-bottom:24px;
  position:relative; overflow:hidden;
  border-left:4px solid var(--accent);
}
.hero::after{
  content:''; position:absolute; inset:0;
  background:repeating-linear-gradient(-45deg,transparent,transparent 40px,
    rgba(255,255,255,.012) 40px,rgba(255,255,255,.012) 41px);
  pointer-events:none;
}
.hero-eye{
  font-size:.7rem; font-weight:600; letter-spacing:.2em;
  text-transform:uppercase; color:var(--accent-lt); margin:0 0 8px;
}
.hero h1{
  font-family:'Playfair Display',serif; font-size:2.4rem;
  font-weight:900; color:#FFFFFF; margin:0 0 10px; line-height:1.1;
}
.hero-sub{ font-size:.95rem; color:rgba(255,255,255,.6); margin:0; font-weight:300; max-width:540px; }

/* ── TITRES DE SECTION — ligne grise, seul point orange = le petit carré ── */
.sec{
  font-family:'Playfair Display',serif; font-size:1.25rem; font-weight:700;
  color:var(--sec-c); margin:28px 0 4px; padding-bottom:8px;
  border-bottom:1px solid var(--card-sep);
  display:flex; align-items:center; gap:8px;
}
.sec::before{
  content:''; display:inline-block; width:4px; height:20px;
  background:var(--accent); border-radius:2px; flex-shrink:0;
}
.sec-s{ font-size:.82rem; color:var(--secs-c); margin:0 0 16px; font-style:italic; }

/* ── KPI CARDS — neutre, barre de couleur en haut ── */
.kpi{
  background:var(--card-bg); border-radius:12px; padding:18px 20px;
  border:1px solid var(--bdr);
  box-shadow:0 1px 4px rgba(0,0,0,.06);
  position:relative; overflow:hidden;
}
.kpi::before{
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:var(--card-sep);          /* gris par défaut */
}
.kpi.g::before{ background:var(--green); }
.kpi.r::before{ background:var(--red);   }
.kpi.b::before{ background:var(--blue);  }
.kpi.o::before{ background:var(--accent);}  /* orange seulement si classe .o */
.kpi-lbl{
  font-size:.72rem; font-weight:600; text-transform:uppercase;
  letter-spacing:.08em; color:var(--card-sub); margin:0 0 6px;
}
.kpi-val{
  font-family:'Playfair Display',serif; font-size:1.75rem; font-weight:700;
  color:var(--card-txt); margin:0 0 4px; line-height:1;
}
.kpi-val span{
  font-size:.88rem; font-weight:400; color:var(--card-sub);
  font-family:'DM Sans',sans-serif;
}
.kpi-ref{
  font-size:.73rem; color:var(--card-ref); margin:6px 0 0;
  border-top:1px solid var(--card-sep); padding-top:5px; line-height:1.5;
}

/* ── CARDS (résultats tests) ── */
.card{
  background:var(--card-bg); border-radius:12px; padding:18px 22px;
  border:1px solid var(--bdr); box-shadow:0 1px 3px rgba(0,0,0,.05);
  margin-bottom:12px;
}
.card-hd{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.card-title{ font-weight:600; font-size:.92rem; color:var(--card-txt); }
.bdg{
  padding:3px 9px; border-radius:20px; font-size:.68rem;
  font-weight:700; letter-spacing:.06em; text-transform:uppercase;
}
.bdg.s{ background:var(--bdg-s-bg); color:var(--bdg-s-c); }
.bdg.n{ background:var(--bdg-n-bg); color:var(--bdg-n-c); }
.bdg.g{ background:var(--bdg-g-bg); color:var(--bdg-g-c); }
.stat-vals{
  font-size:.81rem; color:var(--stat-c); margin:0 0 8px;
  font-variant-numeric:tabular-nums;
}
.interp{
  font-size:.86rem; color:var(--interp-c); line-height:1.7;
  border-left:3px solid var(--card-sep); padding-left:12px; margin:10px 0 0;
}

/* ── INSIGHT CARDS ── */
.ins{
  border-radius:12px; padding:15px 18px; margin:10px 0;
  display:flex; gap:14px; align-items:flex-start; border:1px solid;
}
.ins.gn{ background:var(--ins-gn-bg); border-color:var(--ins-gn-bd); }
.ins.am{ background:var(--ins-am-bg); border-color:var(--ins-am-bd); }
.ins.rd{ background:var(--ins-rd-bg); border-color:var(--ins-rd-bd); }
.ins.bl{ background:var(--ins-bl-bg); border-color:var(--ins-bl-bd); }
.ins-ic{ font-size:1.3rem; flex-shrink:0; margin-top:2px; }
.ins-t{ font-weight:600; font-size:.87rem; margin:0 0 4px; }
.ins.gn .ins-t{ color:var(--ins-gn-t); }
.ins.am .ins-t{ color:var(--ins-am-t); }
.ins.rd .ins-t{ color:var(--ins-rd-t); }
.ins.bl .ins-t{ color:var(--ins-bl-t); }
.ins-tx{ font-size:.84rem; line-height:1.65; margin:0; color:var(--ins-tx); }

/* ── CORRELATION ROWS ── */
.corr-row{
  display:flex; align-items:center; gap:10px; padding:9px 13px;
  border-radius:8px; background:var(--corr-bg);
  border:1px solid var(--bdr); margin-bottom:7px;
}
.corr-v{ font-weight:500; font-size:.87rem; color:var(--card-txt); flex:1; }
.corr-r{
  font-family:'Playfair Display',serif; font-size:1rem;
  font-weight:700; min-width:46px; text-align:right;
}
.corr-r.p{ color:var(--green); }
.corr-r.n{ color:var(--red);   }
@media (prefers-color-scheme: dark) {
  .corr-r.p{ color:#4ADE80; }
  .corr-r.n{ color:#FCA5A5; }
}
.corr-bw{ flex:2; height:6px; background:var(--corr-track); border-radius:3px; overflow:hidden; }
.corr-b{ height:100%; border-radius:3px; }

/* ── SIDEBAR LOGO ── */
.sbl{
  font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:900;
  color:white!important;padding-bottom:14px;border-bottom:1px solid rgba(255,107,0,.3);margin-bottom:14px;}
</style>
""", unsafe_allow_html=True)

# data
@st.cache_data
def load():
    dataOrange = pd.read_csv("orange.csv")
    dataOrange = dataOrange.rename(columns={
        'Size (cm)':'Size(cm)','Weight (g)':'Weight(g)',
        'Brix (Sweetness)':'Brix(Sweetness)','pH (Acidity)':'pH(Acidity)',
        'Softness (1-5)':'Softness(1-5)','HarvestTime (days)':'HarvestTime(days)',
        'Ripeness (1-5)':'Ripeness(1-5)','Blemishes (Y/N)':'Blemishes(Y/N)',
        'Quality (1-5)':'Quality(1-5)'
    })
    dataOrange['sans_defaut'] = dataOrange['Blemishes(Y/N)'] == 'N'
    return dataOrange

dataOrange = load()

# Valeurs de référence globales (dataset complet, avant filtres)
REF_QUAL   = dataOrange['Quality(1-5)'].mean()
REF_BRIX   = dataOrange['Brix(Sweetness)'].mean()
REF_PH     = dataOrange['pH(Acidity)'].mean()
REF_RIPE   = dataOrange['Ripeness(1-5)'].mean()
REF_WEIGHT = dataOrange['Weight(g)'].mean()
REF_HARV   = dataOrange['HarvestTime(days)'].mean()
REF_SOFT   = dataOrange['Softness(1-5)'].mean()
REF_SIZE   = dataOrange['Size(cm)'].mean()

NUM = ['Size(cm)','Weight(g)','Brix(Sweetness)','pH(Acidity)',
       'Softness(1-5)','HarvestTime(days)','Ripeness(1-5)','Quality(1-5)']
NLBL = {'Size(cm)':'Taille (cm)','Weight(g)':'Poids (g)',
        'Brix(Sweetness)':'Sucrosité (Brix)','pH(Acidity)':'pH / Acidité',
        'Softness(1-5)':'Mollesse','HarvestTime(days)':'Temps récolte (j)',
        'Ripeness(1-5)':'Maturité','Quality(1-5)':'Qualité'}
BG        = "#1E1E1E"
PLT_BG    = "#252525"
PLT_TXT   = "#F0F0F0"
PLT_SUB   = "#BBBBBB"
PLT_ANN   = "#FFD580"
PLT_WARN  = "#FF7070"
PLT_GRID  = "rgba(255,255,255,0.07)"
OSEQ      = ['#FF6B00','#FF8C3C','#FFA866','#FFB47F','#FFCA99']

def plt_base(height=None, title_size=13):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG,
        font=dict(family="DM Sans", color=PLT_TXT),
        title_font=dict(size=title_size, color=PLT_TXT),
        xaxis=dict(gridcolor=PLT_GRID, linecolor="rgba(255,255,255,0.15)",
                   tickfont=dict(color=PLT_SUB), title_font=dict(color=PLT_SUB),
                   zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor=PLT_GRID, linecolor="rgba(255,255,255,0.15)",
                   tickfont=dict(color=PLT_SUB), title_font=dict(color=PLT_SUB),
                   zerolinecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", font=dict(color=PLT_TXT),
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    if height: d['height'] = height
    return d

# sidebar 
with st.sidebar:
    st.markdown('<div class="sbl"> Orange Analytics</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "  Accueil",
        "  Chiffres clés du lot",
        "  Répartition des oranges",
        "  Couleurs, Variétés et Défauts",
        "  Facteurs influençant la qualité",
        "  Liens entre les catégories",
        "  Variation selon les caractéristiques",
        "  Ce qui fait une bonne orange",
        "  Résultats et Recommandations",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Filtres**")
    sel_color   = st.multiselect("Couleur", sorted(dataOrange['Color'].unique()),   default=sorted(dataOrange['Color'].unique()))
    sel_variety = st.multiselect("Variété", sorted(dataOrange['Variety'].unique()), default=sorted(dataOrange['Variety'].unique()))
    q_min,q_max = st.slider("Qualité",1.0,5.0,(1.0,5.0),0.5)
    st.markdown("---")
    st.caption("TP Data Science - EDA Orange Dataset")

fdf = dataOrange[dataOrange['Color'].isin(sel_color) & dataOrange['Variety'].isin(sel_variety) & dataOrange['Quality(1-5)'].between(q_min,q_max)].copy()

corr_mat     = fdf[NUM].corr()
corr_quality = corr_mat['Quality(1-5)'].drop('Quality(1-5)').sort_values(ascending=False)
qual_moy     = fdf['Quality(1-5)'].mean()
pct_hq       = (fdf['Quality(1-5)']>=4).mean()*100
pct_lq       = (fdf['Quality(1-5)']<=2).mean()*100
pct_sd       = fdf['sans_defaut'].mean()*100


# VUE D'ENSEMBLE

if "Accueil" in page:
    st.markdown("""<div class="hero">
      <p class="hero-eye">TP Data Science - Analyse Exploratoire</p>
      <h1>Orange Analytics</h1>
      <p class="hero-sub">Analyse des caractéristiques physico-chimiques des oranges pour identifier les facteurs déterminant leur qualité: 241 fruits et 11 variables.<br> EDA complet</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">En un coup d\'oeil</div><p class="sec-s">Les chiffres les plus importants pour juger la qualité de ce lot d\'oranges</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Note de qualité moyenne</p><p class="kpi-val">{qual_moy:.2f}<span> sur 5</span></p><p class="kpi-ref">Note moyenne sur l\'ensemble du lot</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi g"><p class="kpi-lbl"> Oranges bien notées</p><p class="kpi-val">{pct_hq:.1f}<span>%</span></p><p class="kpi-ref">{int(pct_hq/100*len(fdf))} oranges sur {len(fdf)} ont une note de 4 ou 5 sur 5</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi r"><p class="kpi-lbl"> Oranges à écarter</p><p class="kpi-val">{pct_lq:.1f}<span>%</span></p><p class="kpi-ref">{int(pct_lq/100*len(fdf))} oranges jugées de mauvaise qualité (note 1 ou 2)</p></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Oranges sans tache ni marque</p><p class="kpi-val">{pct_sd:.1f}<span>%</span></p><p class="kpi-ref">{fdf["sans_defaut"].sum()} oranges d\'apparence parfaite sur {len(fdf)}</p></div>', unsafe_allow_html=True)

    c5,c6,c7,c8 = st.columns(4)
    with c5: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Douceur moyenne</p><p class="kpi-val">{fdf["Brix(Sweetness)"].mean():.1f}</p><p class="kpi-ref">Plus ce chiffre est élevé, plus l\'orange est sucrée et bonne</p></div>', unsafe_allow_html=True)
    with c6: st.markdown(f'<div class="kpi r"><p class="kpi-lbl"> Jours écoulés depuis la récolte</p><p class="kpi-val">{fdf["HarvestTime(days)"].mean():.0f}<span> jours</span></p><p class="kpi-ref">Plus l\'orange attend, moins elle est bonne.à surveiller</p></div>', unsafe_allow_html=True)
    with c7: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Niveau de maturité moyen</p><p class="kpi-val">{fdf["Ripeness(1-5)"].mean():.2f}<span> sur 5</span></p><p class="kpi-ref">Les fruits bien mûrs sont systématiquement mieux notés</p></div>', unsafe_allow_html=True)
    with c8: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Niveau d\'acidité moyen</p><p class="kpi-val">{fdf["pH(Acidity)"].mean():.2f}</p><p class="kpi-ref">Toutes les oranges ont une acidité très proche : c\'est un bon signe</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Aperçu des données</div>', unsafe_allow_html=True)
    st.dataframe(fdf.drop(columns=['sans_defaut']).head(10), use_container_width=True, height=320)


# STATISTIQUES DESCRIPTIVES

elif "Chiffres" in page:
    st.markdown('<div class="sec"> Statistiques Descriptives</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-s">Section 3 du notebook</div>', unsafe_allow_html=True)
    stats = fdf[NUM].describe().T.round(4).reset_index().rename(columns={"index":"Variable"})
    st.dataframe(stats, use_container_width=True, height=340)

    st.markdown("""<div class="ins bl"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation: Statistiques descriptives</p>
      <p class="ins-tx">Sur les 241 oranges analysées, les résultats sont globalement encourageants. La qualité se révèle assez homogène, avec une note moyenne de 3,8 sur 5 ; la plupart des fruits dépassent d'ailleurs la barre des 4. Côté profil gustatif, on est sur des oranges bien sucrées, avec un indice Brix autour de 10,9, et une acidité tout à fait classique pour la variété (pH de 3,47 en moyenne).<br><br>Les gabarits varient pas mal d'un fruit à l'autre  entre 100 g et 300 g, ce qui traduit une certaine diversité physique dans l'échantillon. Enfin, la maturité est au rendez-vous : les fruits sont bien mûrs en moyenne (3,6/5), récoltés environ 15 jours après le pic de maturité.</p>
    </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Boxplots par couleur</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">Chaque boîte montre la médiane (trait central), le 1er et 3e quartile (bords de la boîte), et les valeurs extrêmes (moustaches)</p>', unsafe_allow_html=True)
    sel = st.selectbox("Variable à analyser", NUM, format_func=lambda x: NLBL[x], index=7)
    fig = px.box(fdf, y=sel, color="Color", color_discrete_sequence=OSEQ,
                 title=f"Distribution de « {NLBL[sel]} » selon la couleur de l'orange",
                 labels={sel: NLBL[sel], "Color": "Couleur"},
                 points="outliers")
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
        yaxis_title=f"{NLBL[sel]}",
        xaxis_title="Couleur de l'orange",
        legend_title="Couleur",
        title_font_size=14,
        annotations=[dict(
            x=1.0, y=1.08, xref="paper", yref="paper",
            text="💡 Trait central = médiane · Bords de boîte = Q1/Q3 · Points = valeurs extrêmes",
            showarrow=False, font=dict(size=10, color=PLT_ANN),
            xanchor="right"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)


# DISTRIBUTIONS NUMÉRIQUES

elif "Répartition" in page:
    st.markdown('<div class="sec"> Distribution des Variables Numériques</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Analyse univariée — Section 4 du notebook</div>', unsafe_allow_html=True)

    # Vue globale identique au notebook (4×2)
    st.markdown('<p class="sec-s">Chaque histogramme montre combien d\'oranges tombent dans chaque intervalle de valeurs · <span style="color:#C0392B">──</span> Moyenne &nbsp;·&nbsp; <span style="color:#1A5F8A">···</span> Médiane</p>', unsafe_allow_html=True)
    fig = make_subplots(rows=4, cols=2, subplot_titles=[NLBL[c] for c in NUM],
                        vertical_spacing=0.08, horizontal_spacing=0.1)
    pal = ['#FF6B00','#FF8C3C','#FFA866','#FFB47F','#FF9A3C','#E05800','#CC4A00','#FF7A1A']
    for i, col in enumerate(NUM):
        r, c = i//2+1, i%2+1
        d = fdf[col].dropna()
        fig.add_trace(go.Histogram(
            x=d, marker_color=pal[i], nbinsx=20,
            name=NLBL[col], showlegend=False,
            marker_line_color="white", marker_line_width=0.8,
            hovertemplate=f"<b>{NLBL[col]}</b><br>Valeur : %{{x}}<br>Nombre d'oranges : %{{y}}<extra></extra>"
        ), row=r, col=c)
        fig.add_vline(x=d.mean(), line_dash="dash", line_color="red", line_width=1.5, row=r, col=c)
        fig.add_vline(x=d.median(), line_dash="dot", line_color="#60AAFF", line_width=1.5, row=r, col=c)
    fig.update_layout(
        height=950, paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
        title_text="Vue d'ensemble — Distribution des 8 Variables Numériques",
        title_font_size=14,
        title_font_color=PLT_TXT
    )
    # Axes labels sur chaque sous-graphique
    for i, col in enumerate(NUM):
        r, c = i//2+1, i%2+1
        fig.update_xaxes(title_text=NLBL[col], title_font_size=10, row=r, col=c)
        fig.update_yaxes(title_text="Nb oranges", title_font_size=10, row=r, col=c)
    st.plotly_chart(fig, use_container_width=True)

    # Interprétation complète du notebook
    st.markdown("""<div class="ins bl"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation : Distribution des Variables Numériques</p>
      <p class="ins-tx">
        <strong>Qualité (Quality 1-5)</strong> : La qualité est clairement orientée vers le haut : la grande majorité des oranges obtiennent une note de 4 ou 5, avec une moyenne de 3,82. Les fruits de mauvaise qualité (note 1 ou 2) sont vraiment l'exception. C'est le signe d'un lot globalement sain et bien sélectionné.<br><br>
        <strong>Maturité (Ripeness 1-5)</strong> : La maturité suit la même tendance : les notes 4 et 5 dominent largement, même si on note une présence non négligeable de fruits à 1 et 2. Cela suggère que la récolte mélange des fruits bien mûrs et quelques fruits cueillis trop tôt ou trop tard.<br><br>
        <strong>Douceur (Brix/Sweetness)</strong> : C'est l'observation la plus intéressante du graphique : la distribution présente <strong>deux bosses distinctes</strong>, l'une autour de 8 et l'autre autour de 14. Cela ne ressemble pas à un hasard, on a probablement deux groupes de fruits différents, que ce soit deux variétés d'oranges, deux origines géographiques, ou deux stades de maturité bien distincts. C'est une piste qui mérite d'être creusée.<br><br>
        <strong>Acidité (pH)</strong> : Le pH est très concentré entre 3,2 et 3,6, avec une moyenne de 3,47. Les oranges sont remarquablement homogènes en termes d'acidité. Tout fruit qui sort de cette fourchette mérite attention : anomalie de mesure, variété atypique, ou fruit en mauvais état.<br><br>
        <strong>Texture (Softness 1-5)</strong> : Distribution très particulière : les valeurs se concentrent presque exclusivement sur 1, 3 et 5, avec très peu de valeurs intermédiaires. Cela suggère que la texture a été évaluée de façon subjective et catégorielle plutôt que mesurée précisément. Les évaluateurs ont tranché : une orange est soit ferme, soit moyenne, soit molle.<br><br>
        <strong>Taille et Poids</strong> : Les deux distributions sont relativement étalées (6 à 10 cm · 100 à 300 g). Rien d'anormal: c'est simplement le reflet de la variabilité naturelle des fruits.<br><br>
        <strong>Temps de récolte (HarvestTime)</strong> : La distribution n'est pas uniforme. On observe un pic notable autour de 12-13 jours, puis les valeurs se répartissent plus librement jusqu'à 25 jours. L'échantillon couvre différents stades post-récolte, avec une légère surreprésentation des fruits récoltés à mi-parcours.<br><br>
        <em>En résumé : cet échantillon décrit globalement des oranges de bonne qualité et bien mûres, avec une acidité très stable. Le point le plus intrigant reste la douceur, dont la distribution bimodale soulève une vraie question sur la composition du lot.</em>
      </p>
    </div></div>""", unsafe_allow_html=True)

    # Détail interactif
    st.markdown('<div class="sec">Analyse individuelle interactive</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">Sélectionnez une variable pour voir son histogramme détaillé et son violin plot</p>', unsafe_allow_html=True)
    col_choice = st.selectbox("Choisir une variable", NUM, format_func=lambda x: NLBL[x])
    c1, c2 = st.columns(2)
    with c1:
        d = fdf[col_choice].dropna()
        fig = px.histogram(fdf, x=col_choice, nbins=25, color_discrete_sequence=['#FF6B00'],
                           title=f"Histogramme — {NLBL[col_choice]}",
                           labels={col_choice: NLBL[col_choice], "count": "Nombre d'oranges"},
                           marginal="box")
        fig.add_vline(x=d.mean(), line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Moy. = {d.mean():.2f}", annotation_position="top right",
                      annotation_font_color=PLT_WARN, annotation_font_size=11)
        fig.add_vline(x=d.median(), line_dash="dot", line_color="#60AAFF", line_width=2,
                      annotation_text=f"Méd. = {d.median():.2f}", annotation_position="top left",
                      annotation_font_color="#60AAFF", annotation_font_size=11)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
            xaxis_title=NLBL[col_choice],
            yaxis_title="Nombre d'oranges",
            title_font_size=13,
            annotations=[dict(
                x=0.5, y=-0.22, xref="paper", yref="paper",
                text=f" Chaque barre = intervalle de valeurs · Hauteur = nombre d'oranges dans cet intervalle",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(fdf, y=col_choice, box=True, points="outliers",
                        color_discrete_sequence=['#FF8C3C'],
                        title=f"Violin Plot — {NLBL[col_choice]}",
                        labels={col_choice: NLBL[col_choice]})
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
            yaxis_title=NLBL[col_choice],
            xaxis_title="",
            title_font_size=13,
            annotations=[dict(
                x=0.5, y=-0.18, xref="paper", yref="paper",
                text="🎻 Largeur = densité de fruits à cette valeur · Boîte intérieure = Q1/médiane/Q3 · Points = valeurs extrêmes",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)


# VARIABLES CATÉGORIELLES

elif "Couleurs" in page:
    st.markdown('<div class="sec"> Analyse des Variables Catégorielles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Analyse univariée — Section 5 du notebook</div>', unsafe_allow_html=True)

    # Graphique 1 : Distribution des Couleurs (bar)
    st.markdown('<div class="sec-s">Graphique 1 — Répartition des oranges selon leur couleur</div>', unsafe_allow_html=True)
    color_counts = fdf['Color'].value_counts().reset_index()
    color_counts.columns = ['Color','Nombre']
    avg_line = color_counts['Nombre'].mean()
    fig = px.bar(color_counts, x='Color', y='Nombre',
                 color='Color',
                 color_discrete_sequence=['#CC3300','#FF6B00','#FF9A3C','#FFB866','#FFD700'],
                 title="Nombre d'oranges par couleur — Répartition de la variable Color",
                 labels={'Color': 'Couleur de l\'orange', 'Nombre': 'Nombre d\'oranges'},
                 text='Nombre')
    fig.add_hline(y=avg_line, line_dash="dash", line_color="rgba(255,255,255,0.35)", line_width=1.5,
                  annotation_text=f"Moyenne : {avg_line:.0f} oranges/couleur",
                  annotation_position="top right", annotation_font_size=10)
    fig.update_traces(textposition='outside', textfont_size=12)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG, showlegend=False,
        font_family="DM Sans", height=400,
        xaxis_title="Couleur de l'orange",
        yaxis_title="Nombre d'oranges dans le lot",
        title_font_size=13,
        annotations=[dict(
            x=0.5, y=-0.2, xref="paper", yref="paper",
            text=" Lecture : chaque barre = une couleur · Hauteur = nombre d'oranges de cette couleur dans le dataset",
            showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Graphique 2 : Distribution des Variétés (horizontal bar)
    st.markdown('<div class="sec-s">Graphique 2 — Répartition des oranges selon leur variété botanique</div>', unsafe_allow_html=True)
    var_counts = fdf['Variety'].value_counts().reset_index()
    var_counts.columns = ['Variety','Nombre']
    avg_var = var_counts['Nombre'].mean()
    fig = px.bar(var_counts, x='Nombre', y='Variety', orientation='h',
                 color='Nombre', color_continuous_scale=['#FFD580','#FF6B00','#CC3300'],
                 title="Nombre d'oranges par variété — Répartition de la variable Variety",
                 labels={'Variety': 'Variété botanique', 'Nombre': 'Nombre d\'oranges'},
                 text='Nombre')
    fig.add_vline(x=avg_var, line_dash="dash", line_color="rgba(255,255,255,0.35)", line_width=1.5,
                  annotation_text=f"Moy. : {avg_var:.0f}", annotation_position="top right",
                  annotation_font_size=10)
    fig.update_traces(textposition='outside', textfont_size=11)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
        yaxis={'categoryorder':'total ascending'},
        height=560, coloraxis_showscale=False,
        xaxis_title="Nombre d'oranges",
        yaxis_title="Variété d'orange",
        title_font_size=13,
        annotations=[dict(
            x=1.0, y=-0.08, xref="paper", yref="paper",
            text=" Lecture : chaque barre = une variété · Plus la barre est longue, plus la variété est représentée · Ligne pointillée = moyenne",
            showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="right"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Graphique 3 : Distribution des Défauts (pie + bar)
    st.markdown('<div class="sec-s">Graphique 3 — Présence et types de défauts visuels (variable Blemishes)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    blem_counts = fdf['Blemishes(Y/N)'].value_counts().reset_index()
    blem_counts.columns = ['Blemish','Nombre']

    with c1:
        san_def = pd.DataFrame({
            'Statut': ['Sans défaut (N)', 'Avec défaut'],
            'Nombre': [fdf['sans_defaut'].sum(), (~fdf['sans_defaut']).sum()]
        })
        fig = px.pie(san_def, names='Statut', values='Nombre',
                     color_discrete_sequence=['#2E7D4F','#C0392B'],
                     title="Proportion d'oranges sans défaut vs avec défaut",
                     hole=0.42)
        fig.update_traces(
            textinfo='percent+label+value',
            hovertemplate="<b>%{label}</b><br>Nombre : %{value}<br>Proportion : %{percent}<extra></extra>"
        )
        fig.update_layout(
            paper_bgcolor=BG, font_family="DM Sans", height=380,
            title_font_size=12,
            legend=dict(orientation="h", y=-0.15),
            annotations=[dict(
                x=0.5, y=0.5, text=f"{int(pct_sd)}%<br>sains",
                showarrow=False, font=dict(size=14, color="#5BE89A", family="Playfair Display")
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(blem_counts, x='Nombre', y='Blemish', orientation='h',
                     color='Nombre', color_continuous_scale=['#FFD580','#FF6B00','#CC3300'],
                     title="Détail de chaque type de défaut recensé",
                     labels={'Blemish': 'Type de défaut', 'Nombre': 'Nombre d\'oranges concernées'},
                     text='Nombre')
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
            yaxis={'categoryorder':'total ascending'}, height=380,
            coloraxis_showscale=False,
            xaxis_title="Nombre d'oranges avec ce défaut",
            yaxis_title="Type de défaut (code Blemishes)",
            title_font_size=12,
            annotations=[dict(
                x=1.0, y=-0.12, xref="paper", yref="paper",
                text="📌 N = aucun défaut · Codes = types de défauts spécifiques (soleil, cicatrice, moisissure…)",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="right"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Graphique 4 : Interprétation du notebook
    st.markdown("""<div class="ins gn"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation : Variables Catégorielles</p>
      <p class="ins-tx">Les oranges de cet échantillon présentent un profil globalement très sain. Côté couleur, les teintes foncées (Deep Orange, Light Orange, Orange-Red) dominent largement, signe d'une bonne maturité générale, tandis que le Yellow-Orange, caractéristique des fruits jeunes, reste anecdotique.<br><br>
      La diversité variétale est notable : plusieurs variétés sont représentées avec des effectifs assez équilibrés, la Cara Cara prenant légèrement la tête devant la Star Ruby et la Temple. Aucune variété n'écrase les autres, ce qui donne à cet échantillon une bonne représentativité.<br><br>
      Enfin, les défauts sont l'exception plutôt que la règle : 149 fruits (61,8%) sont parfaitement sains, et quand un défaut apparaît, il est le plus souvent superficiel, coup de soleil, cicatrice sans impact réel sur la qualité du fruit.</p>
    </div></div>""", unsafe_allow_html=True)


# CORRÉLATIONS

elif "Facteurs" in page:
    st.markdown('<div class="sec"> Matrice de Corrélation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Section 6.1 du notebook — Variables numériques</div>', unsafe_allow_html=True)

    # Heatmap triangulaire (identique au notebook)
    st.markdown('<p class="sec-s">Chaque cellule = coefficient de corrélation r entre deux variables ·  rouge = corrélation positive ·  bleu = corrélation négative · Plus la couleur est intense, plus le lien est fort</p>', unsafe_allow_html=True)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    z_masked = corr_mat.copy().values.astype(float)
    z_masked[mask] = np.nan
    text_masked = corr_mat.round(2).values.astype(str)
    text_masked[mask] = ""

    fig = go.Figure(go.Heatmap(
        z=z_masked, x=[NLBL[c] for c in corr_mat.columns],
        y=[NLBL[c] for c in corr_mat.columns],
        colorscale=[[0,"#1A5F8A"],[0.5,"white"],[1,"#C0392B"]],
        zmid=0, text=text_masked, texttemplate="%{text}",
        textfont={"size":11}, zmin=-1, zmax=1,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Corrélation r = %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="r",
            tickvals=[-1,-0.5,0,0.5,1],
            ticktext=["−1 (opposé)","−0.5","0 (aucun)","0.5","+1 (parfait)"],
            lenmode="fraction", len=0.8
        )
    ))
    fig.update_layout(
        title="Matrice de Corrélation — Toutes les Variables Numériques<br><sup>Triangle inférieur uniquement · r proche de 0 = pas de lien · r proche de ±1 = lien fort</sup>",
        height=540, paper_bgcolor=BG, font_family="DM Sans",
        xaxis_tickangle=-35,
        xaxis_title="",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    # Corrélations avec la qualité
    st.markdown('<div class="sec-s">Graphique de droite : Classement des variables par force de corrélation avec la Qualité</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Corrélations avec la Qualité (triées)**")
        st.markdown('<p style="font-size:.78rem;color:#FFB866"> vert = corrélation positive (hausse ensemble) ·  rouge = corrélation négative (l\'un monte, l\'autre baisse)</p>', unsafe_allow_html=True)
        for var, r in corr_quality.items():
            bc = "#2E7D4F" if r > 0 else "#C0392B"
            cls = "p" if r > 0 else "n"
            st.markdown(f"""<div class="corr-row">
              <span class="corr-v">{NLBL[var]}</span>
              <div class="corr-bw"><div class="corr-b" style="width:{abs(r)*100:.0f}%;background:{bc}"></div></div>
              <span class="corr-r {cls}">{r:+.2f}</span>
            </div>""", unsafe_allow_html=True)
    with c2:
        cq_df = corr_quality.reset_index()
        cq_df.columns = ['Variable','r']
        cq_df['label'] = cq_df['Variable'].map(NLBL)
        cq_df['couleur'] = cq_df['r'].apply(lambda x: '#2E7D4F' if x > 0 else '#C0392B')
        fig = go.Figure(go.Bar(
            x=cq_df['r'], y=cq_df['label'], orientation='h',
            marker_color=cq_df['couleur'],
            text=cq_df['r'].round(2), textposition='outside',
            hovertemplate="<b>%{y}</b><br>Corrélation avec Qualité : r = %{x:.2f}<extra></extra>"
        ))
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_width=1.5)
        fig.update_layout(title="Corrélation de chaque variable avec la Qualité",
                          xaxis_title="Coefficient r", paper_bgcolor=BG,
                          plot_bgcolor=PLT_BG, font_family="DM Sans",
                          height=320, yaxis={'categoryorder':'array','categoryarray':cq_df['label'].tolist()})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="ins bl"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation : Matrice de corrélation</p>
      <p class="ins-tx">
        <strong>Brix (sucre) et Qualité (r = +0.63)</strong> : C'est le lien le plus fort de la matrice : une orange sucrée est presque systématiquement bien notée.<br><br>
        <strong>HarvestTime et Qualité (r = −0.47)</strong> : Plus une orange reste longtemps après récolte, plus sa qualité se dégrade. Le temps est l'ennemi de la fraîcheur.<br><br>
        <strong>Softness et Qualité (r = −0.32)</strong> : Un fruit qui ramollit perd en qualité, mais l'effet est plus modéré que le temps de stockage.
      </p>
    </div></div>""", unsafe_allow_html=True)

    # Scatter interactif
    st.markdown('<div class="sec">Nuage de points interactif</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">Choisissez deux variables pour visualiser leur relation · Chaque point = une orange · La droite de tendance indique la direction du lien</p>', unsafe_allow_html=True)
    cc1,cc2 = st.columns(2)
    xv = cc1.selectbox("Axe X", NUM, format_func=lambda x: NLBL[x], index=2)
    yv = cc2.selectbox("Axe Y", NUM, format_func=lambda x: NLBL[x], index=7)
    r_val = fdf[[xv, yv]].corr().iloc[0,1]
    direction = "positive" if r_val > 0 else "négative"
    force = "forte" if abs(r_val) > 0.5 else ("modérée" if abs(r_val) > 0.25 else "faible")
    st.markdown(f'<p style="font-size:.82rem;color:#FFB866;margin-bottom:8px"> Corrélation entre ces deux variables : <strong>r = {r_val:+.2f}</strong> — relation {direction} {force}</p>', unsafe_allow_html=True)
    fig = px.scatter(fdf, x=xv, y=yv, color="Color", color_discrete_sequence=OSEQ,
                     trendline="ols",
                     hover_data=["Variety","Quality(1-5)"],
                     title=f"Relation entre {NLBL[xv]} et {NLBL[yv]}<br><sup>Chaque point = une orange · Couleur = couleur de l'orange · Droite = tendance linéaire (OLS)</sup>",
                     labels={xv: NLBL[xv], yv: NLBL[yv], "Color": "Couleur"})
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
        xaxis_title=f"{NLBL[xv]}",
        yaxis_title=f"{NLBL[yv]}",
        legend_title="Couleur de l'orange",
        title_font_size=13,
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=f" Survolez un point pour voir les détails de l'orange · La droite de tendance montre la direction générale du lien",
            showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)


# TEST KHI-2

elif "Liens" in page:
    st.markdown('<div class="sec"> Test du Khi-2 — Variables Catégorielles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Section 6.2 du notebook — Relation entre variables catégorielles</div>', unsafe_allow_html=True)

    cat_cols = ['Color','Variety','Blemishes(Y/N)']
    khi2_data = []
    for v1, v2 in itertools.combinations(cat_cols, 2):
        ct = pd.crosstab(fdf[v1], fdf[v2])
        chi2, p, dof, _ = chi2_contingency(ct)
        khi2_data.append((v1, v2, chi2, p, dof))

    interps_khi2 = {
        ('Color','Variety'):
            ("Relation significative et forte. La couleur d'une orange n'est pas un hasard, elle est bien liée à sa variété. Certaines variétés produisent systématiquement des couleurs particulières, ce qui est biologiquement logique.",True),
        ('Color','Blemishes(Y/N)'):
            ("Pas de relation significative. La couleur d'une orange ne permet pas de prédire si elle aura des défauts ou non. Un fruit rouge-orangé n'est pas plus ou moins abîmé qu'un fruit jaune-orangé.",False),
        ('Variety','Blemishes(Y/N)'):
            ("Zone grise : la p-value frôle le seuil (0.080 et 0.05). On ne peut pas conclure formellement, mais ce n'est pas non plus une indépendance totale. Certaines variétés pourraient être légèrement plus sujettes aux défauts, mais l'échantillon n'est pas assez grand pour le confirmer.",None),
    }

    # Résultats texte
    st.markdown("#### Résultats du test")
    for v1, v2, chi2, p, dof in khi2_data:
        interp, sig = interps_khi2.get((v1,v2), ("—", False))
        if sig is True:
            bdg = '<span class="bdg s">Significatif</span>'
            force = "forte" if chi2 > 50 else "modérée"
        elif sig is False:
            bdg = '<span class="bdg n">Non significatif</span>'
            force = "forte" if chi2 > 50 else "modérée"
        else:
            bdg = '<span class="bdg g">Zone grise</span>'
            force = "forte"
        st.markdown(f"""<div class="card">
          <div class="card-hd"><span class="card-title">{v1} et {v2}</span>{bdg}</div>
          <p class="stat-vals">Chi² = {chi2:.3f} &nbsp;·&nbsp; p-value = {p:.4f} &nbsp;·&nbsp; ddl = {dof} &nbsp;·&nbsp; Force : {force}</p>
          <p class="interp">{interp}</p>
        </div>""", unsafe_allow_html=True)

    # Tableau de contingence interactif
    st.markdown('<div class="sec-s">Tableau de contingence: Croisement des effectifs entre deux variables catégorielles · Chaque cellule = nombre d\'oranges appartenant aux deux catégories simultanément</div>', unsafe_allow_html=True)
    v1_sel = st.selectbox("Variable 1 (lignes)", cat_cols, index=0)
    v2_sel = st.selectbox("Variable 2 (colonnes)", [c for c in cat_cols if c != v1_sel], index=0)
    ct_df = pd.crosstab(fdf[v1_sel], fdf[v2_sel])
    fig = px.imshow(ct_df, color_continuous_scale=['white','#FF6B00'],
                    title=f"Tableau de contingence : {v1_sel} (lignes) × {v2_sel} (colonnes)<br><sup>Chaque cellule = nombre d'oranges ayant cette combinaison de catégories · Plus la cellule est foncée, plus l'effectif est élevé</sup>",
                    text_auto=True, aspect="auto",
                    labels=dict(x=v2_sel, y=v1_sel, color="Effectif"))
    fig.update_layout(
        paper_bgcolor=BG, font_family="DM Sans", height=420,
        xaxis_title=f"Modalités de {v2_sel}",
        yaxis_title=f"Modalités de {v1_sel}",
        coloraxis=dict(colorbar=dict(title="Nb oranges"))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Graphique Chi² comparatif
    st.markdown('<div class="sec-s">Comparaison des valeurs Chi² — Plus la valeur est grande, plus la relation entre les deux variables est forte</div>', unsafe_allow_html=True)
    chi2_df = pd.DataFrame([(f"{v1} × {v2}", chi2, p, p < 0.05) for v1,v2,chi2,p,_ in khi2_data],
                            columns=['Paire','Chi2','p','sig'])
    colors_khi = ['#2E7D4F' if s else ('#FFA500' if p < 0.1 else '#C0392B')
                  for s, p in zip(chi2_df['sig'], chi2_df['p'])]
    fig = go.Figure(go.Bar(
        x=chi2_df['Paire'], y=chi2_df['Chi2'],
        marker_color=colors_khi,
        text=chi2_df['Chi2'].round(1), textposition='outside',
        hovertemplate="<b>%{x}</b><br>Chi² = %{y:.1f}<extra></extra>"
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='orange', line_width=2,
                  annotation_text='Seuil indicatif — force forte (Chi²> 50)',
                  annotation_position='top left', annotation_font_size=11)
    fig.update_layout(
        title="Valeurs Chi² pour chaque paire de variables catégorielles<br><sup> vert = relation significative (p&lt;0.05) ·  orange = zone grise ·  rouge = pas de relation · Plus Chi² est grand, plus le lien est fort</sup>",
        yaxis_title='Valeur Chi² (plus grand = lien plus fort)',
        xaxis_title='Paire de variables testées',
        paper_bgcolor=BG, plot_bgcolor=PLT_BG,
        font_family='DM Sans', height=400, showlegend=False,
        title_font_size=13
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="ins am"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation globale du Khi-2</p>
      <p class="ins-tx">Le test du Khi-2 révèle que la couleur et la variété sont fortement liées entre elles (Chi²=233,8, p≈0), ce qui est biologiquement logique. En revanche, ni la couleur ni la variété ne permettent de prédire la présence de défauts avec certitude. La conclusion clé : les défauts visuels sont indépendants du profil variétal, ce qui confirme leur caractère accidentel et superficiel.</p>
    </div></div>""", unsafe_allow_html=True)


# ANOVA / T-TEST

elif "Variation" in page:
    st.markdown('<div class="sec"> ANOVA / T-test : Numériques et Catégorielles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Section 6.3 du notebook — Relation entre variables numériques et catégorielles</div>', unsafe_allow_html=True)

    cat_cols = ['Color','Variety','Blemishes(Y/N)']
    anova_results = {}
    test_names = {}
    for cat in cat_cols:
        mods = fdf[cat].nunique()
        test_names[cat] = "T-test" if mods == 2 else "ANOVA"
        pvals = []
        for num in NUM:
            groupes = [fdf[fdf[cat]==m][num].dropna() for m in fdf[cat].unique()]
            if mods == 2:
                _, p = ttest_ind(*groupes)
            else:
                _, p = f_oneway(*groupes)
            pvals.append(p)
        anova_results[cat] = pvals

    interps_anova = {
        'Color': ("blue","Color et Variables numériques (ANOVA)",
            "C'est la relation la plus forte : toutes les variables numériques sont significativement liées à la couleur, sans exception. La couleur d'une orange n'est pas qu'esthétique, elle reflète fidèlement son profil complet : taille, poids, sucre, acidité, texture, temps de récolte et qualité. La couleur est quasiment un résumé visuel de l'orange."),
        'Variety': ("am","Variety et Variables numériques (ANOVA)",
            "La variété influence significativement la plupart des variables, sauf le poids et la maturité (Ripeness). Autrement dit, selon la variété, les oranges diffèrent en taille, sucre, acidité, texture et qualité, mais leur poids et leur niveau de maturité restent comparables d'une variété à l'autre."),
        'Blemishes(Y/N)': ("rd","Blemishes et Variables numériques (T-test)",
            "C'est la relation la plus faible des trois. Les défauts ne sont liés significativement qu'à quelques variables isolées (poids, pH, maturité notamment), et pas du tout à la qualité finale (p=0.117). Cela confirme ce qu'on avait observé avec le Khi-2 : les défauts sont superficiels et n'impactent pas vraiment la qualité intrinsèque du fruit."),
    }

    # Résultats détaillés par variable catégorielle
    for cat, pvals in anova_results.items():
        color, title, interp = interps_anova[cat]
        st.markdown(f"""<div class="card">
          <div class="card-hd"><span class="card-title">{title}</span></div>
          <p class="interp">{interp}</p>
        </div>""", unsafe_allow_html=True)

        rows_html = ""
        for num, p in zip(NUM, pvals):
            sig = p < 0.05
            bdg = '<span class="bdg s">Significatif</span>' if sig else '<span class="bdg n">Non significatif</span>'
            pbar = min(1, -np.log10(p+1e-10) / 6)
            bar_color = '#2E7D4F' if sig else '#C0392B'
            rows_html += f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;
                        border-bottom:1px solid #F5F0EB;font-size:.85rem;">
              <span style="flex:2;font-weight:500;color:#1A0A00">{NLBL[num]}</span>
              <div style="flex:3;height:7px;background:#F0EDE8;border-radius:4px;overflow:hidden">
                <div style="width:{pbar*100:.0f}%;height:100%;background:{bar_color};border-radius:4px"></div>
              </div>
              <span style="min-width:70px;font-variant-numeric:tabular-nums;color:#555;font-size:.8rem">p = {p:.4f}</span>
              {bdg}
            </div>"""

        st.markdown(f'<div style="background:white;border-radius:12px;border:1px solid var(--bdr);overflow:hidden;margin-bottom:16px">{rows_html}</div>', unsafe_allow_html=True)

    # Graphique ANOVA comparatif (identique au notebook)
    st.markdown('<div class="sec-s">Graphique comparatif — Plus la barre dépasse la ligne rouge, plus la relation est statistiquement forte · La hauteur représente −log10(p-value) : plus c\'est haut = plus c\'est significatif</div>', unsafe_allow_html=True)
    seuil = -np.log10(0.05)
    labels_short = [NLBL[c] for c in NUM]
    colors_cat = ['#CC3300','#FF9A3C','#1A5F8A']

    fig = go.Figure()
    for i, (cat, pvals) in enumerate(anova_results.items()):
        heights = [-np.log10(p+1e-10) for p in pvals]
        fig.add_trace(go.Bar(
            name=cat, x=labels_short, y=heights,
            marker_color=colors_cat[i], opacity=0.88,
            text=[f"{h:.1f}" for h in heights], textposition='outside',
            textfont=dict(size=9),
            hovertemplate=f"<b>{cat}</b><br>Variable : %{{x}}<br>−log10(p) = %{{y:.2f}}<br>Significatif si > {seuil:.2f}<extra></extra>"
        ))
    fig.add_hline(y=seuil, line_dash='dash', line_color='red', line_width=2,
                  annotation_text='Seuil de significativité p = 0.05 → Toute barre AU-DESSUS = relation confirmée',
                  annotation_position='top right',
                  annotation_font_color=PLT_WARN, annotation_font_size=10)
    fig.update_layout(
        barmode='group',
        title="ANOVA / T-test — Significativité de chaque variable numérique selon la variable catégorielle<br><sup>Axe Y = −log10(p-value) · Au-dessus de la ligne rouge = relation statistiquement significative (p &lt; 0.05)</sup>",
        yaxis_title='−log10(p-value)  ·  Plus haut = plus significatif',
        xaxis_title='Variable numérique analysée',
        paper_bgcolor=BG, plot_bgcolor=PLT_BG,
        font_family='DM Sans', height=480, title_font_size=13,
        legend=dict(orientation='h', y=1.1, title="Variable catégorielle testée",
                    bgcolor="rgba(30,30,30,0.85)", bordercolor="#FFB866", borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="ins bl"><span class="ins-ic"></span><div>
      <p class="ins-t">Interprétation du graphique</p>
      <p class="ins-tx">
        La ligne rouge pointillée est le seuil : toute barre qui la dépasse = relation significative (p &lt; 0.05).<br><br>
        <strong>Color (rouge foncé) , le plus puissant</strong> : Toutes les barres dépassent largement le seuil. La couleur est liée à absolument toutes les variables numériques, avec une force particulièrement élevée sur le Brix, la Softness, HarvestTime et la Qualité. C'est la variable catégorielle la plus informative du dataset.<br><br>
        <strong>Variety (orange), très proche de Color</strong> : Quasiment toutes les barres dépassent le seuil, sauf Poids et Maturité qui restent en dessous. La variété explique bien les différences entre fruits, mais ne détermine ni leur poids ni leur maturité.<br><br>
        <strong>Blemishes (bleu), le plus faible</strong> : C'est visuellement frappant : la majorité des barres bleues restent sous la ligne rouge. Seuls le pH, le poids et la Maturité dépassent le seuil. Les défauts sont donc largement indépendants des caractéristiques physiques et chimiques, et surtout indépendants de la <strong>qualité</strong> (p=0.117).
      </p>
    </div></div>""", unsafe_allow_html=True)


# ANALYSE QUALITÉ

elif "bonne orange" in page:
    st.markdown('<div class="sec"> Analyse Approfondie de la Qualité</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">La Qualité (1-5) est la variable cible de l\'analyse — voici comment elle se distribue et ce qui l\'influence</p>', unsafe_allow_html=True)

    # Graphique 1 : Distribution + Qualité par couleur
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#FFE0C0;margin:12px 0 4px">Graphique 1 — Distribution de la Qualité &nbsp;|&nbsp; Graphique 2 — Qualité moyenne selon la couleur</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">À gauche : comment les notes se répartissent sur l\'ensemble du lot · À droite : est-ce que la couleur de l\'orange influence sa note ?</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        d_q = fdf['Quality(1-5)']
        fig = px.histogram(fdf, x='Quality(1-5)', nbins=20,
                           color_discrete_sequence=['#FF6B00'],
                           title="Distribution de la variable Qualité (1 à 5)<br><sup>Chaque barre = une note · Hauteur = nombre d'oranges ayant cette note</sup>",
                           labels={'Quality(1-5)': 'Note de Qualité (1=mauvaise · 5=excellente)', 'count': "Nombre d'oranges"},
                           marginal="box")
        fig.add_vline(x=d_q.mean(), line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Moyenne = {d_q.mean():.2f}",
                      annotation_position="top right", annotation_font_color=PLT_WARN, annotation_font_size=11)
        fig.add_vline(x=d_q.median(), line_dash="dot", line_color="#60AAFF", line_width=2,
                      annotation_text=f"Médiane = {d_q.median():.1f}",
                      annotation_position="top left", annotation_font_color="#60AAFF", annotation_font_size=11)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
            xaxis_title="Note de Qualité (1 = mauvaise · 5 = excellente)",
            yaxis_title="Nombre d'oranges",
            title_font_size=12,
            annotations=[dict(
                x=0.5, y=-0.25, xref="paper", yref="paper",
                text="📊 Une distribution penchée vers la droite (vers 4-5) indique un lot de bonne qualité",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        qc = fdf.groupby('Color')['Quality(1-5)'].mean().reset_index().sort_values('Quality(1-5)', ascending=False)
        qc['Qualite'] = qc['Quality(1-5)'].round(2)
        fig = px.bar(qc, x='Color', y='Quality(1-5)', color='Color',
                     color_discrete_sequence=['#CC3300','#FF6B00','#FF9A3C','#FFB866','#FFD700'],
                     title="Qualité moyenne selon la couleur de l'orange<br><sup>Chaque barre = qualité moyenne des oranges de cette couleur · Score sur 5</sup>",
                     labels={'Color': 'Couleur', 'Quality(1-5)': 'Qualité moyenne (/ 5)'},
                     text=qc['Qualite'])
        fig.add_hline(y=fdf['Quality(1-5)'].mean(), line_dash="dash", line_color="rgba(255,255,255,0.35)", line_width=1.5,
                      annotation_text=f"Moy. globale : {fdf['Quality(1-5)'].mean():.2f}",
                      annotation_position="top right", annotation_font_size=10)
        fig.update_traces(textposition='outside', textfont_size=12)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, showlegend=False, font_family="DM Sans",
            xaxis_title="Couleur de l'orange",
            yaxis_title="Qualité moyenne (échelle 1–5)",
            yaxis_range=[0, 5.5],
            title_font_size=12,
            annotations=[dict(
                x=0.5, y=-0.22, xref="paper", yref="paper",
                text="📌 Ligne pointillée = qualité moyenne générale du lot · Les valeurs annotées au-dessus de chaque barre sont les scores moyens",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Graphique 3 : Qualité par variété
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#FFE0C0;margin:20px 0 4px">Graphique 3 — Qualité moyenne par variété d\'orange</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">Chaque barre représente la note de qualité moyenne des oranges de cette variété · Permet d\'identifier les variétés les plus qualitatives</p>', unsafe_allow_html=True)
    qv = fdf.groupby('Variety')['Quality(1-5)'].mean().reset_index().sort_values('Quality(1-5)', ascending=False)
    fig = px.bar(qv, x='Variety', y='Quality(1-5)',
                 color='Quality(1-5)',
                 color_continuous_scale=['#FFD580','#FF6B00','#C0392B'],
                 title="Qualité moyenne par variété d'orange: classement du meilleur au moins bon<br><sup>Couleur = intensité de la qualité · Plus foncé = meilleur · Score sur 5</sup>",
                 labels={'Variety': 'Variété d\'orange', 'Quality(1-5)': 'Qualité moyenne (/ 5)'},
                 text=qv['Quality(1-5)'].round(2))
    fig.add_hline(y=fdf['Quality(1-5)'].mean(), line_dash="dash", line_color="red", line_width=1.5,
                  annotation_text=f"Moy. : {fdf['Quality(1-5)'].mean():.2f}",
                  annotation_position="top right", annotation_font_color=PLT_WARN, annotation_font_size=11)
    fig.update_traces(textposition='outside', textfont_size=10)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLT_BG, xaxis_tickangle=-40, height=450,
        font_family="DM Sans", title_font_size=12,
        xaxis_title="Variété d'orange",
        yaxis_title="Note de qualité moyenne (1–5)",
        yaxis_range=[0, 5.8],
        coloraxis_showscale=False,
        annotations=[dict(
            x=0.5, y=-0.28, xref="paper", yref="paper",
            text="📌 Ligne rouge = qualité moyenne globale (3.82/5) · Les variétés au-dessus de cette ligne sont supérieures à la moyenne",
            showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="center"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Graphique 4 : Défauts vs qualité
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#FFE0C0;margin:20px 0 4px">Graphique 4 — Qualité selon la présence de défauts &nbsp;|&nbsp; Graphique 5 — Qualité par type de défaut</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-s">Question clé : est-ce qu\'une orange avec des défauts visuels a forcément une moins bonne qualité ? (réponse : NON, p = 0.117)</p>', unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        lbl = fdf['sans_defaut'].map({True:"Sans défaut (N)", False:"Avec défaut"})
        fig = px.box(fdf, x=lbl, y='Quality(1-5)', color=lbl,
                     color_discrete_map={"Sans défaut (N)":"#2E7D4F","Avec défaut":"#C0392B"},
                     title="Qualité selon la présence ou l'absence de défauts<br><sup>Boîte = Q1/médiane/Q3 · Moustaches = min/max · Points = valeurs extrêmes</sup>",
                     labels={'x': 'Statut des défauts', 'Quality(1-5)': 'Qualité (1–5)'})
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, showlegend=False, font_family="DM Sans",
            xaxis_title="Présence de défauts visuels",
            yaxis_title="Note de Qualité (1–5)",
            title_font_size=12,
            annotations=[dict(
                x=0.5, y=-0.22, xref="paper", yref="paper",
                text=" T-test : p = 0.117 — la différence n'est PAS statistiquement significative",
                showarrow=False, font=dict(size=10, color=PLT_WARN), xanchor="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        bq = fdf.groupby('Blemishes(Y/N)')['Quality(1-5)'].mean().reset_index().sort_values('Quality(1-5)')
        fig = px.bar(bq, x='Quality(1-5)', y='Blemishes(Y/N)', orientation='h',
                     color='Quality(1-5)',
                     color_continuous_scale=['#CC2200','#FF6B00','#FFD700'],
                     title="Qualité moyenne par type de défaut recensé<br><sup>N = aucun défaut · Autres codes = types de défauts spécifiques · Score moyen / 5</sup>",
                     labels={'Blemishes(Y/N)': 'Code défaut', 'Quality(1-5)': 'Qualité moyenne'},
                     text=bq['Quality(1-5)'].round(2))
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=PLT_BG, font_family="DM Sans",
            coloraxis_showscale=False,
            xaxis_title="Note de qualité moyenne (1–5)",
            yaxis_title="Code du type de défaut (N = sans défaut)",
            xaxis_range=[0, 5.5],
            title_font_size=12,
            annotations=[dict(
                x=1.0, y=-0.18, xref="paper", yref="paper",
                text=" Les scores proches entre N et les autres codes confirment l'absence d'impact des défauts sur la qualité",
                showarrow=False, font=dict(size=10, color=PLT_ANN), xanchor="right"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="ins am"><span class="ins-ic"></span><div>
      <p class="ins-t">Les défauts visuels n'impactent pas la qualité : résultat clé</p>
      <p class="ins-tx">Le T-test confirme ce que le graphique montre : la relation entre défauts visuels et qualité n'est pas statistiquement significative (p = 0.117). Quand un défaut apparaît, il est le plus souvent superficiel, un coup de soleil, une cicatrice, sans impact réel sur la qualité intrinsèque du fruit. Un fruit avec des défauts peut tout à fait avoir une bonne note de qualité.</p>
    </div></div>""", unsafe_allow_html=True)


# CONCLUSION & KPIs

elif "Résultats" in page:
    st.markdown("""<div class="hero">
      <p class="hero-eye">Résultats finaux · EDA Orange Dataset</p>
      <h1>Conclusion & KPIs</h1>
      <p class="hero-sub">Synthèse complète des informations utiles identifiées par l'analyse exploratoire</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec"> Les 3 choses qui font une bonne orange</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown('<div class="kpi g"><span style="font-size:1.4rem"></span><p class="kpi-lbl">Ce qui compte le plus</p><p class="kpi-val" style="font-size:1.15rem">Le taux de sucre</p><p class="kpi-ref">Plus l\'orange est sucrée, meilleure est sa note de qualité</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="kpi r"><span style="font-size:1.4rem"></span><p class="kpi-lbl">Ce qu\'il faut surveiller</p><p class="kpi-val" style="font-size:1.15rem">Le temps après récolte</p><p class="kpi-ref">Plus l\'orange attend, plus sa qualité se dégrade</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="kpi"><span style="font-size:1.4rem"></span><p class="kpi-lbl">Les signes d\'alerte</p><p class="kpi-val" style="font-size:1.15rem">La mollesse et l\'acidité</p><p class="kpi-ref">Quand le fruit ramollit ou devient trop acide, la qualité baisse</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec"> Chiffres clés du lot analysé</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi g"><p class="kpi-lbl"> Note de qualité moyenne</p><p class="kpi-val">3.82<span> sur 5</span></p><p class="kpi-ref">La majorité des oranges sont bien notées</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi g"><p class="kpi-lbl"> Oranges bien notées</p><p class="kpi-val">65.1<span>%</span></p><p class="kpi-ref">157 oranges sur 241 ont une note de 4 ou 5 sur 5</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi r"><p class="kpi-lbl"> Oranges à écarter</p><p class="kpi-val">9.5<span>%</span></p><p class="kpi-ref">23 oranges jugées de mauvaise qualité dans ce lot</p></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Oranges sans tache ni marque</p><p class="kpi-val">61.8<span>%</span></p><p class="kpi-ref">149 oranges d\'apparence parfaite sur 241</p></div>', unsafe_allow_html=True)

    c5,c6,c7,c8 = st.columns(4)
    with c5: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Douceur moyenne</p><p class="kpi-val">10.9</p><p class="kpi-ref">Plus ce chiffre est élevé, plus l\'orange est sucrée et bonne</p></div>', unsafe_allow_html=True)
    with c6: st.markdown(f'<div class="kpi r"><p class="kpi-lbl"> Jours depuis la récolte</p><p class="kpi-val">15<span> jours</span></p><p class="kpi-ref">Plus l\'orange attend, moins elle est bonne</p></div>', unsafe_allow_html=True)
    with c7: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Niveau d\'acidité moyen</p><p class="kpi-val">3.47</p><p class="kpi-ref">Toutes les oranges ont une acidité très proche : c\'est un bon signe</p></div>', unsafe_allow_html=True)
    with c8: st.markdown(f'<div class="kpi"><p class="kpi-lbl"> Niveau de maturité moyen</p><p class="kpi-val">3.6<span> sur 5</span></p><p class="kpi-ref">Les fruits bien mûrs sont systématiquement mieux notés</p></div>', unsafe_allow_html=True)

    # Synthèse résultats statistiques
    st.markdown('<div class="sec"> Synthèse des Tests Statistiques</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Khi-2 — Variables catégorielles**")
        for v1,v2,chi2,p,sig,interp in [
            ("Color","Variety",233.8,0.000,True,"La couleur est biologiquement liée à la variété."),
            ("Color","Blemishes",55.4,0.117,False,"La couleur ne prédit pas les défauts."),
            ("Variety","Blemishes",285.2,0.080,None,"Zone grise : tendance possible, échantillon insuffisant."),
        ]:
            if sig is True: bdg='<span class="bdg s">Significatif</span>'
            elif sig is False: bdg='<span class="bdg n">Non significatif</span>'
            else: bdg='<span class="bdg g">Zone grise</span>'
            st.markdown(f'<div class="card"><div class="card-hd"><span class="card-title">{v1} et {v2}</span>{bdg}</div><p class="stat-vals">Chi² = {chi2} · p = {p}</p><p class="interp">{interp}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**ANOVA/T-test: Numériques et Catégorielles**")
        for cat, desc, bdg_txt, bdg_cls in [
            ("Color","Toutes les variables sont significativement liées à la couleur.","100% significatif","s"),
            ("Variety","Influence la plupart des variables, sauf Poids et Maturité.","Largement significatif","s"),
            ("Blemishes","Pas du tout lié à la Qualité (p=0.117). Défauts superficiels.","Qualité non affectée","n"),
        ]:
            st.markdown(f'<div class="card"><div class="card-hd"><span class="card-title">{cat} et Variables numériques</span><span class="bdg {bdg_cls}">{bdg_txt}</span></div><p class="interp">{desc}</p></div>', unsafe_allow_html=True)

    # Conclusion narrative complète
    st.markdown('<div class="sec"> Conclusion Générale</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ins bl"><span class="ins-ic"></span><div>
      <p class="ins-t">Ce que l'analyse exploratoire nous apprend</p>
      <p class="ins-tx">Cet échantillon de 241 oranges décrit globalement des fruits de <strong>bonne qualité et bien mûrs</strong>, avec une acidité très stable. Le point le plus intrigant reste la sucrosité, dont la distribution en deux groupes distincts (autour de 8°Bx et 14°Bx) soulève une vraie question sur la composition du lot, probablement deux variétés ou deux origines différentes.<br><br>
      La <strong>sucrosité est le prédicteur numéro un</strong> de la qualité (r = +0.63). Loin devant tous les autres facteurs. C'est aussi le seul levier sur lequel on peut agir à la source, via le choix variétal et le timing de récolte.<br><br>
      Les <strong>défauts visuels ne pénalisent pas la qualité</strong> , résultat contre-intuitif mais statistiquement solide (p = 0.117). Un fruit avec des taches de soleil ou une petite cicatrice mérite la même attention qu'un fruit parfait en apparence.<br><br>
      La <strong>couleur est un résumé visuel de l'orange</strong> : liée significativement à toutes les variables numériques sans exception (ANOVA, p &lt; 0.05 pour chacune), elle constitue un indicateur de terrain rapide et fiable.</p>
    </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ins gn"><span class="ins-ic"></span><div>
      <p class="ins-t">Recommandations opérationnelles</p>
      <p class="ins-tx">
        1. <strong>Surveiller le Brix en priorité</strong> : c'est le KPI qualité le plus prédictif. Un seuil minimum de 10°Bx devrait être défini pour la sélection.<br>
        2. <strong>Minimiser le temps post-récolte</strong> : chaque jour supplémentaire dégrade la qualité (r = −0.47). La logistique rapide est un levier qualité direct.<br>
        3. <strong>Ne pas écarter les fruits sur leur apparence seule</strong> : les défauts visuels n'impactent pas la qualité réelle (p = 0.117).<br>
        4. <strong>Investiguer la bimodalité du Brix</strong> : les deux groupes distincts (8° et 14°) méritent une analyse séparée par variété ou origine.
      </p>
    </div></div>""", unsafe_allow_html=True)
