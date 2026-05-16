import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
import itertools

# -------------------------------
# Page configuration
st.set_page_config(page_title="🍊 Orange Quality Analysis", layout="wide", page_icon="🍊")

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FF8C00; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #2E7D32; margin-top: 1rem; margin-bottom: 0.5rem; }
    .interpretation { background-color: #FFF8E1; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF8C00; margin-top: 1rem; }
    .metric-card { background-color: #F5F5F5; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Data loading with caching
@st.cache_data
def load_data():
    df = pd.read_csv("orange.csv")
    df = df.rename(columns={
        'Size (cm)': 'Size(cm)',
        'Weight (g)': 'Weight(g)',
        'Brix (Sweetness)': 'Brix(Sweetness)',
        'pH (Acidity)': 'pH(Acidity)',
        'Softness (1-5)': 'Softness(1-5)',
        'HarvestTime (days)': 'HarvestTime(days)',
        'Ripeness (1-5)': 'Ripeness(1-5)',
        'Blemishes (Y/N)': 'Blemishes(Y/N)',
        'Quality (1-5)': 'Quality(1-5)'
    })
    return df

df = load_data()

# Numerical and categorical columns
num_cols = ['Size(cm)', 'Weight(g)', 'Brix(Sweetness)', 'pH(Acidity)',
            'Softness(1-5)', 'HarvestTime(days)', 'Ripeness(1-5)', 'Quality(1-5)']
cat_cols = ['Color', 'Variety', 'Blemishes(Y/N)']

# -------------------------------
# Sidebar navigation with icons
st.sidebar.title("🍊 Navigation")
page = st.sidebar.radio("", 
    ["🏠 Home", "📊 Data Overview", "📈 Numerical Distributions", 
     "🏷️ Categorical Analysis", "🔗 Bivariate Analysis", "📌 Conclusions"],
    format_func=lambda x: x.split()[1] if x.startswith("🏠") else x.split()[1] if x.startswith("📊") else x.split()[1] if x.startswith("📈") else x.split()[1] if x.startswith("🏷️") else x.split()[1] if x.startswith("🔗") else x.split()[1])
# Simplify: just use the full label
page = st.sidebar.radio("Go to", 
    ["🏠 Home", "📊 Data Overview", "📈 Numerical Distributions", 
     "🏷️ Categorical Analysis", "🔗 Bivariate Analysis", "📌 Conclusions"])

# Helper functions for plots (improved styling)
@st.cache_data
def plot_histograms():
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('Distribution of Numerical Variables', fontsize=16, fontweight='bold', color='#FF8C00')
    for i, col in enumerate(num_cols):
        ax = axes[i // 2, i % 2]
        data = df[col].dropna()
        ax.hist(data, bins=20, color='#FF8C00', edgecolor='white', alpha=0.85)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='blue', linestyle=':', linewidth=1.8, label=f'Median: {data.median():.2f}')
        ax.set_title(col, fontweight='bold', fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

@st.cache_data
def plot_categorical():
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Categorical Variables Analysis', fontsize=16, fontweight='bold', color='#FF8C00')
    
    # Color distribution
    color_counts = df['Color'].value_counts()
    axes[0].bar(color_counts.index, color_counts.values, color='#FF8C00', edgecolor='white')
    axes[0].set_title('Color Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Top 12 varieties
    variety_counts = df['Variety'].value_counts().head(12)
    axes[1].barh(variety_counts.index, variety_counts.values, color='#FF8C00', edgecolor='white')
    axes[1].set_title('Top 12 Varieties', fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].invert_yaxis()
    
    # Blemishes
    blem_counts = df['Blemishes(Y/N)'].value_counts()
    axes[2].barh(blem_counts.index, blem_counts.values, color='#FF6347', edgecolor='white')
    axes[2].set_title('Blemishes Distribution', fontweight='bold')
    axes[2].set_xlabel('Count')
    for i, v in enumerate(blem_counts.values):
        axes[2].text(v + 0.5, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    return fig

@st.cache_data
def plot_correlation():
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix – Numerical Variables', fontsize=14, fontweight='bold')
    return fig

@st.cache_data
def plot_anova_pvalues():
    results = {}
    for cat in cat_cols:
        pvals = []
        for num in num_cols:
            groups = [df[df[cat] == val][num].dropna() for val in df[cat].unique()]
            if len(groups) == 2:
                _, p = ttest_ind(*groups)
            else:
                _, p = f_oneway(*groups)
            pvals.append(p)
        results[cat] = pvals
    
    x = np.arange(len(num_cols))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#FF8C00', '#FFA500', '#1E88E5']
    for i, (cat, pvals) in enumerate(results.items()):
        heights = [-np.log10(p) if p > 0 else 5 for p in pvals]
        ax.bar(x + i*width, heights, width, color=colors[i], alpha=0.85, label=cat, edgecolor='white')
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.2, label='p=0.05 threshold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(num_cols, rotation=30, ha='right')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Significance of Relations (ANOVA / t-test)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig

# -------------------------------
# Page content
if page == "🏠 Home":
    st.markdown('<div class="main-header">🍊 Orange Quality Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Objective
    Explore and understand the physico‑chemical characteristics of oranges to identify the key factors determining their quality.
    
    ### 📊 Dataset Overview
    - **241 orange samples** with **11 features**
    - **Physical:** Size, Weight, Softness
    - **Chemical:** Brix (sweetness), pH (acidity)
    - **Harvest:** HarvestTime, Ripeness
    - **Visual:** Color, Variety, Blemishes
    - **Target:** Quality (1‑5)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.image("https://images.unsplash.com/photo-1547514715-18b3c41f0ccd?w=800", caption="Orange quality assessment", use_container_width=True)

elif page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 8 rows**")
        st.dataframe(df.head(8), use_container_width=True)
    with col2:
        st.write("**Data Types**")
        st.dataframe(df.dtypes.to_frame("Type"), use_container_width=True)
    
    st.write("**Descriptive Statistics (Numerical)**")
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
    
    st.write("**Missing values**")
    st.write(df.isnull().sum())
    
    st.write("**Duplicates**")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

elif page == "📈 Numerical Distributions":
    st.markdown('<div class="sub-header">📈 Distribution of Numerical Variables</div>', unsafe_allow_html=True)
    st.pyplot(plot_histograms())
    with st.expander("📖 Interpretation (click to expand)"):
        st.markdown("""
        <div class='interpretation'>
        <b>Quality (1‑5):</b> Mostly high (4‑5), low quality is rare → overall good batch.<br><br>
        <b>Ripeness (1‑5):</b> Dominated by 4‑5, but a few under‑ripe fruits exist.<br><br>
        <b>Brix (Sweetness):</b> Bimodal distribution (peaks near 8 and 14). Suggests two distinct groups (varieties or harvest stages).<br><br>
        <b>pH (Acidity):</b> Very concentrated between 3.2‑3.6; homogeneous acidity.<br><br>
        <b>Softness (1‑5):</b> Discrete values (1,3,5) – likely subjective scoring rather than continuous measure.<br><br>
        <b>Size & Weight:</b> Wide natural spread (6‑10 cm, 100‑300 g).<br><br>
        <b>HarvestTime:</b> Slight peak at 12‑13 days, otherwise fairly uniform.
        </div>
        """, unsafe_allow_html=True)

elif page == "🏷️ Categorical Analysis":
    st.markdown('<div class="sub-header">🏷️ Categorical Variables</div>', unsafe_allow_html=True)
    st.pyplot(plot_categorical())
    with st.expander("📖 Interpretation"):
        st.markdown("""
        <div class='interpretation'>
        <b>Color:</b> Dominated by deep orange, light orange and orange‑red → generally good maturity.<br><br>
        <b>Variety:</b> 24 varieties present; Cara Cara, Star Ruby and Temple are most frequent.<br><br>
        <b>Blemishes:</b> 149 fruits have no blemishes. Defects are mostly superficial (sunburn, scars) and do not necessarily lower quality.
        </div>
        """, unsafe_allow_html=True)

elif page == "🔗 Bivariate Analysis":
    st.markdown('<div class="sub-header">🔗 Bivariate Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("📊 Correlation Matrix (Numerical)")
    st.pyplot(plot_correlation())
    st.markdown("""
    <div class='interpretation'>
    <b>Key correlations with Quality:</b><br>
    - <b>Brix (+0.63)</b> – strong positive: sweeter oranges are rated higher.<br>
    - <b>HarvestTime (-0.47)</b> – moderate negative: longer post‑harvest storage lowers quality.<br>
    - <b>Softness (-0.32)</b> – weak negative: softer fruits tend to have lower quality.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📐 Chi‑square Tests (Categorical pairs)")
    chi_results = {}
    for var1, var2 in itertools.combinations(cat_cols, 2):
        ct = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, _ = chi2_contingency(ct)
        chi_results[f"{var1} vs {var2}"] = (chi2, p, dof)
    
    for name, (chi2, p, dof) in chi_results.items():
        st.write(f"**{name}:** χ² = {chi2:.2f}, p = {p:.4f}, df = {dof}")
        if p < 0.05:
            st.success("✅ Significant relationship")
        else:
            st.info("❌ No significant relationship")
    
    st.subheader("📉 ANOVA / t‑test (Numerical vs Categorical)")
    st.pyplot(plot_anova_pvalues())
    with st.expander("📖 Interpretation"):
        st.markdown("""
        <div class='interpretation'>
        <b>Color</b> – strongest influence: significant for <i>all</i> numerical variables (p < 0.05).<br><br>
        <b>Variety</b> – significant for most variables except <code>Weight(g)</code> and <code>Ripeness(1-5)</code>.<br><br>
        <b>Blemishes</b> – mostly non‑significant (only pH, weight, ripeness show some effect). Defects are largely independent of fruit characteristics.
        </div>
        """, unsafe_allow_html=True)

elif page == "📌 Conclusions":
    st.markdown('<div class="sub-header">📌 Key Findings & Conclusions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='interpretation'>
    ✅ <b>Global quality is high</b> (average 3.8/5). Low‑quality oranges are exceptions.<br><br>
    ✅ <b>Sweetness (Brix) is the strongest predictor of quality</b> (corr = 0.63).<br><br>
    ✅ <b>Post‑harvest time</b> deteriorates quality significantly (corr = –0.47).<br><br>
    ✅ <b>Colour</b> is an excellent summary of the fruit’s profile: strongly linked to all numerical variables.<br><br>
    ✅ <b>Variety</b> explains differences in size, sugar, acidity and softness, but not weight or ripeness.<br><br>
    ✅ <b>Blemishes</b> are mostly superficial and do not predict overall quality.<br><br>
    ✅ The bimodal distribution of Brix suggests two distinct sub‑populations (possibly different varieties or harvest times) that deserve further investigation.
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Recommendation:** Focus quality control on sugar content and harvest timing. Colour and variety serve as reliable visual indicators.")
