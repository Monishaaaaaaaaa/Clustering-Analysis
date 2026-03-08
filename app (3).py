import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-number { font-size: 2rem; font-weight: bold; }
    .segment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Segment Config ───────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    'VIP Champions':        '#00FFB3',
    'Affluent Loyalists':   '#00B4FF',
    'Deal-Seeking Families':'#FF6B00',
    'Budget Families':      '#FF2D78',
}

DARK_BG    = '#0D1117'
DARK_PANEL = '#161B22'
DARK_GRID  = '#21262D'

SEGMENT_EMOJI = {
    'VIP Champions':       '👑',
    'Affluent Loyalists':  '💎',
    'Deal-Seeking Families':'🏷️',
    'Budget Families':     '👨‍👩‍👧',
}

SEGMENT_DESC = {
    'VIP Champions':        'Highest income & spending. Most valuable customers. Prioritize retention & upsell.',
    'Affluent Loyalists':   'High income, frequent buyers, campaign responsive. Prime upsell targets.',
    'Deal-Seeking Families':'Mid income, heavy deal usage, low campaign response. Promotion-driven.',
    'Budget Families':      'Low income, high children, price-sensitive. Cost-effective engagement.',
}

FEATURES = ['Income','Recency','Total_Spending','Total_Purchases',
            'NumDealsPurchases','Total_Children','Total_Accepted_Campaigns',
            'NumWebPurchases','NumCatalogPurchases']

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('segmented_customers.csv')
    return df

@st.cache_resource
def load_models():
    km     = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca    = joblib.load('pca.pkl')
    return km, scaler, pca

df = load_data()
km, scaler, pca_model = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/customer-insight.png", width=80)
    st.title("🎯 Customer Segmentation")
    st.markdown("**DS Group 4**")
    st.divider()

    st.markdown("### 🔍 Filter Segments")
    all_segments = list(SEGMENT_COLORS.keys())
    selected_segments = st.multiselect(
        "Select segments to display:",
        options=all_segments,
        default=all_segments
    )

    st.divider()
    st.markdown("### 📊 About")
    st.info("""
    **Model:** K-Means (K=4)
    **Features:** 9 behavioral & monetary
    **Scaling:** StandardScaler
    **Method:** Elbow + Silhouette
    """)

    st.markdown("### 📈 Dataset")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Features Used", "9")
    st.metric("Clusters", "4")

# ── Filter data ───────────────────────────────────────────────────────────────
filtered_df = df[df['Segment'].isin(selected_segments)] if selected_segments else df

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation Dashboard")
st.markdown("**K-Means Clustering · 4 Segments · 2,240 Customers**")
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, seg in zip([k1, k2, k3, k4], all_segments):
    seg_df = df[df['Segment'] == seg]
    pct = len(seg_df) / len(df) * 100
    color = SEGMENT_COLORS[seg]
    emoji = SEGMENT_EMOJI[seg]
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:1.8rem">{emoji}</div>
        <div style="color:{color}; font-weight:700; font-size:1rem">{seg}</div>
        <div class="metric-number" style="color:{color}">{len(seg_df):,}</div>
        <div style="color:#888; font-size:0.85rem">{pct:.1f}% of customers</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Cluster Map",
    "📊 Segment Profiles",
    "🔥 Feature Heatmap",
    "📈 Deep Dive",
    "🔮 Predict My Segment"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — PCA Cluster Map
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🗺️ Customer Clusters in 2D Space (PCA)")
    st.caption("K-Means trained on 9 features · Visualized via PCA (2 components)")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.scatter(
            filtered_df,
            x='PCA1', y='PCA2',
            color='Segment',
            color_discrete_map=SEGMENT_COLORS,
            hover_data={
                'PCA1': False, 'PCA2': False,
                'Income': ':,.0f',
                'Total_Spending': ':,.0f',
                'Recency': True,
                'Segment': True
            },
            title="K-Means Customer Segments — PCA Projection",
            opacity=0.7,
            height=520
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            plot_bgcolor=DARK_BG,
            paper_bgcolor=DARK_PANEL,
            font=dict(color='#FFFFFF'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                       font=dict(color='white')),
            xaxis=dict(showgrid=True, gridcolor=DARK_GRID, color='#aaaaaa',
                      zerolinecolor=DARK_GRID),
            yaxis=dict(showgrid=True, gridcolor=DARK_GRID, color='#aaaaaa',
                      zerolinecolor=DARK_GRID),
            title=dict(font=dict(color='white')),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Segment Guide")
        for seg in all_segments:
            color = SEGMENT_COLORS[seg]
            emoji = SEGMENT_EMOJI[seg]
            desc  = SEGMENT_DESC[seg]
            st.markdown(f"""
            <div style="background:#161B22; border-left: 4px solid {color};
                        padding:10px 12px; border-radius:6px; margin-bottom:10px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.4)">
                <div style="color:{color}; font-weight:700">{emoji} {seg}</div>
                <div style="color:#aaaaaa; font-size:0.82rem; margin-top:4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Cluster size donut
    st.subheader("📊 Segment Size Distribution")
    seg_counts = df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig_donut = px.pie(
        seg_counts, names='Segment', values='Count',
        color='Segment', color_discrete_map=SEGMENT_COLORS,
        hole=0.5, height=380
    )
    fig_donut.update_traces(textposition='outside', textinfo='label+percent')
    fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_donut, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Segment Profiles
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Segment Feature Profiles")

    profile_features = ['Income', 'Total_Spending', 'Total_Purchases',
                        'Recency', 'NumDealsPurchases', 'Total_Children',
                        'Total_Accepted_Campaigns']

    profile = filtered_df.groupby('Segment')[profile_features].mean().round(1)

    # Boxplots
    box_feat = st.selectbox("Select feature to compare across segments:", profile_features)
    fig_box = px.box(
        filtered_df, x='Segment', y=box_feat,
        color='Segment', color_discrete_map=SEGMENT_COLORS,
        title=f"{box_feat} by Segment",
        height=420
    )
    fig_box.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Bar comparison
    st.subheader("📊 Average Feature Values by Segment")
    bar_feat = st.selectbox("Select feature for bar chart:", profile_features, index=0)
    bar_data = filtered_df.groupby('Segment')[bar_feat].mean().reset_index()
    bar_data.columns = ['Segment', 'Average']
    bar_data = bar_data.sort_values('Average', ascending=False)

    fig_bar = px.bar(
        bar_data, x='Segment', y='Average',
        color='Segment', color_discrete_map=SEGMENT_COLORS,
        title=f"Average {bar_feat} per Segment",
        text='Average', height=380
    )
    fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_bar.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Profile table
    st.subheader("📋 Full Segment Profile Table")
    st.dataframe(
        profile.style
            .background_gradient(cmap='Blues', axis=0)
            .format("{:,.1f}"),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════
# TAB 3 — Feature Heatmap
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔥 Normalised Feature Heatmap")
    st.caption("Each value is normalised 0–1 within its feature. Higher = stronger signal.")

    cluster_means = df.groupby('Segment')[FEATURES].mean()
    norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    fig_heat = go.Figure(data=go.Heatmap(
        z=norm.values,
        x=FEATURES,
        y=norm.index.tolist(),
        colorscale='RdYlGn',
        text=np.round(norm.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    fig_heat.update_layout(
        title="Segment × Feature Intensity",
        height=380,
        xaxis=dict(tickangle=-30),
        margin=dict(l=160, r=20, t=60, b=80)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Radar chart
    st.subheader("🕸️ Radar Chart — Segment DNA")
    radar_features = ['Income','Total_Spending','Total_Purchases',
                      'Total_Accepted_Campaigns','NumDealsPurchases']

    radar_data = df.groupby('Segment')[radar_features].mean()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

    fig_radar = go.Figure()
    for seg in all_segments:
        if seg in radar_norm.index:
            vals = radar_norm.loc[seg].tolist()
            vals += [vals[0]]
            cats = radar_features + [radar_features[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself',
                name=seg,
                line_color=SEGMENT_COLORS[seg],
                opacity=0.6
            ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=480,
        title="Segment Feature DNA (normalised)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — Deep Dive
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Income vs Spending by Segment")

    fig_scatter2 = px.scatter(
        filtered_df,
        x='Income', y='Total_Spending',
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.6,
        trendline='ols',
        title="Income vs Total Spending — colored by Segment",
        height=480,
        hover_data=['Recency', 'Total_Purchases']
    )
    fig_scatter2.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_scatter2, use_container_width=True)

    # Recency vs Spending Quadrant
    st.subheader("🎯 Recency–Spending Quadrant")
    rec_med = df['Recency'].median()
    spe_med = df['Total_Spending'].median()

    def quadrant(row):
        if row['Recency'] <= rec_med and row['Total_Spending'] >= spe_med:
            return '👑 Champions'
        elif row['Recency'] > rec_med and row['Total_Spending'] >= spe_med:
            return '⚠️ At Risk'
        elif row['Recency'] <= rec_med and row['Total_Spending'] < spe_med:
            return '🌱 Casual Buyers'
        else:
            return '💤 Inactive'

    plot_df = filtered_df.copy()
    plot_df['Quadrant'] = plot_df.apply(quadrant, axis=1)

    fig_quad = px.scatter(
        plot_df, x='Recency', y='Total_Spending',
        color='Quadrant',
        opacity=0.6, height=460,
        title="Recency–Spending Quadrant Analysis"
    )
    fig_quad.add_vline(x=rec_med, line_dash='dash', line_color='gray', opacity=0.5)
    fig_quad.add_hline(y=spe_med, line_dash='dash', line_color='gray', opacity=0.5)
    fig_quad.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_quad, use_container_width=True)

    # Summary Stats
    st.subheader("📋 Business Summary Table")
    summary = df.groupby('Segment').agg(
        Customers       = ('Segment', 'count'),
        Avg_Income      = ('Income', 'mean'),
        Avg_Spending    = ('Total_Spending', 'mean'),
        Avg_Purchases   = ('Total_Purchases', 'mean'),
        Avg_Recency     = ('Recency', 'mean'),
        Avg_Deals       = ('NumDealsPurchases', 'mean'),
        Avg_Campaigns   = ('Total_Accepted_Campaigns', 'mean'),
    ).round(1).reset_index()

    st.dataframe(
        summary.style.background_gradient(cmap='Greens', subset=['Avg_Income','Avg_Spending']),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════
# TAB 5 — Predict My Segment
# ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔮 Predict Customer Segment")
    st.markdown("Enter a customer's details below to predict which segment they belong to.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 💰 Financial")
        income         = st.number_input("Annual Income (₹)", min_value=0, max_value=200000, value=50000, step=1000)
        total_spending = st.number_input("Total Spending (₹)", min_value=0, max_value=5000, value=500, step=50)
        num_deals      = st.number_input("Deal Purchases", min_value=0, max_value=20, value=2)

    with col2:
        st.markdown("#### 🛒 Purchase Behavior")
        total_purchases  = st.number_input("Total Purchases", min_value=0, max_value=50, value=10)
        num_web          = st.number_input("Web Purchases", min_value=0, max_value=30, value=3)
        num_catalog      = st.number_input("Catalog Purchases", min_value=0, max_value=30, value=2)

    with col3:
        st.markdown("#### 👨‍👩‍👧 Personal")
        recency          = st.number_input("Days Since Last Purchase", min_value=0, max_value=100, value=30)
        total_children   = st.number_input("Total Children", min_value=0, max_value=5, value=1)
        campaigns        = st.number_input("Campaigns Accepted", min_value=0, max_value=6, value=0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict My Segment", use_container_width=True, type="primary")

    if predict_btn:
        # Build input array in same feature order as training
        input_data = np.array([[
            income, recency, total_spending, total_purchases,
            num_deals, total_children, campaigns,
            num_web, num_catalog
        ]])

        # Scale using saved scaler
        input_scaled = scaler.transform(input_data)

        # Predict cluster
        cluster_num  = km.predict(input_scaled)[0]

        # Map cluster number to segment name
        # Build mapping from existing data
        cluster_to_segment = df.groupby('KMeans_Cluster')['Segment'].first().to_dict()
        segment = cluster_to_segment.get(cluster_num, f'Cluster {cluster_num}')
        color   = SEGMENT_COLORS.get(segment, '#888888')
        emoji   = SEGMENT_EMOJI.get(segment, '🎯')
        desc    = SEGMENT_DESC.get(segment, '')

        st.divider()
        st.markdown("### 🎯 Prediction Result")

        res1, res2 = st.columns([1, 2])

        with res1:
            st.markdown(f"""
            <div style="background:{DARK_PANEL}; border: 3px solid {color};
                        border-radius: 16px; padding: 30px; text-align:center;
                        box-shadow: 0 4px 20px {color}44">
                <div style="font-size: 3rem">{emoji}</div>
                <div style="color:{color}; font-size:1.4rem; font-weight:800; margin-top:8px">{segment}</div>
                <div style="color:#888; font-size:0.85rem; margin-top:6px">Cluster {cluster_num}</div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            st.markdown(f"""
            <div style="background:{DARK_PANEL}; border-left: 5px solid {color};
                        border-radius: 8px; padding: 20px;
                        box-shadow: 0 2px 12px rgba(0,0,0,0.3)">
                <div style="font-size:1rem; font-weight:700; color:{color}; margin-bottom:8px">
                    {emoji} What this means:
                </div>
                <div style="color:#cccccc; font-size:0.95rem; line-height:1.7">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PCA scatter showing WHERE this customer sits ──────────────────────
        st.markdown("### 🗺️ Where Does This Customer Sit in the Cluster Map?")
        st.caption("⭐ Star = Your Customer  ·  Dots = All 2,240 existing customers")

        # Project new customer into PCA space
        new_pca = pca_model.transform(input_scaled)
        new_pca1 = new_pca[0, 0]
        new_pca2 = new_pca[0, 1]

        # Build scatter of all existing customers
        fig_loc = go.Figure()

        # Plot each segment as a separate trace
        for seg in all_segments:
            seg_data = df[df['Segment'] == seg]
            fig_loc.add_trace(go.Scatter(
                x=seg_data['PCA1'],
                y=seg_data['PCA2'],
                mode='markers',
                name=seg,
                marker=dict(
                    color=SEGMENT_COLORS[seg],
                    size=5,
                    opacity=0.4
                ),
                hovertemplate=f"<b>{seg}</b><extra></extra>"
            ))

        # Plot the NEW customer as a big star
        fig_loc.add_trace(go.Scatter(
            x=[new_pca1],
            y=[new_pca2],
            mode='markers+text',
            name='⭐ Your Customer',
            text=['  ← You'],
            textposition='middle right',
            textfont=dict(size=13, color=color, family='Arial Black'),
            marker=dict(
                symbol='star',
                size=22,
                color=color,
                line=dict(color='white', width=2),
                opacity=1.0
            ),
            hovertemplate=f"<b>Your Customer</b><br>Predicted: {segment}<extra></extra>"
        ))

        fig_loc.update_layout(
            plot_bgcolor=DARK_BG,
            paper_bgcolor=DARK_PANEL,
            font=dict(color='white'),
            height=480,
            title=dict(
                text=f"Customer Predicted as: <b>{emoji} {segment}</b>",
                font=dict(size=14, color=color)
            ),
            xaxis=dict(showgrid=True, gridcolor=DARK_GRID, color='#aaaaaa',
                      zerolinecolor=DARK_GRID, title='PC1'),
            yaxis=dict(showgrid=True, gridcolor=DARK_GRID, color='#aaaaaa',
                      zerolinecolor=DARK_GRID, title='PC2'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                       xanchor='right', x=1, font=dict(color='white')),
        )
        st.plotly_chart(fig_loc, use_container_width=True)

        # Marketing recommendation
        st.markdown("### 💡 Recommended Marketing Action")
        actions = {
            'VIP Champions':        '🎁 Offer exclusive loyalty rewards and early access to premium products. Prioritize retention — this customer generates the highest value.',
            'Affluent Loyalists':   '📦 Upsell premium bundles and catalog offers. This customer responds well to campaigns — target with personalized high-value promotions.',
            'Deal-Seeking Families':'🏷️ Engage with targeted discount campaigns and family bundle offers. Avoid heavy margin discounts — use time-limited deals to drive urgency.',
            'Budget Families':      '💸 Focus on value-for-money messaging and essential product promotions. Cost-effective engagement channels like email work best.',
        }
        action = actions.get(segment, 'Analyze further to determine best strategy.')
        st.info(action)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem'>
    🎯 Customer Segmentation Dashboard · DS Group 4 · K-Means Clustering (K=4)
</div>
""", unsafe_allow_html=True)
