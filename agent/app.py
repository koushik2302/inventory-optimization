"""
app.py
------
Streamlit frontend for the Inventory Optimization LLM Agent.

Layout:
  - Left panel:  Chat interface with the Qwen 3 8B agent
  - Right panel: Dynamic visualizations (ABC-XYZ heatmap, cost charts)
  - Sidebar:     Store selector, config options

Usage:
    streamlit run agent/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory Optimization | ABC-XYZ Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
try:
    from tools import (
        get_classification,
        calculate_safety_stock,
        simulate_policy_change,
        get_overstock_alerts,
        get_promotion_adjustment,
        compare_policies_summary,
    )
    from agent import InventoryAgent
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    IMPORT_ERROR = str(e)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "inventory.db"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Main theme */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #1a1d2e;
    border-right: 1px solid #2d3250;
}

/* Chat messages */
.user-msg {
    background: #2d3250;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8eaf6;
    font-size: 0.95rem;
}
.agent-msg {
    background: #1e2140;
    border-left: 3px solid #7c83fd;
    border-radius: 4px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8eaf6;
    font-size: 0.95rem;
}
.thinking-box {
    background: #12141f;
    border: 1px dashed #3d4466;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #8892b0;
    font-size: 0.82rem;
    font-style: italic;
}
.tool-badge {
    display: inline-block;
    background: #2d3250;
    border: 1px solid #7c83fd;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #7c83fd;
    margin: 2px;
}
.metric-card {
    background: #1a1d2e;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
h1, h2, h3 {
    color: #e8eaf6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📦 Inventory Optimizer")
    st.markdown("---")

    # Store selector
    available_stores = []
    if DB_PATH.exists() and TOOLS_AVAILABLE:
        try:
            conn = sqlite3.connect(DB_PATH)
            stores = pd.read_sql(
                "SELECT DISTINCT store_nbr FROM abc_xyz_matrix ORDER BY store_nbr",
                conn
            )["store_nbr"].tolist()
            conn.close()
            available_stores = stores
        except Exception:
            available_stores = list(range(1, 7))
    else:
        available_stores = list(range(1, 7))

    selected_store = st.selectbox(
        "🏪 Select Store",
        available_stores or [1, 2, 3, 4, 5, 6],
        index=0,
        help="Choose a store to analyze"
    )

    st.markdown("---")

    # Model status
    agent_available = False
    if TOOLS_AVAILABLE:
        agent = InventoryAgent()
        agent_available = agent.check_ollama_connection()

    if agent_available:
        st.success("🟢 Qwen 3 8B — Connected")
    else:
        st.warning("🟡 Ollama not running — Direct tool mode")
        st.caption("Start with: `ollama serve`")
        st.caption("Pull model: `ollama pull qwen3:8b`")

    st.markdown("---")

    # Quick actions
    st.markdown("**Quick Actions**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            if "agent_instance" in st.session_state:
                st.session_state.agent_instance.reset_conversation()
            st.rerun()
    with col2:
        if st.button("📊 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Example queries
    st.markdown("**Example Queries**")
    examples = [
        f"Show me the ABC-XYZ matrix for store {selected_store}",
        f"What are the overstock risks in store {selected_store}?",
        f"What happens if I drop CZ service to 80% in store {selected_store}?",
        f"Beverages promotion next week — what buffer does store {selected_store} need?",
        f"Compare all 3 policies for store {selected_store}",
    ]
    for ex in examples:
        if st.button(ex[:45] + "...", key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.pending_message = ex


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_abc_xyz_matrix(store_id: int):
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM abc_xyz_matrix WHERE store_nbr = ?",
            conn, params=(store_id,)
        )
        conn.close()
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_policy_comparison(store_id: int):
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM safety_stock_results WHERE store_nbr = ?",
            conn, params=(store_id,)
        )
        conn.close()
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_sensitivity_curve():
    curve_path = PROJECT_ROOT / "data" / "cleaned" / "sensitivity_service_level.csv"
    if curve_path.exists():
        return pd.read_csv(curve_path)
    return None


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------
def plot_abc_xyz_heatmap(df: pd.DataFrame, store_id: int):
    """Plot ABC-XYZ 3×3 heatmap showing SKU counts."""
    if df is None or df.empty:
        return None

    matrix = df.groupby(["abc_class", "xyz_class"]).agg(
        num_skus=("item_nbr", "count"),
        revenue_share=("revenue_proxy", "sum"),
    ).reset_index()

    # Create pivot for heatmap
    pivot_skus = matrix.pivot(index="abc_class", columns="xyz_class", values="num_skus").fillna(0)
    pivot_rev  = matrix.pivot(index="abc_class", columns="xyz_class", values="revenue_share").fillna(0)

    # Reorder axes
    for p in [pivot_skus, pivot_rev]:
        p.index    = pd.CategoricalIndex(p.index, categories=["A", "B", "C"], ordered=True)
        p.columns  = pd.CategoricalIndex(p.columns, categories=["X", "Y", "Z"], ordered=True)
        p.sort_index(inplace=True)
        p.sort_index(axis=1, inplace=True)

    # Service level annotations
    sl_map = {
        ("A","X"): "99%", ("A","Y"): "97%", ("A","Z"): "95%",
        ("B","X"): "95%", ("B","Y"): "93%", ("B","Z"): "90%",
        ("C","X"): "90%", ("C","Y"): "88%", ("C","Z"): "85%",
    }

    # Custom text for cells
    text = []
    for abc in ["A","B","C"]:
        row_text = []
        for xyz in ["X","Y","Z"]:
            n = int(pivot_skus.loc[abc, xyz]) if abc in pivot_skus.index and xyz in pivot_skus.columns else 0
            sl = sl_map.get((abc,xyz), "")
            row_text.append(f"{abc}{xyz}<br>{n} SKUs<br>SL: {sl}")
        text.append(row_text)

    z = pivot_skus.values.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["X (Stable)", "Y (Moderate)", "Z (Erratic)"],
        y=["A (High Rev)", "B (Mid Rev)", "C (Low Rev)"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        colorscale=[[0, "#1a1d2e"], [0.5, "#3d4466"], [1, "#7c83fd"]],
        showscale=True,
        colorbar=dict(title="SKU Count", tickfont=dict(color="#e8eaf6")),
    ))

    fig.update_layout(
        title=dict(text=f"ABC-XYZ Matrix — Store {store_id}", font=dict(color="#e8eaf6", size=16)),
        xaxis=dict(title="Demand Variability (XYZ)", title_font=dict(color="#8892b0"), tickfont=dict(color="#e8eaf6")),
        yaxis=dict(title="Revenue Contribution (ABC)", title_font=dict(color="#8892b0"), tickfont=dict(color="#e8eaf6")),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_policy_comparison(df: pd.DataFrame):
    """Bar chart: holding cost per policy."""
    if df is None or df.empty:
        return None

    totals = {
        "Uniform (95%)": df["hc_uniform"].sum(),
        "3-Tier (ABC)": df["hc_3tier"].sum(),
        "9-Cell (ABC-XYZ)": df["hc_9cell"].sum(),
    }

    fig = go.Figure(go.Bar(
        x=list(totals.keys()),
        y=list(totals.values()),
        marker_color=["#e57373", "#ffb74d", "#81c784"],
        text=[f"{v:,.0f}" for v in totals.values()],
        textposition="auto",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title=dict(text="Annual Holding Cost by Policy", font=dict(color="#e8eaf6", size=14)),
        xaxis=dict(tickfont=dict(color="#e8eaf6")),
        yaxis=dict(title="Holding Cost (unit-cost proxy)", tickfont=dict(color="#e8eaf6")),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_sensitivity_curve(df: pd.DataFrame):
    """Service level vs. holding cost curve."""
    if df is None or df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["service_level"] * 100,
        y=df["total_annual_holding_cost"],
        name="Holding Cost",
        line=dict(color="#7c83fd", width=2),
        fill="tozeroy",
        fillcolor="rgba(124,131,253,0.1)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["service_level"] * 100,
        y=df["total_safety_stock_units"],
        name="Safety Stock (units)",
        line=dict(color="#81c784", width=2, dash="dot"),
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text="Cost-Service Level Tradeoff Curve", font=dict(color="#e8eaf6", size=14)),
        xaxis=dict(title="Service Level (%)", tickfont=dict(color="#e8eaf6")),
        yaxis=dict(title="Annual Holding Cost", tickfont=dict(color="#e8eaf6")),
        yaxis2=dict(title="Safety Stock (units)", tickfont=dict(color="#81c784")),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e8eaf6"),
        legend=dict(font=dict(color="#e8eaf6")),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = InventoryAgent() if TOOLS_AVAILABLE else None

# Check for pending message (from sidebar buttons)
pending = st.session_state.pop("pending_message", None)

# Header
st.markdown(f"## 📦 Inventory Optimization — Store {selected_store}")
st.markdown("---")

# Check if data is available
data_ready = DB_PATH.exists() and TOOLS_AVAILABLE

if not TOOLS_AVAILABLE:
    st.error(f"⚠️  Could not import tools module: `{IMPORT_ERROR}`")
    st.info("Run the analysis pipeline first:\n```\npython scripts/01_load_data.py\npython scripts/02_classify_skus.py\npython scripts/03_safety_stock.py\n```")
elif not DB_PATH.exists():
    st.warning("⚠️  Database not found. Run the data pipeline first.")
    st.code("python scripts/download_data.py\npython scripts/01_load_data.py\npython scripts/02_classify_skus.py\npython scripts/03_safety_stock.py")

# Two-column layout
left_col, right_col = st.columns([1, 1], gap="medium")

# ---- LEFT: Chat Interface ----
with left_col:
    st.markdown("### 💬 Inventory Advisor")

    # Chat history display
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="agent-msg">
            👋 Hello! I'm your inventory optimization advisor.<br><br>
            I can help you with:<br>
            • <b>ABC-XYZ classification</b> — which SKUs are in each cell<br>
            • <b>Safety stock & reorder points</b> — with assumptions stated<br>
            • <b>Policy comparison</b> — uniform vs 3-tier vs 9-cell<br>
            • <b>Overstock alerts</b> — SKUs costing more than they earn<br>
            • <b>Promotion adjustments</b> — pre-position stock for promos<br><br>
            Try asking: <i>"Show me the ABC-XYZ matrix for this store"</i>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

                # Show tool calls used
                if msg.get("tool_calls"):
                    tools_used = " ".join(
                        f'<span class="tool-badge">🔧 {tc["tool"]}</span>'
                        for tc in msg["tool_calls"]
                    )
                    st.markdown(f"<small>{tools_used}</small>", unsafe_allow_html=True)

                # Show thinking (collapsible)
                if msg.get("thinking"):
                    with st.expander("🧠 Model thinking", expanded=False):
                        st.markdown(f'<div class="thinking-box">{msg["thinking"]}</div>',
                                    unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about inventory, safety stock, promotions...")
    if pending:
        user_input = pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🤔 Analyzing..."):
            if data_ready and st.session_state.agent_instance:
                agent_inst = st.session_state.agent_instance
                if agent_available:
                    result = agent_inst.chat(user_input)
                else:
                    result = agent_inst.chat_without_llm(user_input)

                response_text = result.get("response", "No response generated.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": result.get("tool_calls", []),
                    "thinking": result.get("thinking", ""),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️  Data not ready. Please run the analysis pipeline first.",
                })

        st.rerun()


# ---- RIGHT: Visualizations ----
with right_col:
    st.markdown("### 📊 Analytics Dashboard")

    if data_ready:
        df_matrix = load_abc_xyz_matrix(selected_store)
        df_ss     = load_policy_comparison(selected_store)
        df_curve  = load_sensitivity_curve()

        # Tabs for different charts
        tab1, tab2, tab3 = st.tabs(["🗂️ ABC-XYZ Matrix", "💰 Policy Cost", "📈 Sensitivity"])

        with tab1:
            if df_matrix is not None and not df_matrix.empty:
                fig = plot_abc_xyz_heatmap(df_matrix, selected_store)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Cell distribution table
                summary = df_matrix.groupby(["abc_class","xyz_class","cell"]).agg(
                    num_skus=("item_nbr","count"),
                    revenue_share=("revenue_proxy","sum"),
                    avg_cv=("cv","mean"),
                    service_level=("service_level_9cell","first"),
                ).reset_index()
                summary["rev_pct"] = (summary["revenue_share"] / summary["revenue_share"].sum() * 100).round(1)
                st.dataframe(
                    summary[["cell","num_skus","rev_pct","avg_cv","service_level"]].rename(columns={
                        "cell":"Cell","num_skus":"SKUs","rev_pct":"Rev%","avg_cv":"Avg CV","service_level":"SL"
                    }),
                    use_container_width=True,
                    height=220,
                )
            else:
                st.info("Run the analysis pipeline to see the ABC-XYZ matrix.")

        with tab2:
            if df_ss is not None and not df_ss.empty:
                fig2 = plot_policy_comparison(df_ss)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                hc_uniform = df_ss["hc_uniform"].sum()
                hc_9cell   = df_ss["hc_9cell"].sum()
                savings    = hc_uniform - hc_9cell
                with m1:
                    st.metric("Uniform HC", f"{hc_uniform:,.0f}")
                with m2:
                    st.metric("9-Cell HC", f"{hc_9cell:,.0f}")
                with m3:
                    st.metric("Cost Change", f"{savings:+,.0f}", delta=f"{savings/max(hc_uniform,1)*100:+.1f}%")
            else:
                st.info("Run scripts/03_safety_stock.py to see policy comparison.")

        with tab3:
            if df_curve is not None and not df_curve.empty:
                fig3 = plot_sensitivity_curve(df_curve)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                st.caption("Tradeoff between service level and holding cost (uniform policy baseline).")
            else:
                st.info("Run scripts/03_safety_stock.py to generate sensitivity data.")
    else:
        st.info("Run the data pipeline to populate the dashboard.")
        st.code("""
# Step 1: Download data
python scripts/download_data.py

# Step 2: Load into SQLite
python scripts/01_load_data.py

# Step 3: Classify SKUs
python scripts/02_classify_skus.py

# Step 4: Compute safety stock
python scripts/03_safety_stock.py
        """)
