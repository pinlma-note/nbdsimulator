import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gamma
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide", page_title="NBDマーケティング・シミュレータ", page_icon="📊")

# ------------------------------------------------
# マスタデータ定義
# ------------------------------------------------
CATEGORY_MASTERS = {
    "ヘアケア > シャンプー": {"freq": 4.0, "k": 0.8},
    "スキンケア > 化粧水": {"freq": 3.0, "k": 0.5},
    "UVケア > 日焼け止め": {"freq": 2.0, "k": 0.6},
    "洗濯・仕上げ剤 > 液体洗剤": {"freq": 10.0, "k": 0.4},
    "バス用品 > ボディソープ": {"freq": 5.0, "k": 0.5}
}

# ------------------------------------------------
# ロジック関数
# ------------------------------------------------
def get_seasonality(start_month, lifetime_months, category):
    if any(k in category for k in ["UV", "シャンプー", "ボディソープ"]):
        base_season = np.array([0.8, 0.8, 0.9, 1.1, 1.3, 1.5, 1.5, 1.4, 1.0, 0.9, 0.8, 0.8])
    elif any(k in category for k in ["保湿", "スキンケア"]):
        base_season = np.array([1.3, 1.2, 1.0, 0.9, 0.8, 0.8, 0.8, 0.9, 1.1, 1.3, 1.4, 1.4])
    else:
        base_season = np.ones(12)
    seasons = [base_season[(start_month - 1 + i) % 12] for i in range(lifetime_months)]
    return np.array(seasons)

def simulate_plan(plan_name, p_params, env_params):
    target_pop = 10000000 
    lifetime = 24 
    max_stores = 20000 
    
    cat = env_params['category']
    channel = env_params['channel_type']
    base_M = p_params['pref_M']
    price = env_params['price']
    cost = env_params['cost']
    
    # バリエーションによるプレファレンス分散
    var_count = p_params['variations']
    M_total = base_M * (var_count ** 0.6)
    M_indiv = (M_total / var_count) * 0.9 if var_count > 1 else M_total 
    
    # セット率
    trt_rate = 0.85 if p_params['has_trt'] else 0.0
    trial_rate = 0.50 if p_params['has_trial'] else 0.0
    
    # 需要計算
    awareness = min(1.0, p_params['ad_budget'] / 50000000)
    accessible_pop = target_pop * awareness * p_params['dist_rate']
    anchor_total_demand = accessible_pop * M_total
    
    months = np.arange(1, lifetime + 1)
    diffusion = (months**1.5) * np.exp(-months / 3)
    diffusion = diffusion / diffusion.sum() 
    seasonality = get_seasonality(env_params['start_month'], lifetime, cat)
    
    monthly_demand = anchor_total_demand * diffusion * seasonality
    monthly_demand = monthly_demand * (anchor_total_demand / monthly_demand.sum()) 
    
    # 在庫・CF計算
    inventory = np.zeros(lifetime)
    cashflow = np.zeros(lifetime)
    sales_per_store = np.zeros(lifetime)
    
    anchor_lot = p_params['initial_lot']
    trt_lot = anchor_lot * trt_rate
    trial_lot = anchor_lot * trial_rate
    
    initial_investment = (anchor_lot * cost) + (trt_lot * cost) + (trial_lot * cost * 0.2) + p_params['ad_budget']
    current_inv = anchor_lot 
    current_cf = - initial_investment
    
    stockout_month = -1
    total_sales_vol = 0
    actual_stores = max(1, max_stores * p_params['dist_rate'])
    
    for i in range(lifetime):
        demand = monthly_demand[i]
        sales = min(current_inv, demand)
        if current_inv < demand and stockout_month == -1:
            stockout_month = i + 1
        current_inv -= sales
        total_sales_vol += sales
        monthly_rev = (sales * price) + (sales * trt_rate * price) + (sales * trial_rate * (price * 0.1))
        current_cf += monthly_rev
        inventory[i] = current_inv
        cashflow[i] = current_cf
        sales_per_store[i] = sales / actual_stores

    # リスク判定
    avg_monthly_rev_per_store = (total_sales_vol * price * (1 + trt_rate + (trial_rate * 0.1)) / lifetime) / actual_stores
    if channel == "ドラッグストア (DG)":
        shelf_drop_risk = avg_monthly_rev_per_store < 15000
    else:
        shelf_drop_risk = avg_monthly_rev_per_store < 40000
    indiv_risk = (M_indiv < 0.05) and var_count > 1
    
    return {
        "name": plan_name,
        "total_revenue": current_cf + initial_investment - p_params['ad_budget'],
        "final_cf": current_cf,
        "bottom_cf": np.min(cashflow),
        "stockout": stockout_month > 0,
        "stockout_month": stockout_month,
        "shelf_drop": shelf_drop_risk,
        "indiv_risk": indiv_risk,
        "excess_inv": current_inv,
        "recommended_lots": {"Main": int(anchor_lot), "Refill": int(trt_lot), "Trial": int(trial_lot)},
        "cf_history": cashflow.tolist(),
        "inv_history": inventory.tolist(),
        "params": p_params
    }

# ------------------------------------------------
# UI構築
# ------------------------------------------------
st.title("📊 NBDフル機能シミュレータ")

# --- 保存・読込 ---
with st.sidebar:
    st.header("💾 データ管理")
    uploaded_file = st.file_uploader("JSON設定を読み込む", type="json")
    if uploaded_file:
        st.success("読込完了")

# --- STEP1: 市場環境 ---
st.header("⚙️ STEP1: カテゴリ・市場・M値設定")
c1, c2, c3 = st.columns(3)
with c1:
    cat_minor = st.selectbox("小カテゴリ", list(CATEGORY_MASTERS.keys()))
    cat_info = CATEGORY_MASTERS[cat_minor]
with c2:
    start_month = st.number_input("発売開始月", 1, 12, 3)
with c3:
    channel_type = st.selectbox("主戦場チャネル", ["ドラッグストア (DG)", "バラエティショップ (VS)"])

m_mode = st.radio("M値の設定方法", ["シェアから逆算", "直接入力"], horizontal=True)
if m_mode == "シェアから逆算":
    target_share = st.slider("目標ユニットシェア (%)", 0.1, 10.0, 2.0)
    calculated_m = cat_info['freq'] * (target_share / 100)
    st.info(f"算出プレファレンス(M): {calculated_m:.3f}")
else:
    calculated_m = st.number_input("直接入力(M)", 0.01, 1.00, 0.10)

st.divider()

# --- STEP2: 戦略とコスト ---
st.header("📝 STEP2: ブランド戦略とコスト設定")
c_price, c_cost, c_brand = st.columns(3)
price = c_price.number_input("単価(円)", value=1500)
cost = c_cost.number_input("原価(円)", value=400)
brand_name = c_brand.text_input("ブランド名", "New Brand")

env_params = {"category": cat_minor, "start_month": start_month, "channel_type": channel_type, "price": price, "cost": cost}

colA, colB = st.columns(2)
def input_plan(label, key, def_lot, def_ad, def_dist, m_val):
    with st.container(border=True):
        st.subheader(label)
        p_m = st.number_input(f"M値 - {key}", value=m_val, key=f"m{key}")
        vars = st.number_input("バリエーション数", 1, 5, 1, key=f"v{key}")
        t1, t2 = st.columns(2)
        has_trt = t1.checkbox("連動SKUあり", value=True, key=f"t{key}")
        has_trial = t2.checkbox("トライアルSKUあり", value=False, key=f"tr{key}")
        lot = st.number_input("初回ロット", value=def_lot, key=f"l{key}")
        ad = st.number_input("広告予算", value=def_ad, key=f"a{key}")
        dist = st.slider("目標配荷率", 0.1, 1.0, def_dist, key=f"d{key}")
        return {"pref_M": p_m, "variations": vars, "has_trt": has_trt, "has_trial": has_trial, "initial_lot": lot, "ad_budget": ad, "dist_rate": dist}

with colA:
    pA = input_plan("プラン A", "A", 200000, 30000000, 0.8, calculated_m)
with colB:
    pB = input_plan("プラン B", "B", 80000, 10000000, 0.4, calculated_m)

# --- プランC生成 ---
if st.button("🤖 AI補正プラン(C)を作成", type="primary"):
    res_A = simulate_plan("temp", pA, env_params)
    new_lot = int(res_A['total_revenue'] / price * 0.5) if res_A['stockout'] else pA['initial_lot']
    st.session_state['pC'] = {"pref_M": calculated_m, "variations": 1, "has_trt": True, "has_trial": True, "initial_lot": max(50000, new_lot), "ad_budget": 15000000, "dist_rate": 0.5}

# 実行
current_plans = [("A", pA), ("B", pB)]
if 'pC' in st.session_state:
    current_plans.append(("C", st.session_state['pC']))

results = [simulate_plan(name, params, env_params) for name, params in current_plans]

st.divider()

# --- STEP4: 結果・エクスポート ---
st.header("📈 シミュレーション結果とエクスポート")

# KPIテーブル
df_res = pd.DataFrame([{
    "プラン": r['name'],
    "累計売上": f"¥{int(r['total_revenue']):,}",
    "ボトムCF": f"¥{int(r['bottom_cf']):,}",
    "欠品発生": "⚠️" if r['stockout'] else "✅",
    "棚落ちリスク": "❌" if r['shelf_drop'] else "✅",
    "カニバリリスク": "⚠️" if r['indiv_risk'] else "✅",
    "残在庫": f"{int(r['excess_inv']):,}"
} for r in results])
st.table(df_res)

# ロット詳細
with st.expander("📦 推奨生産ロット詳細"):
    for r in results:
        st.write(f"**プラン {r['name']}**: メイン {r['recommended_lots']['Main']:,} / 連動 {r['recommended_lots']['Refill']:,} / トライアル {r['recommended_lots']['Trial']:,}")

# グラフ
c1, c2 = st.columns(2)
with c1:
    st.subheader("キャッシュフロー推移")
    f_cf = go.Figure()
    for r in results: f_cf.add_trace(go.Scatter(y=r['cf_history'], name=f"Plan {r['name']}"))
    st.plotly_chart(f_cf, use_container_width=True)
with c2:
    st.subheader("在庫推移")
    f_inv = go.Figure()
    for r in results: f_inv.add_trace(go.Scatter(y=r['inv_history'], name=f"Plan {r['name']}"))
    st.plotly_chart(f_inv, use_container_width=True)

# エクスポート
st.sidebar.divider()
st.sidebar.download_button("💾 設定をJSON保存", data=json.dumps({"params": [p for n, p in current_plans], "env": env_params}), file_name="nbd_config.json")
st.sidebar.download_button("📑 レポートをCSV出力", data=df_res.to_csv(index=False).encode('utf-8-sig'), file_name="nbd_report.csv")
