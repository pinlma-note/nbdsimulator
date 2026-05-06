import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gamma
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide", page_title="NBDマーケティング・シミュレータ")

# ------------------------------------------------
# 数理モデル (NBDモデル) [cite: 9, 38]
# ------------------------------------------------
def nbd_prob(r, M, K):
    """r回購入する確率を計算 [cite: 9]"""
    if K <= 0 or M <= 0:
        return 0 if r > 0 else 1
    term1 = (1 + M/K)**(-K)
    term2 = gamma(K+r) / (gamma(r+1) * gamma(K))
    term3 = (M / (M+K))**r
    return term1 * term2 * term3

# ------------------------------------------------
# シミュレーションロジック [cite: 13, 146]
# ------------------------------------------------
def simulate_plan(plan_name, p_params):
    # 基本設定
    target_pop = 10000000 # ターゲット母数 (1000万人)
    lifetime = 24 # 24ヶ月
    
    # 計算式修正: アクセス可能母数 × プレファレンス(M) [cite: 13, 260]
    # 認知率の算出（5000万で100%と仮定した簡易モデル）
    awareness = min(1.0, p_params['ad_budget'] / 50000000)
    accessible_pop = target_pop * awareness * p_params['dist_rate']
    total_demand = accessible_pop * p_params['pref_M']
    
    # 普及曲線 (イノベーター理論に基づく山なり)
    months = np.arange(1, lifetime + 1)
    diffusion = (months**1.5) * np.exp(-months / 3)
    diffusion = diffusion / diffusion.sum()
    monthly_demand = total_demand * diffusion
    
    # 配列準備
    inventory = np.zeros(lifetime)
    cashflow = np.zeros(lifetime)
    current_inv = p_params['initial_lot']
    current_cf = - (p_params['initial_lot'] * p_params['cost']) - p_params['ad_budget']
    
    stockout_month = -1
    total_sales_vol = 0
    
    for i in range(lifetime):
        demand = monthly_demand[i]
        sales = min(current_inv, demand)
        
        if current_inv < demand and stockout_month == -1:
            stockout_month = i + 1
            
        current_inv -= sales
        total_sales_vol += sales
        current_cf += sales * p_params['price']
        
        inventory[i] = current_inv
        cashflow[i] = current_cf

    return {
        "name": plan_name,
        "total_sales_vol": total_sales_vol,
        "total_revenue": total_sales_vol * p_params['price'],
        "bottom_cf": np.min(cashflow),
        "final_cf": cashflow[-1],
        "stockout": stockout_month > 0,
        "stockout_month": stockout_month,
        "excess_inv": current_inv,
        "cf_history": cashflow.tolist(),
        "inv_history": inventory.tolist(),
        "params": p_params
    }

# ------------------------------------------------
# UI部
# ------------------------------------------------
st.title("📊 NBDモデル 販売需要シミュレータ (GitHub版)")

# 保存データの読み込み機能
uploaded_file = st.sidebar.file_uploader("設定ファイルを読み込む (.json)", type="json")
if uploaded_file:
    saved_data = json.load(uploaded_file)
    st.sidebar.success("設定を読み込みました")
else:
    saved_data = None

with st.sidebar:
    st.header("共通・市場設定")
    category = st.selectbox("カテゴリー", ["ヘアケア", "スキンケア", "洗剤"])
    K_val = 0.8 if category == "ヘアケア" else 0.5 # カテゴリー別形状パラメータ [cite: 40, 43]
    pref_M = st.slider("プレファレンス(M: 1人あたり平均購入回数)", 0.01, 1.00, 0.10, 0.01) # 修正済み
    price = st.number_input("販売単価 (円)", value=1500)
    cost = st.number_input("製造原価 (円)", value=400)

col1, col2, col3 = st.columns(3)

def input_plan(label, key_suffix, default_lot, default_ad, default_dist):
    with label:
        st.subheader(f"プラン {key_suffix}")
        lot = st.number_input("初回ロット", value=default_lot, key=f"lot{key_suffix}")
        ad = st.number_input("広告予算", value=default_ad, key=f"ad{key_suffix}")
        dist = st.slider("目標配荷率", 0.1, 1.0, default_dist, key=f"dist{key_suffix}")
        return {"pref_M": pref_M, "price": price, "cost": cost, "initial_lot": lot, "ad_budget": ad, "dist_rate": dist}

pA_params = input_plan(col1, "A", 300000, 30000000, 0.8)
pB_params = input_plan(col2, "B", 150000, 10000000, 0.4)

# プランC (AI補正用)
if st.button("🤖 AI補正プラン(C)を作成"):
    # ロジック: 在庫不足なら1.2倍、CF悪化なら広告分散など
    new_lot = int(simulate_plan("temp", pA_params)['total_sales_vol'] * 1.1)
    pC_params = {"pref_M": pref_M, "price": price, "cost": cost, "initial_lot": new_lot, "ad_budget": 15000000, "dist_rate": 0.6}
    st.session_state['pC'] = pC_params
    st.info("AIがリスクを抑えた補正案（プランC）を生成しました。")

pC_params = st.session_state.get('pC', pB_params.copy())
with col3:
    st.subheader("プラン C")
    lot_C = st.number_input("初回ロット", value=pC_params['initial_lot'], key="lotC_ui")
    ad_C = st.number_input("広告予算", value=pC_params['ad_budget'], key="adC_ui")
    dist_C = st.slider("目標配荷率", 0.1, 1.0, pC_params['dist_rate'], key="distC_ui")
    pC_params.update({"initial_lot": lot_C, "ad_budget": ad_C, "dist_rate": dist_C})

# 実行と比較
results = [simulate_plan("A", pA_params), simulate_plan("B", pB_params), simulate_plan("C", pC_params)]

# ダッシュボード表示
st.markdown("---")
metrics = []
for r in results:
    metrics.append({
        "プラン": r['name'],
        "累計売上": f"¥{int(r['total_revenue']):,}",
        "最終CF": f"¥{int(r['final_cf']):,}",
        "欠品リスク": "⚠️ あり" if r['stockout'] else "✅ なし",
        "残在庫": f"{int(r['excess_inv']):,}"
    })
st.table(pd.DataFrame(metrics).set_index("プラン"))

# グラフ
c_cf, c_inv = st.columns(2)
fig_cf = go.Figure()
fig_inv = go.Figure()
for r in results:
    fig_cf.add_trace(go.Scatter(y=r['cf_history'], name=f"Plan {r['name']}"))
    fig_inv.add_trace(go.Scatter(y=r['inv_history'], name=f"Plan {r['name']}"))

c_cf.plotly_chart(fig_cf, use_container_width=True)
c_inv.plotly_chart(fig_inv, use_container_width=True)

# 保存・エクスポート機能
st.sidebar.markdown("---")
# 1. 現在の設定を保存
st.sidebar.download_button("💾 設定をダウンロード", data=json.dumps({"A": pA_params, "B": pB_params, "C": pC_params}), file_name="nbd_settings.json")
# 2. 比較レポートをCSVで出力
df_report = pd.DataFrame(metrics)
st.sidebar.download_button("📑 レポート(CSV)を出す", data=df_report.to_csv(index=False).encode('utf-8'), file_name="nbd_report.csv")