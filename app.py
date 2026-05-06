import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gamma
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide", page_title="NBDマーケティング・シミュレータ", page_icon="📊")

# ------------------------------------------------
# 1. 数理モデル & 計算ロジック
# ------------------------------------------------

def calculate_m_from_search(volume, category_name):
    """
    検索ボリュームからプレファレンスMを推定するロジック
    M = (年間検索総数 / ターゲット母数) * カテゴリー転換係数
    """
    target_pop = 10000000 # ターゲット母数 1000万人
    annual_volume = volume * 12
    
    # カテゴリー別の転換係数 (エボークトセットへの入りやすさ)
    # 洗剤などは検索が少なくても習慣で買うが、美容液は検索と購買が強く連動する
    conversion_factors = {
        "ヘアケア > シャンプー": 0.5,
        "スキンケア > 化粧水": 0.7,
        "UVケア > 日焼け止め": 0.6,
        "洗濯・仕上げ剤 > 液体洗剤": 0.3,
        "バス用品 > ボディソープ": 0.4
    }
    factor = conversion_factors.get(category_name, 0.5)
    
    estimated_m = (annual_volume / target_pop) * factor
    # 現実的な範囲 (0.01〜1.0) にクリップ
    return float(np.clip(estimated_m, 0.01, 1.0))

def get_seasonality(start_month, lifetime_months, category):
    if "UV" in category or "シャンプー" in category or "ボディソープ" in category:
        base_season = np.array([0.8, 0.8, 0.9, 1.1, 1.3, 1.5, 1.5, 1.4, 1.0, 0.9, 0.8, 0.8])
    elif "保湿" in category or "スキンケア" in category:
        base_season = np.array([1.3, 1.2, 1.0, 0.9, 0.8, 0.8, 0.8, 0.9, 1.1, 1.3, 1.4, 1.4])
    else:
        base_season = np.ones(12)
        
    seasons = []
    current_m = start_month - 1
    for _ in range(lifetime_months):
        seasons.append(base_season[current_m % 12])
        current_m += 1
    return np.array(seasons)

def simulate_plan(plan_name, p_params, env_params):
    target_pop = 10000000 
    lifetime = 24 
    max_stores = 20000 
    
    cat = env_params['category']
    channel = env_params['channel_type']
    base_M = p_params['pref_M'] # 各プラン固有のMを使用
    price = env_params['price']
    cost = env_params['cost']
    
    var_count = p_params['variations']
    M_total = base_M * (var_count ** 0.6)
    M_indiv = (M_total / var_count) * 0.9 if var_count > 1 else M_total 
    
    trt_rate = 0.85 if p_params['has_trt'] else 0.0
    trial_rate = 0.50 if p_params['has_trial'] else 0.0
    
    awareness = min(1.0, p_params['ad_budget'] / 50000000)
    accessible_pop = target_pop * awareness * p_params['dist_rate']
    anchor_total_demand = accessible_pop * M_total
    
    months = np.arange(1, lifetime + 1)
    diffusion = (months**1.5) * np.exp(-months / 3)
    diffusion = diffusion / diffusion.sum() 
    seasonality = get_seasonality(env_params['start_month'], lifetime, cat)
    
    monthly_demand = anchor_total_demand * diffusion * seasonality
    monthly_demand = monthly_demand * (anchor_total_demand / monthly_demand.sum()) 
    
    inventory = np.zeros(lifetime)
    cashflow = np.zeros(lifetime)
    sales_per_store = np.zeros(lifetime)
    
    anchor_lot = p_params['initial_lot']
    trt_lot = anchor_lot * trt_rate
    trial_lot = anchor_lot * trial_rate
    total_initial_lot = anchor_lot + trt_lot + trial_lot
    
    current_inv = anchor_lot 
    initial_cost = (anchor_lot * cost) + (trt_lot * cost) + (trial_lot * cost * 0.2) + p_params['ad_budget']
    current_cf = - initial_cost
    
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

    avg_monthly_rev_per_store = (total_sales_vol * price * (1 + trt_rate + (trial_rate * 0.1)) / lifetime) / actual_stores
    if channel == "ドラッグストア (DG)":
        shelf_drop_risk = avg_monthly_rev_per_store < 15000
    else:
        shelf_drop_risk = avg_monthly_rev_per_store < 40000
    indiv_risk = (M_indiv < 0.05) and var_count > 1
    
    return {
        "name": plan_name,
        "total_sales_vol": total_sales_vol * (1 + trt_rate + trial_rate),
        "total_revenue": current_cf + initial_cost - p_params['ad_budget'],
        "bottom_cf": np.min(cashflow),
        "final_cf": cashflow[-1],
        "stockout": stockout_month > 0,
        "stockout_month": stockout_month,
        "shelf_drop": shelf_drop_risk,
        "indiv_risk": indiv_risk,
        "excess_inv": current_inv,
        "recommended_lots": {"Main": int(anchor_lot), "Refill/Trt": int(trt_lot), "Trial": int(trial_lot)},
        "cf_history": cashflow.tolist(),
        "inv_history": inventory.tolist(),
        "params": p_params
    }

# ------------------------------------------------
# 2. UI・ダッシュボード構築
# ------------------------------------------------
st.title("📊 NBDモデル 販売需要シミュレータ")

with st.sidebar:
    st.subheader("💾 データ保存・読込")
    st.file_uploader("設定ファイルを読み込む (.json)", type="json")

# --- STEP1: 市場環境と検索ボリューム ---
st.header("⚙️ STEP1: カテゴリと検索需要の分析")
c1, c2, c3 = st.columns(3)
with c1:
    cat_major = st.selectbox("大カテゴリ", ["ビューティー", "日用品・ヘルスケア"])
with c2:
    if cat_major == "ビューティー":
        cat_minor = st.selectbox("小カテゴリ", ["ヘアケア > シャンプー", "スキンケア > 化粧水", "UVケア > 日焼け止め"])
    else:
        cat_minor = st.selectbox("小カテゴリ", ["洗濯・仕上げ剤 > 液体洗剤", "バス用品 > ボディソープ"])
with c3:
    start_month = st.number_input("発売開始月 (1〜12月)", 1, 12, 3)

c4, c5, c6 = st.columns(3)
with c4:
    channel_type = st.selectbox("主戦場チャネル", ["ドラッグストア (DG)", "バラエティショップ (VS)"])
with c5:
    brand_name = st.text_input("ブランド名", placeholder="例：新・ボタニカルシャンプー")
with c6:
    search_volume = st.number_input("月間指名検索ボリューム (推定)", value=50000, step=1000, help="Google等でのブランド名・関連語の月間検索数")

# AIによるM値の推定
ai_m_value = calculate_m_from_search(search_volume, cat_minor)

if brand_name:
    st.success(f"🤖 **AI需要判定:** 指名検索数 {search_volume:,}/月 に基づき、このブランドの初期プレファレンス(M)を **{ai_m_value:.3f}** と推定しました。")
    st.info(f"※ターゲット母数1000万人に対し、年間約{(search_volume*12/10000000)*100:.1f}%のリーチ。カテゴリー転換率を考慮した実売期待値です。")

st.markdown("---")

# --- STEP2: シナリオ入力 ---
st.header("📝 STEP2: ブランド戦略の入力")
c_price, c_cost = st.columns(2)
price = c_price.number_input("メインSKU 販売単価 (円)", value=1500)
cost = c_cost.number_input("メインSKU 製造原価 (円)", value=400)

env_params = {"category": cat_minor, "start_month": start_month, "channel_type": channel_type, "price": price, "cost": cost}

colA, colB = st.columns(2)
def input_plan(label, key_suffix, def_lot, def_ad, def_dist, def_var, def_trt, def_trial, recommended_m):
    with st.container(border=True):
        st.subheader(label)
        # AIが判定したM値を初期値にする
        pref = st.slider(f"プレファレンス(M) - {key_suffix}", 0.01, 1.00, recommended_m, 0.01, key=f"m{key_suffix}")
        var_count = st.number_input("バリエーション数", 1, 5, def_var, key=f"var{key_suffix}")
        cc1, cc2 = st.columns(2)
        has_trt = cc1.checkbox("連動SKUあり", value=def_trt, key=f"trt{key_suffix}")
        has_trial = cc2.checkbox("トライアルSKUあり", value=def_trial, key=f"trial{key_suffix}")
        
        lot = st.number_input("メインSKU 初回ロット (個)", value=def_lot, key=f"lot{key_suffix}")
        ad = st.number_input("広告予算 (円)", value=def_ad, key=f"ad{key_suffix}")
        dist = st.slider("目標配荷率", 0.1, 1.0, def_dist, key=f"dist{key_suffix}")
        return {"pref_M": pref, "variations": var_count, "has_trt": has_trt, "has_trial": has_trial, "initial_lot": lot, "ad_budget": ad, "dist_rate": dist}

with colA:
    pA = input_plan("🔥 プラン A (強気)", "A", 200000, 30000000, 0.8, 3, True, False, ai_m_value)
with colB:
    pB = input_plan("🛡️ プラン B (保守的)", "B", 100000, 15000000, 0.4, 1, True, True, ai_m_value)

# --- STEP3: AI軍師補正 ---
if st.button("✨ AI補正プラン(C)を作成", type="primary"):
    new_lot = int(simulate_plan("temp", pA, env_params)['total_sales_vol'] * 0.4) # 適正ロットへ
    pC = {"pref_M": ai_m_value, "variations": 2, "has_trt": True, "has_trial": True, "initial_lot": new_lot, "ad_budget": 20000000, "dist_rate": 0.6}
    st.session_state['pC'] = pC

results = [simulate_plan("A", pA, env_params), simulate_plan("B", pB, env_params)]
if 'pC' in st.session_state:
    results.append(simulate_plan("C", st.session_state['pC'], env_params))

# --- STEP4: 結果表示 ---
st.markdown("---")
st.header("📈 シミュレーション結果")
kpi_cols = st.columns(len(results))
for idx, r in enumerate(results):
    with kpi_cols[idx]:
        with st.container(border=True):
            st.markdown(f"### プラン {r['name']}")
            st.metric("累計売上高", f"¥{int(r['total_revenue']):,}")
            st.metric("ボトムCF", f"¥{int(r['bottom_cf']):,}", delta="要資金調達" if r['bottom_cf'] < -50000000 else "安全")
            st.markdown(f"**欠品:** {'⚠️ あり' if r['stockout'] else '✅ なし'}")
            st.markdown(f"**棚落ち:** {'💥 危険' if r['shelf_drop'] else '✅ 安全'}")
