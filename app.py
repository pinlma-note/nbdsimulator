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
def get_seasonality(start_month, lifetime_months, category):
    """開始月に応じた季節変動指数を生成"""
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
    # 環境変数
    target_pop = 10000000 
    lifetime = 24 
    max_stores = 20000 
    
    cat = env_params['category']
    channel = env_params['channel_type']
    base_M = env_params['default_M']
    price = env_params['price']
    cost = env_params['cost']
    
    # ------------------------------------------------
    # SKUポートフォリオと分散ロジックの計算
    # ------------------------------------------------
    var_count = p_params['variations']
    # バリエーション増によるプレファレンスの増幅と分散 (べき乗則 n^0.6)
    M_total = base_M * (var_count ** 0.6)
    M_indiv = (M_total / var_count) * 0.9 if var_count > 1 else M_total # 複数展開時は10%の効率低下ペナルティ
    
    # 連動SKUのセット率設定
    trt_rate = 0.85 if p_params['has_trt'] else 0.0
    trial_rate = 0.50 if p_params['has_trial'] else 0.0
    
    # ------------------------------------------------
    # 需要予測ロジック (アンカーSKU基準)
    # ------------------------------------------------
    awareness = min(1.0, p_params['ad_budget'] / 50000000)
    accessible_pop = target_pop * awareness * p_params['dist_rate']
    anchor_total_demand = accessible_pop * M_total
    
    months = np.arange(1, lifetime + 1)
    diffusion = (months**1.5) * np.exp(-months / 3)
    diffusion = diffusion / diffusion.sum() 
    seasonality = get_seasonality(env_params['start_month'], lifetime, cat)
    
    monthly_demand = anchor_total_demand * diffusion * seasonality
    monthly_demand = monthly_demand * (anchor_total_demand / monthly_demand.sum()) 
    
    # ------------------------------------------------
    # SCM・キャッシュフロー計算 (ブランド合計)
    # ------------------------------------------------
    inventory = np.zeros(lifetime)
    cashflow = np.zeros(lifetime)
    sales_per_store = np.zeros(lifetime)
    
    # 生産ロットの内訳計算
    anchor_lot = p_params['initial_lot']
    trt_lot = anchor_lot * trt_rate
    trial_lot = anchor_lot * trial_rate
    total_initial_lot = anchor_lot + trt_lot + trial_lot
    
    current_inv = anchor_lot # 在庫はアンカーSKUを基準にリスク管理
    
    # 初期投資（全SKUの原価 + 広告費）
    # トライアル品は単価10%、原価20%と仮定
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
        
        # 売上回収（アンカー + 連動SKU + トライアルSKU）
        monthly_rev = (sales * price) + (sales * trt_rate * price) + (sales * trial_rate * (price * 0.1))
        current_cf += monthly_rev
        
        inventory[i] = current_inv
        cashflow[i] = current_cf
        sales_per_store[i] = sales / actual_stores

    # ------------------------------------------------
    # 棚落ちリスク判定（チャネル別ロジック）
    # ------------------------------------------------
    # 1店舗あたりの月間平均売上高（ブランド合計）
    avg_monthly_rev_per_store = (total_sales_vol * price * (1 + trt_rate + (trial_rate * 0.1)) / lifetime) / actual_stores
    
    if channel == "ドラッグストア (DG)":
        # 900mm棚1枚を想定: ブランド合計売上で判定 (閾値: 15,000円/月)
        shelf_drop_risk = avg_monthly_rev_per_store < 15000
    else:
        # バラエティショップ (VS): 単品の絶対売上・効率重視 (閾値: 40,000円/月)
        shelf_drop_risk = avg_monthly_rev_per_store < 40000
        
    # 個別SKUの危険判定 (カニバリゼーションによる1点あたりの弱体化)
    indiv_risk = (M_indiv < 0.05) and var_count > 1
    
    return {
        "name": plan_name,
        "total_sales_vol": total_sales_vol * (1 + trt_rate + trial_rate),
        "total_revenue": current_cf + initial_cost - p_params['ad_budget'], # 簡易売上累計
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
st.markdown("森岡毅氏の理論に基づく、ポートフォリオ（複数SKU）展開とチャネル別・棚落ちリスク判定モデルです。")

# --- サイドバー (保存・読込) ---
with st.sidebar:
    st.subheader("💾 データ保存・読込")
    uploaded_file = st.file_uploader("設定ファイルを読み込む (.json)", type="json")
    if uploaded_file:
        st.success("設定を読み込みました！")

# --- STEP1: 環境設定 ---
st.header("⚙️ STEP1: カテゴリと市場環境の設定")
c1, c2, c3 = st.columns(3)
with c1:
    cat_major = st.selectbox("大カテゴリ", ["ビューティー", "日用品・ヘルスケア"])
with c2:
    if cat_major == "ビューティー":
        cat_minor = st.selectbox("小カテゴリ", ["ヘアケア > シャンプー", "スキンケア > 化粧水", "UVケア > 日焼け止め"])
    else:
        cat_minor = st.selectbox("小カテゴリ", ["洗濯・仕上げ剤 > 液体洗剤", "バス用品 > ボディソープ"])
with c3:
    start_month = st.number_input("発売開始月 (1〜12月)", min_value=1, max_value=12, value=3)

c4, c5 = st.columns(2)
with c4:
    channel_type = st.selectbox("主戦場チャネル", ["ドラッグストア (DG)", "バラエティショップ (VS)"], help="DGは900mm棚のブランド合算効率、VSは絶対売上で棚落ちを判定します")
with c5:
    brand_name = st.text_input("ブランド名を入力してAI評価を実行", placeholder="例：新・無添加ボタニカル")

if brand_name:
    st.success(f"🤖 **AI判定:** 「{cat_minor}」カテゴリにおける「{brand_name}」の初期プレファレンス(M)を **0.12** と推定しました。")
    default_M = 0.12
else:
    default_M = 0.10

st.markdown("---")

# --- STEP2: シナリオ入力 ---
st.header("📝 STEP2: ブランド戦略の入力")
c_price, c_cost = st.columns(2)
price = c_price.number_input("メインSKU 販売単価 (円)", value=1500)
cost = c_cost.number_input("メインSKU 製造原価 (円)", value=400)

env_params = {"category": cat_minor, "start_month": start_month, "channel_type": channel_type, "default_M": default_M, "price": price, "cost": cost}

colA, colB = st.columns(2)
def input_plan(label, key_suffix, def_lot, def_ad, def_dist, def_var, def_trt, def_trial):
    with st.container(border=True):
        st.subheader(label)
        var_count = st.number_input("バリエーション数 (香り・色など)", 1, 5, def_var, key=f"var{key_suffix}", help="増やすと面は広がるが、1点あたりの回転は落ちる(カニバリ)")
        cc1, cc2 = st.columns(2)
        has_trt = cc1.checkbox("連動SKU (詰替/ライン品)", value=def_trt, key=f"trt{key_suffix}")
        has_trial = cc2.checkbox("トライアルSKU (お試し)", value=def_trial, key=f"trial{key_suffix}")
        
        lot = st.number_input("メインSKU 初回ロット (個)", value=def_lot, key=f"lot{key_suffix}")
        ad = st.number_input("広告予算 (円)", value=def_ad, key=f"ad{key_suffix}")
        dist = st.slider("目標配荷率 (全国店舗網)", 0.1, 1.0, def_dist, key=f"dist{key_suffix}")
        return {"variations": var_count, "has_trt": has_trt, "has_trial": has_trial, "initial_lot": lot, "ad_budget": ad, "dist_rate": dist}

with colA:
    pA = input_plan("🔥 プラン A (面展開・強気)", "A", 200000, 30000000, 0.8, 3, True, False)
with colB:
    pB = input_plan("🛡️ プラン B (単品集中・手堅い)", "B", 100000, 15000000, 0.4, 1, True, True)

st.markdown("---")

# --- STEP3: AI軍師による最適化 ---
st.header("🤖 STEP3: AI軍師の補正 (プランC生成)")
if st.button("✨ リスクを抑えた『AI補正プラン(C)』を作成する", type="primary"):
    # AIロジック: 多品種展開のリスクを抑え、トライアル品で底上げ
    new_lot = int(pA['initial_lot'] * 1.3) 
    pC = {"variations": 2, "has_trt": True, "has_trial": True, "initial_lot": new_lot, "ad_budget": 20000000, "dist_rate": 0.6}
    st.session_state['pC'] = pC
    st.info("✅ **AI軍師の診断:** プランAの『3種展開』はカニバリゼーションにより1点あたりの棚落ちリスクが高すぎます。バリエーションを2種に絞り、浮いた原価をトライアル品の投入（新規M獲得）に回すプランCを策定しました。")

results = [simulate_plan("A", pA, env_params), simulate_plan("B", pB, env_params)]
if 'pC' in st.session_state:
    results.append(simulate_plan("C", st.session_state['pC'], env_params))

st.markdown("---")

# --- STEP4: 結果表示 ---
st.header("📈 シミュレーション結果 (24ヶ月)")

kpi_cols = st.columns(len(results))
for idx, r in enumerate(results):
    with kpi_cols[idx]:
        with st.container(border=True):
            st.markdown(f"### プラン {r['name']}")
            st.metric("累計売上高 (ブランド計)", f"¥{int(r['total_revenue']):,}")
            st.metric("ボトムCF (最大必要資金)", f"¥{int(r['bottom_cf']):,}", 
                      delta="資金ショート注意" if r['bottom_cf'] < -50000000 else "安全圏", delta_color="inverse")
            st.markdown(f"**欠品リスク:** {'⚠️ 発生' if r['stockout'] else '✅ なし'}")
            st.markdown(f"**棚落ちリスク ({channel_type}):** {'💥 危険 (回転率低)' if r['shelf_drop'] else '✅ 安全'}")
            if r['indiv_risk']:
                st.markdown("**⚠️ 一部バリエーションの撤去リスクあり**")
            
            with st.expander("📦 推奨生産ロット内訳"):
                st.write(f"- メインSKU: {r['recommended_lots']['Main']:,} 個")
                if pA['has_trt']: st.write(f"- 連動/詰替: {r['recommended_lots']['Refill/Trt']:,} 個")
                if pA['has_trial']: st.write(f"- トライアル: {r['recommended_lots']['Trial']:,} 個")

t1, t2 = st.tabs(["📉 キャッシュフロー推移", "📦 メインSKU 在庫推移"])
with t1:
    fig_cf = go.Figure()
    for r in results: fig_cf.add_trace(go.Scatter(y=r['cf_history'], name=f"Plan {r['name']}"))
    fig_cf.update_layout(xaxis_title="経過月", yaxis_title="CF (円)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cf, use_container_width=True)
with t2:
    fig_inv = go.Figure()
    for r in results: fig_inv.add_trace(go.Scatter(y=r['inv_history'], name=f"Plan {r['name']}"))
    fig_inv.update_layout(xaxis_title="経過月", yaxis_title="在庫数 (個)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_inv, use_container_width=True)

# エクスポート機能
st.sidebar.markdown("---")
st.sidebar.download_button("💾 現在の設定を保存", data=json.dumps({"A": pA, "B": pB, "C": st.session_state.get('pC', {})}), file_name="nbd_settings.json")

df_report = pd.DataFrame([{
    "プラン": r['name'],
    "累計売上高": int(r['total_revenue']),
    "ボトムCF": int(r['bottom_cf']),
    "欠品発生": r['stockout'],
    "棚落ち危険": r['shelf_drop'],
    "カニバリ危険": r['indiv_risk']
} for r in results])
st.sidebar.download_button("📑 比較レポート(CSV)を出力", data=df_report.to_csv(index=False).encode('utf-8-sig'), file_name="nbd_report.csv")
