import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gamma
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide", page_title="NBDマーケティング・シミュレータ", page_icon="📊")

# ------------------------------------------------
# 1. マスタデータ定義 (ターゲット母数とNBDのK値を追加)
# ------------------------------------------------
CATEGORY_MASTERS = {
    "ヘアケア > シャンプー": {"freq": 4.0, "k": 0.8, "pop": 50000000},
    "スキンケア > 化粧水": {"freq": 3.0, "k": 0.4, "pop": 20000000},
    "UVケア > 日焼け止め": {"freq": 2.0, "k": 0.6, "pop": 30000000},
    "洗濯・仕上げ剤 > 液体洗剤": {"freq": 10.0, "k": 0.3, "pop": 50000000},
    "バス用品 > ボディソープ": {"freq": 5.0, "k": 0.5, "pop": 50000000}
}

# ------------------------------------------------
# 2. 数理モデル & 計算ロジック
# ------------------------------------------------
def get_seasonality(start_month, lifetime_months, category):
    # 季節指数（平均が1.0になるように設定）
    if any(k in category for k in ["UV", "シャンプー", "ボディソープ"]):
        base_season = np.array([0.7, 0.7, 0.9, 1.1, 1.3, 1.6, 1.6, 1.4, 0.9, 0.7, 0.6, 0.5])
    elif any(k in category for k in ["保湿", "スキンケア"]):
        base_season = np.array([1.4, 1.2, 1.0, 0.8, 0.7, 0.7, 0.7, 0.8, 1.0, 1.2, 1.4, 1.5])
    else:
        base_season = np.ones(12)
    
    # 発売月からの配列を作成
    seasons = [base_season[(start_month - 1 + i) % 12] for i in range(lifetime_months)]
    return np.array(seasons)

def simulate_plan(plan_name, p_params, env_params):
    cat_info = CATEGORY_MASTERS[env_params['category']]
    target_pop = cat_info['pop']
    k_val = cat_info['k']
    lifetime = 24 
    max_stores = 20000 
    
    channel = env_params['channel_type']
    base_M = p_params['pref_M']
    price = env_params['price']
    cost = env_params['cost']
    
    # UIからの入力値(万単位)を実数に変換
    ad_budget_actual = p_params['ad_budget_man'] * 10000
    initial_lot_actual = p_params['initial_lot_man'] * 10000
    
    # バリエーションによるプレファレンス分散とカニバリゼーション
    var_count = p_params['variations']
    M_total = base_M * (var_count ** 0.6)
    M_indiv = (M_total / var_count) * 0.9 if var_count > 1 else M_total 
    
    # NBDモデルによる浸透率とリピート率の算出
    # ペネトレーション(1回以上買う確率) = 1 - (1 + M/K)^(-K)
    penetration = 1 - (1 + M_indiv / k_val)**(-k_val)
    repeat_rate = (M_indiv / penetration) if penetration > 0 else 0
    
    # セット率
    trt_rate = 0.85 if p_params['has_trt'] else 0.0
    trial_rate = 0.50 if p_params['has_trial'] else 0.0
    
    # 広告費と認知率のS字カーブ（ロジスティック的な漸近曲線）
    # 広告費ゼロなら認知ほぼゼロ。上限は80%認知と仮定。
    awareness = 0.8 * (1 - np.exp(-0.0002 * p_params['ad_budget_man']))
    
    accessible_pop = target_pop * awareness * p_params['dist_rate']
    anchor_total_demand = accessible_pop * M_total
    
    # 普及曲線
    months = np.arange(1, lifetime + 1)
    diffusion = (months**1.5) * np.exp(-months / 3)
    diffusion = diffusion / diffusion.sum() 
    
    # 季節変動係数を掛ける（再正規化しないことで、発売月の有利不利を出す）
    seasonality = get_seasonality(env_params['start_month'], lifetime, env_params['category'])
    monthly_demand = anchor_total_demand * diffusion * seasonality
    
    # 在庫・CF計算
    inventory = np.zeros(lifetime)
    cashflow = np.zeros(lifetime)
    sales_per_store = np.zeros(lifetime)
    
    trt_lot = initial_lot_actual * trt_rate
    trial_lot = initial_lot_actual * trial_rate
    
    initial_investment = (initial_lot_actual * cost) + (trt_lot * cost) + (trial_lot * cost * 0.2) + ad_budget_actual
    current_inv = initial_lot_actual 
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

    # 厳格な棚落ち判定（発売3ヶ月〜6ヶ月目の初速ROSで判定）
    # この期間の月間平均売上が基準を下回れば容赦なくカット
    ros_m3_6_vol = np.mean(sales_per_store[2:6]) if len(sales_per_store) >= 6 else np.mean(sales_per_store)
    ros_m3_6_val = ros_m3_6_vol * price * (1 + trt_rate + (trial_rate * 0.1))
    
    if channel == "ドラッグストア (DG)":
        shelf_drop_risk = ros_m3_6_val < 15000
    else:
        shelf_drop_risk = ros_m3_6_val < 40000
        
    indiv_risk = (M_indiv < 0.05) and var_count > 1
    
    return {
        "name": plan_name,
        "total_revenue": current_cf + initial_investment - ad_budget_actual,
        "final_cf": current_cf,
        "bottom_cf": np.min(cashflow),
        "stockout": stockout_month > 0,
        "stockout_month": stockout_month,
        "shelf_drop": shelf_drop_risk,
        "indiv_risk": indiv_risk,
        "excess_inv": current_inv,
        "awareness": awareness,
        "penetration": penetration,
        "repeat_rate": repeat_rate,
        "recommended_lots": {"Main": int(initial_lot_actual), "Refill": int(trt_lot), "Trial": int(trial_lot)},
        "cf_history": cashflow.tolist(),
        "inv_history": inventory.tolist(),
        "params": p_params
    }

# ------------------------------------------------
# 3. UI構築
# ------------------------------------------------
st.title("📊 NBDフル機能シミュレータ (数理モデル改修版)")

with st.sidebar:
    st.header("💾 データ管理")
    st.file_uploader("JSON設定を読み込む", type="json")
    st.divider()
    st.info("※次フェーズで「過去実績CSV」のアップロード＆答え合わせ機能をここに実装します。")

# --- STEP1: 市場環境 ---
st.header("⚙️ STEP1: カテゴリ・市場・M値設定")
c1, c2, c3 = st.columns(3)
with c1:
    cat_minor = st.selectbox("小カテゴリ", list(CATEGORY_MASTERS.keys()), 
                             help="カテゴリにより、市場母数(パイ)と消費者の購入の偏り(NBDのK値)が数理的に切り替わります。")
    cat_info = CATEGORY_MASTERS[cat_minor]
with c2:
    start_month = st.number_input("発売開始月", 1, 12, 3, 
                                  help="需要ピークと発売月がズレると、生涯売上（面積）そのものが減少する厳密な計算になっています。")
with c3:
    channel_type = st.selectbox("主戦場チャネル", ["ドラッグストア (DG)", "バラエティショップ (VS)"])

st.markdown("#### プレファレンス(M)の決定方法")
target_share = st.slider("目標ユニットシェア (%)", 0.1, 10.0, 2.0, 0.1, 
                         help=f"この市場の年間平均購入回数に基づく逆算です。")
calculated_m = cat_info['freq'] * (target_share / 100)
st.info(f"💡 目標シェア {target_share}% を達成するために必要な **プレファレンス(M): {calculated_m:.3f}**")

st.divider()

# --- STEP2: 戦略とコスト ---
st.header("📝 STEP2: ブランド戦略とコスト設定")
c_price, c_cost, c_brand = st.columns(3)
price = c_price.number_input("単価(円)", value=1500, step=100)
cost = c_cost.number_input("原価(円)", value=400, step=50)
brand_name = c_brand.text_input("ブランド名", "New Brand", help="今後のアップデートでブランド力の係数判定に用います")

env_params = {"category": cat_minor, "start_month": start_month, "channel_type": channel_type, "price": price, "cost": cost}

colA, colB = st.columns(2)
def input_plan(label, key, def_lot_man, def_ad_man, def_dist, m_val):
    with st.container(border=True):
        st.subheader(label)
        
        # --- 基本項目（常に見せる） ---
        p_m = st.number_input(f"M値 - {key}", value=m_val, key=f"m{key}", format="%.3f")
        lot_man = st.number_input("初回ロット (万個)", value=def_lot_man, step=1.0, key=f"l{key}")
        ad_man = st.number_input("広告予算 (万円)", value=def_ad_man, step=500.0, key=f"a{key}", 
                                 help="S字カーブを採用。予算が少なすぎると認知は広がりません。")
        dist = st.slider("目標配荷率", 0.1, 1.0, def_dist, key=f"d{key}")
        
        # --- 詳細項目（アコーディオンで隠す） ---
        with st.expander("🛠️ 詳細ポートフォリオ設定 (SKU・バリエーション)"):
            vars_cnt = st.number_input("バリエーション数", 1, 5, 1, key=f"v{key}", 
                                       help="香りや色の種類。増やすとブランド全体のパイは増えますが、1点あたりの売上が下がり棚落ちリスクが高まります（カニバリ）。")
            t1, t2 = st.columns(2)
            has_trt = t1.checkbox("連動SKUあり", value=True, key=f"t{key}", help="詰替やトリートメントなど、本品に連動して売れるSKUです。")
            has_trial = t2.checkbox("トライアルSKUあり", value=False, key=f"tr{key}", help="サシェなど。新規の獲得(Mの向上)に寄与します。")
            
        return {"pref_M": p_m, "variations": vars_cnt, "has_trt": has_trt, "has_trial": has_trial, 
                "initial_lot_man": lot_man, "ad_budget_man": ad_man, "dist_rate": dist}

with colA:
    pA = input_plan("プラン A (強気)", "A", 20.0, 3000.0, 0.8, calculated_m)
with colB:
    pB = input_plan("プラン B (保守的)", "B", 8.0, 1000.0, 0.4, calculated_m)

# 実行
current_plans = [("A", pA), ("B", pB)]
results = [simulate_plan(name, params, env_params) for name, params in current_plans]

st.divider()

# --- STEP3: 結果・エクスポート ---
st.header("📈 シミュレーション結果")

# NBDモデルの真骨頂である内部指標を表示
k_cols = st.columns(len(results))
for idx, r in enumerate(results):
    with k_cols[idx]:
        with st.container(border=True):
            st.markdown(f"### プラン {r['name']}")
            st.metric("累計売上高", f"¥{int(r['total_revenue']):,}")
            st.metric("ボトムCF", f"¥{int(r['bottom_cf']):,}", delta="資金ショート危険" if r['bottom_cf'] < -30000000 else "安全圏", delta_color="inverse")
            
            st.markdown("#### 🚨 リスク判定")
            st.write(f"- **欠品:** {'⚠️ あり' if r['stockout'] else '✅ なし'}")
            st.write(f"- **棚落ち (3〜6ヶ月初速):** {'💥 危険' if r['shelf_drop'] else '✅ 安全'}")
            st.write(f"- **カニバリ (バリエーション過多):** {'⚠️ 危険' if r['indiv_risk'] else '✅ 安全'}")
            
            st.markdown("#### 🧠 NBD内部指標 (推計)")
            st.caption(f"・想定認知率: {r['awareness']*100:.1f}%\n"
                       f"・浸透率(1回以上買う人): {r['penetration']*100:.2f}%\n"
                       f"・平均リピート回数: {r['repeat_rate']:.2f}回")
            
            with st.expander("📦 推奨生産ロット詳細"):
                st.write(f"メイン: {r['recommended_lots']['Main']:,} 個")
                st.write(f"連動: {r['recommended_lots']['Refill']:,} 個")
                st.write(f"お試し: {r['recommended_lots']['Trial']:,} 個")

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
df_res = pd.DataFrame([{
    "プラン": r['name'],
    "累計売上": f"¥{int(r['total_revenue']):,}",
    "ボトムCF": f"¥{int(r['bottom_cf']):,}",
    "想定認知率": f"{r['awareness']*100:.1f}%",
    "浸透率": f"{r['penetration']*100:.2f}%",
    "リピート回数": f"{r['repeat_rate']:.2f}回",
    "欠品発生": "⚠️" if r['stockout'] else "✅",
    "棚落ちリスク": "❌" if r['shelf_drop'] else "✅",
} for r in results])

st.sidebar.download_button("📑 レポートをCSV出力", data=df_res.to_csv(index=False).encode('utf-8-sig'), file_name="nbd_report.csv")
