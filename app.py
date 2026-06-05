import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# page name in da website title
st.set_page_config(
    page_title="PhonePredict",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# connect to css
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# model re-training
MODEL_PATH = "rf_model.pkl"
DATA_PATH  = "train.csv"

@st.cache_resource(show_spinner="Training model on dataset…")
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model, medians = pickle.load(f)
        return model, medians

    if not os.path.exists(DATA_PATH):
        st.error("train.csv not found. Please put it in the same folder as app.py.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    target = "price_range"
    X = df.drop(columns=[target])
    y = df[target]
    medians = X.median().to_dict()

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((rf, medians), f)
    return rf, medians

model, MEDIANS = load_or_train_model()

if "history" not in st.session_state:
    st.session_state.history = []

FEATURE_ORDER = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi",
]

PRICE_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
PRICE_COLORS = {
    0: ("#e6f0fb", "#1a5fa8", "#1a5fa8"),
    1: ("#e9f5e1", "#2e7d32", "#2e7d32"),
    2: ("#fff4e0", "#b35c00", "#b35c00"),
    3: ("#fdecea", "#b71c1c", "#b71c1c"),
}
BAR_COLORS = ["#378ADD", "#63991F", "#EF9F27", "#E24B4A"]

# header (the phonerange box thingy)
st.markdown("""
<div class="header">
    <h1>PhonePredict</h1>
    <p>Enter your smartphone's technical specs below and let us predict its market price range for you :D</p>
</div>
""", unsafe_allow_html=True)

# functions helper (buat input values)
def yesno(label, key):
    val = st.radio(label, ["Yes", "No"], horizontal=True, key=key, index=1)
    return 1 if val == "Yes" else 0

def num_input(label, key, min_v, max_v, default=None, step=1, fmt="%g", help_txt=None):
    val_default = st.session_state.get(key, 0.0)
    return st.number_input(label, min_value=0.0, max_value=99999.0,
                           value=float(val_default), step=float(step), format=fmt, help=help_txt, key=key)

# clear input
st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
if st.button("🔄 Clear All Input Fields", use_container_width=True):
    categorical_features = {"blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"}
    for k in FEATURE_ORDER:
        if k in categorical_features:
            st.session_state[k] = "No"
        else:
            st.session_state[k] = 0.0
    st.rerun()

# shortcuts
st.markdown('<div class="section-label">⚡ Shortcuts</div>', unsafe_allow_html=True)
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("🧱 Budget-Friendly Specs", use_container_width=True):
        st.session_state.battery_power = 700.0
        st.session_state.ram = 512.0
        st.session_state.clock_speed = 0.8
        st.session_state.n_cores = 2.0
        st.session_state.int_memory = 4.0
        st.session_state.pc = 5.0
        st.session_state.fc = 2.0
        st.session_state.px_height = 200.0
        st.session_state.px_width = 600.0
        st.session_state.sc_h = 8.0
        st.session_state.sc_w = 3.0
        st.session_state.mobile_wt = 185.0
        st.session_state.m_dep = 0.8
        st.session_state.talk_time = 5.0
        st.session_state.blue = "No"
        st.session_state.dual_sim = "Yes"
        st.session_state.four_g = "No"
        st.session_state.three_g = "Yes"
        st.session_state.wifi = "No"
        st.session_state.touch_screen = "No"
        st.rerun()

with p_col2:
    if st.button("⚡ Mid-Price Specs", use_container_width=True):
        st.session_state.battery_power = 1200.0
        st.session_state.ram = 2100.0
        st.session_state.clock_speed = 1.5
        st.session_state.n_cores = 4.0
        st.session_state.int_memory = 32.0
        st.session_state.pc = 11.0
        st.session_state.fc = 5.0
        st.session_state.px_height = 560.0
        st.session_state.px_width = 1250.0
        st.session_state.sc_h = 12.0
        st.session_state.sc_w = 5.0
        st.session_state.mobile_wt = 140.0
        st.session_state.m_dep = 0.5
        st.session_state.talk_time = 11.0
        st.session_state.blue = "Yes"
        st.session_state.dual_sim = "Yes"
        st.session_state.four_g = "Yes"
        st.session_state.three_g = "Yes"
        st.session_state.wifi = "Yes"
        st.session_state.touch_screen = "Yes"
        st.rerun()

with p_col3:
    if st.button("🚀 High-End Specs", use_container_width=True):
        st.session_state.battery_power = 1600.0
        st.session_state.ram = 2800.0
        st.session_state.clock_speed = 2.2
        st.session_state.n_cores = 6.0
        st.session_state.int_memory = 32.0
        st.session_state.pc = 15.0
        st.session_state.fc = 8.0
        st.session_state.px_height = 900.0
        st.session_state.px_width = 1500.0
        st.session_state.sc_h = 14.0
        st.session_state.sc_w = 9.0
        st.session_state.mobile_wt = 120.0
        st.session_state.m_dep = 0.3
        st.session_state.talk_time = 15.0
        st.session_state.blue = "Yes"
        st.session_state.dual_sim = "Yes"
        st.session_state.four_g = "Yes"
        st.session_state.three_g = "Yes"
        st.session_state.wifi = "Yes"
        st.session_state.touch_screen = "Yes"
        st.rerun()

with p_col4:
    if st.button("👑 Premium Specs", use_container_width=True):
        st.session_state.battery_power = 1950.0
        st.session_state.ram = 3850.0
        st.session_state.clock_speed = 2.9
        st.session_state.n_cores = 8.0
        st.session_state.int_memory = 64.0
        st.session_state.pc = 20.0
        st.session_state.fc = 16.0
        st.session_state.px_height = 1400.0
        st.session_state.px_width = 1900.0
        st.session_state.sc_h = 18.0
        st.session_state.sc_w = 14.0
        st.session_state.mobile_wt = 95.0
        st.session_state.m_dep = 0.1
        st.session_state.talk_time = 20.0
        st.session_state.blue = "Yes"
        st.session_state.dual_sim = "Yes"
        st.session_state.four_g = "Yes"
        st.session_state.three_g = "Yes"
        st.session_state.wifi = "Yes"
        st.session_state.touch_screen = "Yes"
        st.rerun()

st.markdown('<div class="section-label">01. Core performance</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    battery_power = num_input("🔋 Battery capacity (mAh)", "battery_power", 0, 1998, step=50)
    st.caption("Range: 501 – 1998 mAh")
    if battery_power != 0 and not (501 <= battery_power <= 1998):
        st.warning("Must be between 501 – 1998 mAh")

with c2:
    ram = num_input("🧠 RAM (MB)", "ram", 0, 3998, step=64)
    st.caption("Range: 256 – 3998 MB")
    if ram != 0 and not (256 <= ram <= 3998):
        st.warning("Must be between 256 – 3998 MB")

with c3:
    clock_speed = num_input("🕒 Clock speed (GHz)", "clock_speed", 0, 3.0, step=0.1, fmt="%.1f")
    st.caption("Range: 0.5 – 3.0 GHz")
    if clock_speed != 0 and not (0.5 <= clock_speed <= 3.0):
        st.warning("Must be between 0.5 – 3.0 GHz")

c4, c5 = st.columns(2)

with c4:
    n_cores = num_input("⚡ Number of CPU cores", "n_cores", 0, 8)
    st.caption("Range: 1 – 8")
    if n_cores != 0 and not (1 <= n_cores <= 8):
        st.warning("Must be between 1 – 8 cores")

with c5:
    int_memory = num_input("💾 Internal storage (GB)", "int_memory", 0, 64)
    st.caption("Range: 2 – 64 GB")
    if int_memory != 0 and not (2 <= int_memory <= 64):
        st.warning("Must be between 2 – 64 GB")

st.markdown('<div class="section-label">02. Camera</div>', unsafe_allow_html=True)
c6, c7 = st.columns(2)

with c6:
    pc = num_input("📸 Primary/rear camera (MP)", "pc", 0, 20)
    st.caption("Range: 0 – 20 MP")
    if pc != 0 and not (0 <= pc <= 20):
        st.warning("Must be between 0 – 20 MP")

with c7:
    fc = num_input("📲 Front camera (MP)", "fc", 0, 19)
    st.caption("Range: 0 – 19 MP")
    if fc != 0 and not (0 <= fc <= 19):
        st.warning("Must be between 0 – 19 MP")

st.markdown('<div class="section-label">03. Display</div>', unsafe_allow_html=True)
c8, c9 = st.columns(2)

with c8:
    px_height = num_input("🖥️ Pixel resolution height", "px_height", 0, 1960, step=10)
    st.caption("Range: 0 – 1960 px")
    if px_height != 0 and not (0 <= px_height <= 1960):
        st.warning("Must be between 0 – 1960 px")

with c9:
    px_width  = num_input("🖥️ Pixel resolution width",  "px_width",  0, 1998, step=10)
    st.caption("Range: 500 – 1998 px")
    if px_width != 0 and not (500 <= px_width <= 1998):
        st.warning("Must be between 500 – 1998 px")

c10, c11 = st.columns(2)

with c10:
    sc_h = num_input("📱 Screen height (cm)", "sc_h", 0, 19)
    st.caption("Range: 5 – 19 cm")
    if sc_h != 0 and not (5 <= sc_h <= 19):
        st.warning("Must be between 5 – 19 cm")

with c11:
    sc_w = num_input("📱 Screen width (cm)",  "sc_w", 0, 18)
    st.caption("Range: 0 – 18 cm")
    if sc_w != 0 and not (0 <= sc_w <= 18):
        st.warning("Must be between 0 – 18 cm")

st.markdown('<div class="section-label">04. Physical</div>', unsafe_allow_html=True)
c12, c13 = st.columns(2)

with c12:
    mobile_wt = num_input("⚖️ Weight (g)", "mobile_wt", 0, 200)
    st.caption("Range: 80 – 200 g")
    if mobile_wt != 0 and not (80 <= mobile_wt <= 200):
        st.warning("Must be between 80 – 200 g")

with c13:
    m_dep = num_input("📏 Depth/thickness (cm)", "m_dep", 0, 1.0, step=0.1, fmt="%.1f")
    st.caption("Range: 0.1 – 1.0 cm")
    if m_dep != 0 and not (0.1 <= m_dep <= 1.0):
        st.warning("Must be between 0.1 – 1.0 cm")

talk_time = num_input("📞 Talk time (hours)", "talk_time", 0, 20)
st.caption("Range: 2 – 20 hours")
if talk_time != 0 and not (2 <= talk_time <= 20):
    st.warning("Must be between 2 – 20 hours")

st.markdown('<div class="section-label">05 · Connectivity</div>', unsafe_allow_html=True)
c14, c15, c16 = st.columns(3)

with c14:
    blue       = yesno("🔵 Bluetooth", "blue")
    dual_sim   = yesno("🎴 Dual SIM", "dual_sim")

with c15:
    four_g     = yesno("📶 4G / LTE", "four_g")
    three_g    = yesno("🌐 3G", "three_g")

with c16:
    wifi         = yesno("🛜 Wi-Fi", "wifi")
    touch_screen = yesno("🤳 Touch screen", "touch_screen")

# predicting ui n logic
if st.button("→  Predict price range", use_container_width=True):
    raw = {
        "battery_power": battery_power,
        "blue":          blue,
        "clock_speed":   clock_speed,
        "dual_sim":      dual_sim,
        "fc":            fc,
        "four_g":        four_g,
        "int_memory":    int_memory,
        "m_dep":         m_dep,
        "mobile_wt":     mobile_wt,
        "n_cores":       n_cores,
        "pc":            pc,
        "px_height":     px_height,
        "px_width":      px_width,
        "ram":           ram,
        "sc_h":          sc_h,
        "sc_w":          sc_w,
        "talk_time":     talk_time,
        "three_g":       three_g,
        "touch_screen":  touch_screen,
        "wifi":          wifi,
    }

    VALID_RANGES = {
        "battery_power": (501, 1998),
        "ram":           (256, 3998),
        "clock_speed":   (0.5, 3.0),
        "n_cores":       (1, 8),
        "int_memory":    (2, 64),
        "pc":            (0, 20),
        "fc":            (0, 19),
        "px_height":     (0, 1960),
        "px_width":      (0, 1998),
        "sc_h":          (5, 19),
        "sc_w":          (0, 18),
        "mobile_wt":     (80, 200),
        "m_dep":         (0.1, 1.0),
        "talk_time":     (2, 20),
    }

    errors = []
    for field, (lo, hi) in VALID_RANGES.items():
        val = raw[field]
        if val != 0 and not (lo <= val <= hi):
            errors.append(f"**{field.replace('_', ' ')}**: {val} (valid range: {lo}–{hi})")

    if errors:
        st.error("Some inputs are outside the valid dataset range!!!\n\n" + "\n\n".join(errors))
        st.stop()
        
    # auto put median
    filled = {k: (v if v != 0 or k in ("blue","dual_sim","four_g","three_g","wifi","touch_screen")
                    else MEDIANS.get(k, v))
              for k, v in raw.items()}
    input_vec = [filled[f] for f in FEATURE_ORDER]

    pred   = model.predict([input_vec])[0]
    probas = model.predict_proba([input_vec])[0]
    label  = PRICE_LABELS[pred]
    bg, text_color, bar_color = PRICE_COLORS[pred]

    new_entry = {
        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
        "Predicted Tier": label,
        "RAM (MB)": int(filled['ram']),
        "Battery (mAh)": int(filled['battery_power']),
        "Internal Storage (GB)": int(filled['int_memory']),
        "Cores": int(filled['n_cores']),
        "Clock": float(filled['clock_speed']),
        "Weight": int(filled['mobile_wt']),
        "Thick": float(filled['m_dep']),
        "PCam": int(filled['pc']),
        "FCam": int(filled['fc']),
        "Height": int(filled['px_height']),
        "Width": int(filled['px_width']),
        "ScH": int(filled['sc_h']),
        "ScW": int(filled['sc_w']),
        "Talk": int(filled['talk_time']),
        "blue": int(filled['blue']),
        "dual_sim": int(filled['dual_sim']),
        "four_g": int(filled['four_g']),
        "three_g": int(filled['three_g']),
        "touch_screen": int(filled['touch_screen']),
        "wifi": int(filled['wifi'])
    }
    st.session_state.history.insert(0, new_entry)

    # results
    TIER_INSIGHTS = {
        0: "🧱 <b>Entry-Level Range</b>: Optimized for budget efficiency. Best suited for essential daily needs, messaging, and long-lasting battery standby.",
        1: "⚡ <b>Mid-Range Standard</b>: The sweet spot. Offers reliable multitasking performance, casual gaming, and great value for money.",
        2: "🔥 <b>High-Tier Performer</b>: Stepping into the premium segment. Highly capable for heavy multitasking, intensive gaming graphics, and fast media rendering.",
        3: "👑 <b>Ultimate Beast</b>: No compromises! Top-tier hardware profile specially engineered for extreme computing performance and cutting-edge features."
    }
    
    insight_text = TIER_INSIGHTS[pred]

    st.markdown(f"""
    <div class="result-card" style="background:{bg}; border: 1.5px solid {text_color}22; width:100%; box-sizing:border-box; text-align:center; margin-bottom: 1rem;">
        <div class="result-label" style="color:{text_color}; margin-bottom: 0.8rem;">PREDICTED PRICE RANGE</div>
        <div class="result-badge" style="color:{text_color}; margin: 0.5rem 0 0.8rem;">{label}</div>
        <div class="result-sub" style="color:{text_color};">Price tier {pred + 1} of 4</div>
    </div>
    
    <div style="background: #1e293b; border-left: 4px solid {text_color}; padding: 1.1rem 1.3rem; border-radius: 8px; margin-bottom: 1.5rem; color: #f1f5f9; font-size: 0.92rem; line-height: 1.6;">
        {insight_text}
    </div>

    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: #0f172a; border: 1px solid #334155; padding: 1.2rem; border-radius: 12px; margin-bottom: 1.8rem; text-align: center;">
        <div>
            <span style="color: #94a3b8; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">RAM</span>
            <strong style="color: #ffffff; font-size: 1.1rem;">{int(filled['ram'])} <span style="font-size: 0.8rem; color: #64748b;">MB</span></strong>
        </div>
        <div>
            <span style="color: #94a3b8; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Battery</span>
            <strong style="color: #ffffff; font-size: 1.1rem;">{int(filled['battery_power'])} <span style="font-size: 0.8rem; color: #64748b;">mAh</span></strong>
        </div>
        <div>
            <span style="color: #94a3b8; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Storage</span>
            <strong style="color: #ffffff; font-size: 1.1rem;">{int(filled['int_memory'])} <span style="font-size: 0.8rem; color: #64748b;">GB</span></strong>
        </div>
        <div>
            <span style="color: #94a3b8; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">CPU Cores</span>
            <strong style="color: #ffffff; font-size: 1.1rem;">{int(filled['n_cores'])} <span style="font-size: 0.8rem; color: #64748b;">Cores</span></strong>
        </div>
    </div>

    <div style="margin-top: 1rem; padding-left: 2px; margin-bottom: 0.6rem;">
        <span style="font-family: 'Poppins', sans-serif; font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;">Prediction Confidence Breakdown</span>
    </div>
    
    <div style="margin-top: 0.5rem;">
    """, unsafe_allow_html=True)

    # the bars thingy
    for i, (prob, clr) in enumerate(zip(probas, BAR_COLORS)):
        pct = round(prob * 100, 1)
        active_weight = "font-weight:600;" if i == pred else ""
        st.markdown(f"""
        <div class="bar-row" style="{active_weight}">
            <span style="min-width:72px; color:{clr if i==pred else '#888'};">{PRICE_LABELS[i]}</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%; background:{clr};"></div>
            </div>
            <span class="bar-pct">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # notes buat median disclaimer
    zeros_filled = [k for k, v in raw.items()
                    if v == 0 and k not in ("blue","dual_sim","four_g","three_g","wifi","touch_screen")]
    
    if zeros_filled:
        DISPLAY_NAMES = {
            "battery_power": "Battery capacity",
            "clock_speed": "Clock speed",
            "fc": "Front camera",
            "int_memory": "Internal storage",
            "m_dep": "Depth/thickness",
            "mobile_wt": "Weight",
            "n_cores": "Number of CPU cores",
            "pc": "Primary camera",
            "px_height": "Pixel height resolution",
            "px_width": "Pixel width resolution",
            "ram": "RAM",
            "sc_h": "Screen height",
            "sc_w": "Screen width",
            "talk_time": "Talk time"
        }
        
        clean_names = [DISPLAY_NAMES.get(k, k) for k in zeros_filled]
        
        st.markdown(
            f'<div class="median-note">Fields left at 0 were auto-filled with their medians: '
            f'{", ".join(clean_names)}</div>',
            unsafe_allow_html=True
        )

# history section
if st.session_state.history:
    st.markdown('<div class="section-label">📋 Prediction History</div>', unsafe_allow_html=True)
    
    HIST_COLORS = {
        "Low": ("#1a5fa8", "#162235"),
        "Medium": ("#2e7d32", "#18251a"),
        "High": ("#b35c00", "#2b2011"),
        "Very High": ("#b71c1c", "#2d1616")
    }
    
    for item in st.session_state.history:
        tier = item["Predicted Tier"]
        border_clr, bg_clr = HIST_COLORS.get(tier, ("#374151", "#111827"))
        
        active_features = []
        if item.get("blue") == 1: active_features.append("🔵 Bluetooth")
        if item.get("dual_sim") == 1: active_features.append("🎴 Dual SIM")
        if item.get("four_g") == 1: active_features.append("📶 4G LTE")
        if item.get("three_g") == 1: active_features.append("🌐 3G")
        if item.get("touch_screen") == 1: active_features.append("📱 Touchscreen")
        if item.get("wifi") == 1: active_features.append("✨ Wi-Fi")
        
        features_txt = " &nbsp;•&nbsp; ".join(active_features) if active_features else "None"
        badge_style = "display:inline-block; margin:4px; padding:5px 10px; font-size:0.75rem; color:#cbd5e1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); border-radius:6px; text-align:center; white-space:nowrap;"

        raw_html = f"""
        <div style="padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2); border-left: 5px solid {border_clr}; background: {bg_clr}; box-sizing: border-box; font-family: 'Inter', sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 0.4rem;">
                <span style="font-size: 0.78rem; color: #94a3b8;">⏱️ {item["Time"]}</span>
                <span style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 0.95rem; letter-spacing: 0.05em; color: {border_clr};">{tier.upper()}</span>
            </div>
            <div style="display: block; width: 100%; margin-bottom: 0.8rem; box-sizing: border-box;">
                <div style="{badge_style}">🧠 RAM: <b style="color:#fff;">{item["RAM (MB)"]} MB</b></div>
                <div style="{badge_style}">🔋 Batt: <b style="color:#fff;">{item["Battery (mAh)"]} mAh</b></div>
                <div style="{badge_style}">💾 Storage: <b style="color:#fff;">{item["Internal Storage (GB)"]} GB</b></div>
                <div style="{badge_style}">⚡ Cores: <b style="color:#fff;">{item.get("Cores")} CPU</b></div>
                <div style="{badge_style}">🕒 Clock: <b style="color:#fff;">{item.get("Clock")} GHz</b></div>
                <div style="{badge_style}">⚖️ Weight: <b style="color:#fff;">{item.get("Weight")} g</b></div>
                <div style="{badge_style}">📏 Thick: <b style="color:#fff;">{item.get("Thick")} cm</b></div>
                <div style="{badge_style}">📸 Rear: <b style="color:#fff;">{item.get("PCam")} MP</b></div>
                <div style="{badge_style}">🤳 Front: <b style="color:#fff;">{item.get("FCam")} MP</b></div>
                <div style="{badge_style}">🖥️ Res: <b style="color:#fff;">{item.get("Width")}x{item.get("Height")}</b></div>
                <div style="{badge_style}">📐 Size: <b style="color:#fff;">{item.get("ScH")}x{item.get("ScW")} cm</b></div>
                <div style="{badge_style}">📞 Talk: <b style="color:#fff;">{item.get("Talk")} hrs</b></div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); padding: 6px 12px; border-radius: 6px; display: flex; gap: 8px; align-items: center; width: 100%; box-sizing: border-box;">
                <span style="font-size: 0.68rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.03em; white-space: nowrap;">🟢 ACTIVE:</span>
                <span style="font-size: 0.75rem; color: #3b82f6; font-weight: 500;">{features_txt}</span>
            </div>
        </div>
        """
        st.markdown(raw_html.replace("\n", ""), unsafe_allow_html=True)
