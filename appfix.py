import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="PhonePredict",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css():
    with open("sty.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

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

if "_pending" in st.session_state:
    for k, v in st.session_state["_pending"].items():
        st.session_state[k] = v
    del st.session_state["_pending"]

FEATURE_ORDER = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi",
]

PRICE_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
PRICE_COLORS = {
    0: ("#eef4fd", "#1a5fa8", "#1a5fa8"),
    1: ("#edf7ed", "#2e7d32", "#2e7d32"),
    2: ("#fff8ed", "#b35c00", "#b35c00"),
    3: ("#fdf0f0", "#b71c1c", "#b71c1c"),
}
BAR_COLORS = ["#378ADD", "#63991F", "#EF9F27", "#E24B4A"]

VALID_RANGES = {
    "battery_power": (501,  1998),
    "ram":           (256,  3998),
    "clock_speed":   (0.5,  3.0),
    "n_cores":       (1,    8),
    "int_memory":    (2,    64),
    "pc":            (0,    20),
    "fc":            (0,    19),
    "px_height":     (0,    1960),
    "px_width":      (0,    1998),
    "sc_h":          (5,    19),
    "sc_w":          (0,    18),
    "mobile_wt":     (80,   200),
    "m_dep":         (0.1,  1.0),
    "talk_time":     (2,    20),
}

# header
st.markdown("""
<div class="header">
    <div class="header-tag">📱 PhonePredict</div>
    <h1>Phone Price Level Predictor</h1>
    <p>Enter your smartphone's technical specs below and let our ML model predict its market price range.</p>
</div>
""", unsafe_allow_html=True)

# helper functions for the input
def yesno(label, key):
    val = st.radio(label, ["Yes", "No"], horizontal=True, key=key, index=1)
    return 1 if val == "Yes" else 0

# change the minimum into the actual minimum range (not 0)
def num_input(label, key, min_v, max_v, default=None, step=1, fmt="%g", help_txt=None):
    raw_val = st.session_state.get(key, None)
    if raw_val is None or float(raw_val) < float(min_v):
        val_default = float(min_v)
    else:
        val_default = float(raw_val)
    return st.number_input(
        label,
        min_value=float(min_v),
        max_value=float(max_v),
        value=val_default,
        step=float(step),
        format=fmt,
        help=help_txt,
        key=key
    )

# shortcuts
st.markdown('<div class="section-label">⚡ Shortcuts</div>', unsafe_allow_html=True)

SHORTCUTS = {
    "🧱 Friendly-Budget": {
        "desc": "Basic daily use phone",
        "data": dict(
            battery_power=700.0, ram=512.0, clock_speed=0.8, n_cores=2.0,
            int_memory=4.0, pc=5.0, fc=2.0, px_height=200.0, px_width=600.0,
            sc_h=8.0, sc_w=3.0, mobile_wt=185.0, m_dep=0.8, talk_time=5.0,
            blue="No", dual_sim="Yes", four_g="No", three_g="Yes", wifi="No", touch_screen="No",
        ),
    },
    "⚡ Mid-Range": {
        "desc": "Balanced performance & value",
        "data": dict(
            battery_power=1200.0, ram=2100.0, clock_speed=1.5, n_cores=4.0,
            int_memory=32.0, pc=11.0, fc=5.0, px_height=560.0, px_width=1250.0,
            sc_h=12.0, sc_w=5.0, mobile_wt=140.0, m_dep=0.5, talk_time=11.0,
            blue="Yes", dual_sim="Yes", four_g="Yes", three_g="Yes", wifi="Yes", touch_screen="Yes",
        ),
    },
    "🚀 High-End": {
        "desc": "Heavy multitasking & gaming",
        "data": dict(
            battery_power=1600.0, ram=2800.0, clock_speed=2.2, n_cores=6.0,
            int_memory=32.0, pc=15.0, fc=8.0, px_height=900.0, px_width=1500.0,
            sc_h=14.0, sc_w=9.0, mobile_wt=120.0, m_dep=0.3, talk_time=15.0,
            blue="Yes", dual_sim="Yes", four_g="Yes", three_g="Yes", wifi="Yes", touch_screen="Yes",
        ),
    },
    "👑 Premium": {
        "desc": "No compromises, flagship tier",
        "data": dict(
            battery_power=1950.0, ram=3850.0, clock_speed=2.9, n_cores=8.0,
            int_memory=64.0, pc=20.0, fc=16.0, px_height=1400.0, px_width=1900.0,
            sc_h=18.0, sc_w=14.0, mobile_wt=95.0, m_dep=0.1, talk_time=20.0,
            blue="Yes", dual_sim="Yes", four_g="Yes", three_g="Yes", wifi="Yes", touch_screen="Yes",
        ),
    },
}

p_col1, p_col2, p_col3, p_col4 = st.columns(4)
for col, (label, info) in zip([p_col1, p_col2, p_col3, p_col4], SHORTCUTS.items()):
    with col:
        d = info["data"]
        conn_pills = []
        if d.get("blue") == "Yes":         conn_pills.append("🔵 BT")
        if d.get("dual_sim") == "Yes":     conn_pills.append("🎴 Dual")
        if d.get("four_g") == "Yes":       conn_pills.append("📶 4G")
        if d.get("three_g") == "Yes":      conn_pills.append("🌐 3G")
        if d.get("wifi") == "Yes":         conn_pills.append("🛜 Wi-Fi")
        if d.get("touch_screen") == "Yes": conn_pills.append("👆 Touch")
        pills_html = "".join(
            f'<span class="sc-pill">{p}</span>' for p in conn_pills
        ) if conn_pills else '<span class="sc-pill">—</span>'

        specs_html = f"""
            <div class="sc-spec-row"><span class="sc-spec-key">🧠 RAM</span><span class="sc-spec-val">{int(d['ram'])} MB</span></div>
            <div class="sc-spec-row"><span class="sc-spec-key">🔋 Battery</span><span class="sc-spec-val">{int(d['battery_power'])} mAh</span></div>
            <div class="sc-spec-row"><span class="sc-spec-key">💾 Storage</span><span class="sc-spec-val">{int(d['int_memory'])} GB</span></div>
            <div class="sc-spec-row"><span class="sc-spec-key">⚡ CPU</span><span class="sc-spec-val">{int(d['n_cores'])}c · {d['clock_speed']} GHz</span></div>
            <div class="sc-spec-row"><span class="sc-spec-key">📸 Cam</span><span class="sc-spec-val">{int(d['pc'])} / {int(d['fc'])} MP</span></div>
        """
        st.markdown(f"""
        <div class="shortcut-card">
            <div class="sc-label">{label}</div>
            <div class="sc-desc">{info['desc']}</div>
            <div class="sc-specs-block">{specs_html}</div>
            <div class="sc-pills">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Load {label}", key=f"preset_{label}", use_container_width=True):
            st.session_state["_pending"] = dict(info["data"])
            st.rerun()

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# category 1-4 specs
st.markdown('<div class="section-label">⚙️ Technical Specs</div>', unsafe_allow_html=True)
col_core, col_cam, col_disp, col_phys = st.columns(4)

with col_core:
    with st.expander("⚡ Core Performance", expanded=True):
        battery_power = num_input("🔋 Battery (mAh)", "battery_power", 501, 1998, step=50)
        st.caption("501 – 1998 mAh")
        ram = num_input("🧠 RAM (MB)", "ram", 256, 3998, step=64)
        st.caption("256 – 3998 MB")
        clock_speed = num_input("🕒 Clock (GHz)", "clock_speed", 0.5, 3.0, step=0.1, fmt="%.1f")
        st.caption("0.5 – 3.0 GHz")
        n_cores = num_input("⚡ CPU cores", "n_cores", 1, 8)
        st.caption("1 – 8 cores")
        int_memory = num_input("💾 Storage (GB)", "int_memory", 2, 64)
        st.caption("2 – 64 GB")

with col_cam:
    with st.expander("📸 Camera", expanded=True):
        pc = num_input("📸 Rear cam (MP)", "pc", 0, 20)
        st.caption("0 – 20 MP")
        fc = num_input("🤳 Front cam (MP)", "fc", 0, 19)
        st.caption("0 – 19 MP")

with col_disp:
    with st.expander("🖥️ Display", expanded=True):
        px_height = num_input("↕️ Pixel height", "px_height", 0, 1960, step=10)
        st.caption("0 – 1960 px")
        px_width = num_input("↔️ Pixel width", "px_width", 0, 1998, step=10)
        st.caption("0 – 1998 px")
        sc_h = num_input("📏 Screen Height (cm)", "sc_h", 5, 19)
        st.caption("5 – 19 cm")
        sc_w = num_input("📐 Screen Width (cm)", "sc_w", 0, 18)
        st.caption("0 – 18 cm")

with col_phys:
    with st.expander("📦 Physical", expanded=True):
        mobile_wt = num_input("⚖️ Weight (g)", "mobile_wt", 80, 200)
        st.caption("80 – 200 g")
        m_dep = num_input("📏 Thickness (cm)", "m_dep", 0.1, 1.0, step=0.1, fmt="%.1f")
        st.caption("0.1 – 1.0 cm")
        talk_time = num_input("📞 Talk time (hrs)", "talk_time", 2, 20)
        st.caption("2 – 20 hours")

# ── SECTION 05: CONNECTIVITY ──────────────────────────────
with st.expander("📶 Connectivity & Features"):
    c14, c15, c16 = st.columns(3)
    with c14:
        blue     = yesno("🔵 Bluetooth", "blue")
        dual_sim = yesno("🎴 Dual SIM",  "dual_sim")
    with c15:
        four_g   = yesno("📶 4G / LTE", "four_g")
        three_g  = yesno("🌐 3G",       "three_g")
    with c16:
        wifi         = yesno("🛜 Wi-Fi",       "wifi")
        touch_screen = yesno("🤳 Touch screen", "touch_screen")

# # ── CHANGE 1: Clear All moved here (bottom of inputs) ─────
# st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
# if st.button("🔄 Clear All Input Fields", use_container_width=True):
#     categorical_features = {"blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"}
#     pending = {}
#     for k in FEATURE_ORDER:
#         if k in categorical_features:
#             pending[k] = "No"
#         elif k in VALID_RANGES:
#             pending[k] = float(VALID_RANGES[k][0])  # min, bukan 0.0
#         else:
#             pending[k] = 0.0
#     st.session_state["_pending"] = pending
#     st.rerun()

# ── PREDICT BUTTON ────────────────────────────────────────
if st.button("→  Predict Price Range", use_container_width=True):
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

    errors = []
    for field, (lo, hi) in VALID_RANGES.items():
        val = raw[field]
        if val != 0 and not (lo <= val <= hi):
            errors.append(f"**{field.replace('_', ' ')}**: {val} (valid: {lo}–{hi})")

    if errors:
        st.error("Some inputs are outside valid range:\n\n" + "\n\n".join(errors))
        st.stop()

    filled = {k: (v if v != 0 or k in ("blue","dual_sim","four_g","three_g","wifi","touch_screen")
                    else MEDIANS.get(k, v))
              for k, v in raw.items()}
    input_vec = [filled[f] for f in FEATURE_ORDER]

    pred   = model.predict([input_vec])[0]
    probas = model.predict_proba([input_vec])[0]

    label = PRICE_LABELS[pred]
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

    TIER_INSIGHTS = {
        0: "🧱 <b>Entry-Level Range</b>: Optimized for budget efficiency. Best suited for essential daily needs, messaging, and long-lasting battery standby.",
        1: "⚡ <b>Mid-Range Standard</b>: The sweet spot. Offers reliable multitasking, casual gaming, and great value for money.",
        2: "🔥 <b>High-Tier Performer</b>: Stepping into premium. Capable of heavy multitasking, intensive gaming, and fast media rendering.",
        3: "👑 <b>Ultimate Flagship</b>: No compromises! Top-tier hardware engineered for extreme performance and cutting-edge features."
    }
    insight_text = TIER_INSIGHTS[pred]

    # Result card — simple label
    st.markdown(f"""
    <div class="result-card" style="background:{bg}; border: 1.5px solid {text_color}33; width:100%; box-sizing:border-box; text-align:center; margin-bottom: 1rem;">
        <div class="result-label" style="color:{text_color}; margin-bottom: 0.4rem;">PREDICTED PRICE RANGE</div>
        <div class="result-badge" style="color:{text_color}; margin: 0.3rem 0 0.2rem;">{label}</div>
        <div class="result-sub" style="color:{text_color};">Price tier {pred + 1} of 4</div>
    </div>

    <div style="background: #f8fafc; border-left: 4px solid {text_color}; padding: 1rem 1.3rem; border-radius: 8px; margin-bottom: 1.5rem; color: #1e293b; font-size: 0.92rem; line-height: 1.6;">
        {insight_text}
    </div>

    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: #f1f5f9; border: 1px solid #e2e8f0; padding: 1.2rem; border-radius: 12px; margin-bottom: 1.8rem; text-align: center;">
        <div>
            <span style="color: #64748b; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">RAM</span>
            <strong style="color: #0f172a; font-size: 1.1rem;">{int(filled['ram'])} <span style="font-size: 0.8rem; color: #94a3b8;">MB</span></strong>
        </div>
        <div>
            <span style="color: #64748b; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Battery</span>
            <strong style="color: #0f172a; font-size: 1.1rem;">{int(filled['battery_power'])} <span style="font-size: 0.8rem; color: #94a3b8;">mAh</span></strong>
        </div>
        <div>
            <span style="color: #64748b; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Storage</span>
            <strong style="color: #0f172a; font-size: 1.1rem;">{int(filled['int_memory'])} <span style="font-size: 0.8rem; color: #94a3b8;">GB</span></strong>
        </div>
        <div>
            <span style="color: #64748b; font-size: 0.72rem; display: block; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">CPU Cores</span>
            <strong style="color: #0f172a; font-size: 1.1rem;">{int(filled['n_cores'])} <span style="font-size: 0.8rem; color: #94a3b8;">Cores</span></strong>
        </div>
    </div>

    <div style="margin-top: 1rem; padding-left: 2px; margin-bottom: 0.6rem;">
        <span style="font-family: 'Poppins', sans-serif; font-size: 0.75rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;">Confidence Breakdown</span>
    </div>

    <div style="margin-top: 0.5rem;">
    """, unsafe_allow_html=True)

    for i, (prob, clr) in enumerate(zip(probas, BAR_COLORS)):
        pct = round(prob * 100, 1)
        active_weight = "font-weight:600;" if i == pred else ""
        st.markdown(f"""
        <div class="bar-row" style="{active_weight}">
            <span style="min-width:90px; color:{clr if i==pred else '#aaa'};">{PRICE_LABELS[i]}</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%; background:{clr};"></div>
            </div>
            <span class="bar-pct" style="color:#334155;">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    zeros_filled = [k for k, v in raw.items()
                    if v == 0 and k not in ("blue","dual_sim","four_g","three_g","wifi","touch_screen")]
    if zeros_filled:
        DISPLAY_NAMES = {
            "battery_power": "Battery", "clock_speed": "Clock speed", "fc": "Front cam",
            "int_memory": "Storage", "m_dep": "Thickness", "mobile_wt": "Weight",
            "n_cores": "CPU cores", "pc": "Rear cam", "px_height": "Pixel height",
            "px_width": "Pixel width", "ram": "RAM", "sc_h": "Screen height",
            "sc_w": "Screen width", "talk_time": "Talk time"
        }
        clean_names = [DISPLAY_NAMES.get(k, k) for k in zeros_filled]
        st.markdown(
            f'<div class="median-note">ℹ️ Fields left at 0 were auto-filled with dataset medians: {", ".join(clean_names)}</div>',
            unsafe_allow_html=True
        )

# ── CHANGE 1: Clear All moved here (bottom of inputs) ─────
st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
if st.button("🔄 Clear All Input Fields", use_container_width=True):
    categorical_features = {"blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"}
    pending = {}
    for k in FEATURE_ORDER:
        if k in categorical_features:
            pending[k] = "No"
        elif k in VALID_RANGES:
            pending[k] = float(VALID_RANGES[k][0])  # min, bukan 0.0
        else:
            pending[k] = 0.0
    st.session_state["_pending"] = pending
    st.rerun()

# ── HISTORY ───────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="section-label">📋 Prediction History</div>', unsafe_allow_html=True)

    HIST_COLORS = {
        "Low":       ("#1a5fa8", "#eef4fd"),
        "Medium":    ("#2e7d32", "#edf7ed"),
        "High":      ("#b35c00", "#fff8ed"),
        "Very High": ("#b71c1c", "#fdf0f0"),
    }

    for item in st.session_state.history:
        tier = item["Predicted Tier"]
        border_clr, bg_clr = HIST_COLORS.get(tier, ("#475569", "#f8fafc"))

        active_features = []
        if item.get("blue") == 1: active_features.append("🔵 BT")
        if item.get("dual_sim") == 1: active_features.append("🎴 Dual SIM")
        if item.get("four_g") == 1: active_features.append("📶 4G")
        if item.get("three_g") == 1: active_features.append("🌐 3G")
        if item.get("touch_screen") == 1: active_features.append("📱 Touch")
        if item.get("wifi") == 1: active_features.append("🛜 Wi-Fi")
        features_txt = " · ".join(active_features) if active_features else "None"

        badge_style = "display:inline-block; margin:3px; padding:4px 9px; font-size:0.73rem; color:#475569; background:#fff; border:1px solid #e2e8f0; border-radius:5px; white-space:nowrap;"

        raw_html = f"""
        <div style="padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 0.8rem; border-left: 4px solid {border_clr}; background: {bg_clr}; box-sizing: border-box; font-family: 'Inter', sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem;">
                <span style="font-size: 0.78rem; color: #94a3b8;">⏱️ {item["Time"]}</span>
                <span style="font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 0.9rem; color: {border_clr};">{tier.upper()}</span>
            </div>
            <div style="margin-bottom: 0.7rem; box-sizing: border-box;">
                <div style="{badge_style}">🧠 RAM: <b style="color:#0f172a;">{item["RAM (MB)"]} MB</b></div>
                <div style="{badge_style}">🔋 {item["Battery (mAh)"]} mAh</div>
                <div style="{badge_style}">💾 {item["Internal Storage (GB)"]} GB</div>
                <div style="{badge_style}">⚡ {item.get("Cores")} cores · {item.get("Clock")} GHz</div>
                <div style="{badge_style}">⚖️ {item.get("Weight")}g · {item.get("Thick")}cm</div>
                <div style="{badge_style}">📸 {item.get("PCam")}MP / 🤳{item.get("FCam")}MP</div>
                <div style="{badge_style}">🖥️ {item.get("Width")}×{item.get("Height")}px</div>
                <div style="{badge_style}">📞 {item.get("Talk")} hrs</div>
            </div>
            <div style="background: rgba(0,0,0,0.03); border: 1px solid #e2e8f0; padding: 5px 10px; border-radius: 6px; font-size: 0.75rem; color: #475569;">
                🟢 {features_txt}
            </div>
        </div>
        """
        st.markdown(raw_html.replace("\n", ""), unsafe_allow_html=True)
