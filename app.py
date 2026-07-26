"""
ECG Arrhythmia Classifier — Streamlit Dashboard
================================================

Upload ECG data (500 data points), get arrhythmia predictions with
confidence scores and 1D Grad-CAM visualizations.

Hosted on Hugging Face Spaces.
"""

import json
import io

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from model import ECG_1D, CLASS_NAMES, CLASS_FULL_NAMES, NUM_CLASSES
from gradcam import create_gradcam_figure


# ======================================================================
#  CONFIGURABLE CONFIDENCE THRESHOLD
#  ----------------------------------
#  Adjust this single value to tune the rejection mechanism.
#  If the highest predicted class probability falls below this
#  threshold, the prediction is marked as "Uncertain / Inconclusive".
#
#  Recommended: calibrate on a held-out validation set.
# ======================================================================
CONFIDENCE_THRESHOLD: float = 0.60


# ======================================================================
#  Constants
# ======================================================================
HF_REPO_ID = "Codemaster67/ECG_Arythmia"
HF_FILENAME = "ECG_model_6_classes.pth"
EXPECTED_LENGTH = 500


# ======================================================================
#  Page configuration
# ======================================================================
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
#  Custom CSS — premium dark medical theme
# ======================================================================
st.markdown("""
<style>
/* ── Import Google Font ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root overrides ────────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ───────────────────────────────────── */
.stApp {
    background: linear-gradient(165deg, #0a0e1a 0%, #0d1321 40%, #111827 100%);
}

/* ── Main block container: stop content from tucking under
       the header / toggle arrow, and give it breathing room ── */
.block-container {
    padding-top: 3rem !important;
    max-width: 1200px;
}

/* ── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0f172a 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.08);
}

/* ── Cards ─────────────────────────────────────────────── */
.glass-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(20, 30, 55, 0.6));
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative;
    z-index: 1;
}

/* ── Prediction result cards ───────────────────────────── */
.prediction-confident {
    background: linear-gradient(135deg, rgba(0, 60, 40, 0.5), rgba(0, 80, 55, 0.3));
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.08);
}
.prediction-uncertain {
    background: linear-gradient(135deg, rgba(80, 50, 0, 0.4), rgba(100, 60, 0, 0.2));
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.08);
}

.prediction-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #94a3b8;
    margin-bottom: 8px;
    font-weight: 600;
}

.prediction-class {
    font-size: 32px;
    font-weight: 800;
    margin: 4px 0;
    line-height: 1.2;
}
.prediction-class.confident { color: #34d399; }
.prediction-class.uncertain { color: #fbbf24; }

.prediction-confidence {
    font-size: 48px;
    font-weight: 800;
    margin: 8px 0;
}
.prediction-confidence.confident { color: #10b981; }
.prediction-confidence.uncertain { color: #f59e0b; }

.prediction-subtext {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

/* ── Probability bars ──────────────────────────────────── */
.prob-container {
    margin-bottom: 10px;
}
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.prob-name {
    font-size: 13px;
    font-weight: 600;
    color: #cbd5e1;
}
.prob-value {
    font-size: 13px;
    font-weight: 700;
    color: #e2e8f0;
    font-variant-numeric: tabular-nums;
}
.prob-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(30, 41, 59, 0.8);
    border-radius: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ── Section headers ───────────────────────────────────── */
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header .icon {
    font-size: 20px;
}

/* ── Title area ────────────────────────────────────────── */
.app-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #7c3aed, #f43f5e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
    line-height: 1.15;
}
.app-subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 24px;
    font-weight: 400;
}

/* ── Upload area styling ───────────────────────────────── */
section[data-testid="stSidebar"] .stFileUploader > div {
    border-color: rgba(0, 212, 255, 0.2) !important;
}

/* ── Divider ───────────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
    margin: 20px 0;
}

/* ── Info badge ────────────────────────────────────────── */
.info-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
    color: #67e8f9;
    font-weight: 500;
}

/* ── Columns: stack cleanly on narrow / mobile viewports
       instead of overlapping ─────────────────────────────── */
@media (max-width: 768px) {
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}

/* ── Hide Streamlit branding elements (safely) ─────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Keep the header itself present (don't collapse its height —
   that's what was causing the toggle arrow and title to overlap),
   just make its background transparent. */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 3.5rem;
}

/* Hide only the Deploy button / toolbar — NOT the whole header,
   so we don't accidentally take the sidebar arrow down with it. */
div[data-testid="stToolbar"] {
    visibility: hidden;
}
.stDeployButton {
    display: none !important;
}

/* Force the sidebar collapse/expand arrow to always render,
   above everything else, and be clickable. */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    position: relative;
}
button[kind="header"] {
    visibility: visible !important;
    display: flex !important;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
#  Model loading (cached)
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Download model from HuggingFace Hub and load weights."""
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = ECG_1D(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ======================================================================
#  Inference
# ======================================================================
def run_inference(model: ECG_1D, ecg_data: np.ndarray):
    """
    Run model inference and return probabilities.

    Parameters
    ----------
    model : ECG_1D
    ecg_data : np.ndarray
        1D array of 500 data points.

    Returns
    -------
    probs : np.ndarray  — shape (NUM_CLASSES,), softmax probabilities
    pred_idx : int       — index of highest-probability class
    """
    tensor = torch.tensor(ecg_data, dtype=torch.float32).reshape(1, 1, EXPECTED_LENGTH)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx, tensor


# ======================================================================
#  Helpers
# ======================================================================
def get_bar_color(prob: float, is_top: bool) -> str:
    """Return gradient color for probability bar."""
    if is_top and prob >= CONFIDENCE_THRESHOLD:
        return "linear-gradient(90deg, #059669, #10b981, #34d399)"
    elif is_top:
        return "linear-gradient(90deg, #d97706, #f59e0b, #fbbf24)"
    elif prob > 0.15:
        return "linear-gradient(90deg, #1e40af, #3b82f6, #60a5fa)"
    else:
        return "linear-gradient(90deg, #1e293b, #334155, #475569)"


def parse_ecg_input(uploaded_file=None, json_text: str = "") -> tuple:
    """
    Parse ECG data from various input formats.

    Returns (data, error_message). data is None on failure.
    """
    try:
        if uploaded_file is not None:
            name = uploaded_file.name.lower()

            if name.endswith(".npy"):
                data = np.load(uploaded_file).flatten().astype(np.float32)

            elif name.endswith(".csv") or name.endswith(".txt"):
                content = uploaded_file.read().decode("utf-8")
                # Try comma-separated first, then newline-separated
                values = []
                for line in content.strip().split("\n"):
                    for val in line.split(","):
                        val = val.strip()
                        if val:
                            try:
                                values.append(float(val))
                            except ValueError:
                                continue
                if not values:
                    return None, "No numeric values found in the file."
                data = np.array(values, dtype=np.float32)

            else:
                return None, f"Unsupported file format: `{name}`. Use .csv, .txt, or .npy"

        elif json_text.strip():
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                data = np.array(parsed, dtype=np.float32).flatten()
            else:
                return None, "JSON input must be a list of numbers."
        else:
            return None, None  # No input yet — not an error

        # Validate length
        if len(data) < EXPECTED_LENGTH:
            return None, (
                f"**Insufficient data points.** Expected {EXPECTED_LENGTH}, "
                f"got {len(data)}. Please provide exactly {EXPECTED_LENGTH} values."
            )
        elif len(data) > EXPECTED_LENGTH:
            data = data[:EXPECTED_LENGTH]
            st.sidebar.info(
                f"ℹ️ Input had {len(data)} points — trimmed to first {EXPECTED_LENGTH}."
            )

        return data, None

    except json.JSONDecodeError:
        return None, "Invalid JSON. Please paste a valid JSON array, e.g. `[0.1, 0.2, ...]`"
    except Exception as e:
        return None, f"Error reading input: {str(e)}"


# ======================================================================
#  Sidebar
# ======================================================================
with st.sidebar:
    st.markdown('<div class="app-title">❤️ ECG AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Arrhythmia Classification with Explainable AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Upload section
    st.markdown("#### 📤 Upload ECG Data")
    st.caption("Upload a file with **500 data points** (single-lead ECG).")

    uploaded_file = st.file_uploader(
        "Choose file",
        type=["csv", "txt", "npy"],
        label_visibility="collapsed",
        help="CSV, TXT (comma or newline separated), or NumPy .npy file",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # JSON paste alternative
    with st.expander("📋 Or paste JSON array", expanded=False):
        json_text = st.text_area(
            "Paste ECG data as JSON",
            placeholder='[0.12, -0.05, 0.34, ...]  (500 values)',
            height=120,
            label_visibility="collapsed",
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Info section
    st.markdown("#### ℹ️ Model Info")
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px 20px;">
            <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">
                ARCHITECTURE
            </div>
            <div style="font-size:14px; color:#e2e8f0; font-weight:600; margin-bottom:12px;">
                ResNet-50 (1D)
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">
                CLASSES
            </div>
            <div style="font-size:13px; color:#e2e8f0; margin-bottom:12px;">
                {' · '.join(CLASS_NAMES)}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">
                INPUT SHAPE
            </div>
            <div style="font-size:13px; color:#e2e8f0; margin-bottom:12px;">
                (1, 1, 500)
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">
                CONFIDENCE THRESHOLD
            </div>
            <div style="font-size:13px; color:#67e8f9; font-weight:600;">
                {CONFIDENCE_THRESHOLD * 100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
#  Main content area
# ======================================================================

# Title
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <div class="app-title" style="font-size: 42px;">
            ECG Arrhythmia Classifier
        </div>
        <div class="app-subtitle" style="font-size: 16px;">
            Deep learning-powered ECG analysis with Grad-CAM explainability
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
with st.spinner("🔄 Loading model from HuggingFace Hub..."):
    model = load_model()

# Parse input
ecg_data, error_msg = parse_ecg_input(uploaded_file, json_text if 'json_text' in dir() else "")

if error_msg:
    st.error(error_msg)
    st.stop()

if ecg_data is None:
    # No input yet — show welcome state
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 64px; margin-bottom: 16px;">🫀</div>
            <div style="font-size: 22px; font-weight: 700; color: #e2e8f0; margin-bottom: 12px;">
                Upload ECG Data to Begin
            </div>
            <div style="font-size: 14px; color: #64748b; max-width: 480px; margin: 0 auto; line-height: 1.6;">
                Use the sidebar to upload a CSV, TXT, or NPY file containing
                <strong style="color: #67e8f9;">500 data points</strong> from a single-lead ECG recording.
                You can also paste data as a JSON array.
            </div>
            <div style="margin-top: 24px;">
                <span class="info-badge">Supported: CSV · TXT · NPY · JSON</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ======================================================================
#  Run inference
# ======================================================================
probs, pred_idx, input_tensor = run_inference(model, ecg_data)
max_prob = float(probs[pred_idx])
is_confident = max_prob >= CONFIDENCE_THRESHOLD

# ── ECG Signal Preview ────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="icon">📊</span> ECG Signal Preview</div>',
    unsafe_allow_html=True,
)

fig_signal, ax_signal = plt.subplots(figsize=(14, 3), facecolor="#0E1117")
ax_signal.set_facecolor("#0E1117")
ax_signal.plot(ecg_data, color="#00d4ff", linewidth=0.9, alpha=0.9)
ax_signal.fill_between(range(len(ecg_data)), ecg_data, alpha=0.08, color="#00d4ff")
ax_signal.set_xlim(0, len(ecg_data))
ax_signal.set_xlabel("Sample Index", color="#888888", fontsize=10)
ax_signal.set_ylabel("Amplitude", color="#888888", fontsize=10)
ax_signal.tick_params(colors="#666666", labelsize=8)
for spine in ax_signal.spines.values():
    spine.set_color("#222222")
ax_signal.grid(True, alpha=0.06, color="#ffffff")
plt.tight_layout()
st.pyplot(fig_signal)
plt.close(fig_signal)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Prediction + Probabilities (side by side) ─────────────────────────
col_pred, col_probs = st.columns([1, 1.3], gap="large")

with col_pred:
    st.markdown(
        '<div class="section-header"><span class="icon">🎯</span> Prediction</div>',
        unsafe_allow_html=True,
    )

    if is_confident:
        pred_class = CLASS_NAMES[pred_idx]
        pred_full = CLASS_FULL_NAMES.get(pred_class, pred_class)
        st.markdown(
            f"""
            <div class="prediction-confident">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-class confident">{pred_full}</div>
                <div style="font-size: 13px; color: #6ee7b7; margin-bottom: 12px;">
                    ({pred_class})
                </div>
                <div class="prediction-confidence confident">
                    {max_prob * 100:.1f}%
                </div>
                <div class="prediction-subtext">
                    Confidence above threshold ({CONFIDENCE_THRESHOLD * 100:.0f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        candidate_class = CLASS_NAMES[pred_idx]
        candidate_full = CLASS_FULL_NAMES.get(candidate_class, candidate_class)
        st.markdown(
            f"""
            <div class="prediction-uncertain">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-class uncertain">Uncertain / Inconclusive</div>
                <div class="prediction-confidence uncertain">
                    {max_prob * 100:.1f}%
                </div>
                <div class="prediction-subtext" style="margin-top: 12px;">
                    Highest candidate: <strong style="color: #fbbf24;">{candidate_full}</strong>
                    ({candidate_class}) at {max_prob * 100:.1f}%
                </div>
                <div class="prediction-subtext" style="margin-top: 4px;">
                    Below confidence threshold ({CONFIDENCE_THRESHOLD * 100:.0f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_probs:
    st.markdown(
        '<div class="section-header"><span class="icon">📈</span> Class Probabilities</div>',
        unsafe_allow_html=True,
    )

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]

    bars_html = ""
    for idx in sorted_indices:
        name = CLASS_NAMES[idx]
        full_name = CLASS_FULL_NAMES.get(name, name)
        prob = float(probs[idx])
        is_top = idx == pred_idx
        color = get_bar_color(prob, is_top)
        width_pct = max(prob * 100, 0.5)  # minimum visible width

        highlight = ""
        if is_top:
            highlight = ' style="color: #34d399; font-weight: 700;"' if is_confident else ' style="color: #fbbf24; font-weight: 700;"'

        bars_html += f"""
        <div class="prob-container">
            <div class="prob-header">
                <span class="prob-name"{highlight}>{full_name} ({name})</span>
                <span class="prob-value"{highlight}>{prob * 100:.2f}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width: {width_pct}%; background: {color};"></div>
            </div>
        </div>
        """

    st.markdown(
        f'<div class="glass-card">{bars_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Grad-CAM Visualization ────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><span class="icon">🔬</span> Grad-CAM Explainability</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Highlights the regions of the ECG signal that most influenced the model's prediction. "
    "Warmer colors indicate higher importance."
)

# Need gradients → enable grad for this block
input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
target_layer = model.model.layer4[2].conv3

gradcam_target = pred_idx
fig_gradcam = create_gradcam_figure(
    model,
    target_layer,
    input_tensor_grad,
    gradcam_target,
    class_name=CLASS_FULL_NAMES.get(CLASS_NAMES[pred_idx], CLASS_NAMES[pred_idx]),
)
st.pyplot(fig_gradcam)
plt.close(fig_gradcam)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; padding: 16px 0; color: #475569; font-size: 12px;">
        Built with Streamlit · Model: ResNet-50 (1D) · 
        <a href="https://huggingface.co/Codemaster67/ECG_Arythmia" 
           target="_blank" style="color: #67e8f9; text-decoration: none;">
            View on HuggingFace
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)