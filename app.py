import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.interpolate import interp1d

# ==============================================================================
# SUBTYPE CONFIGURATIONS (Marczyk et al.)
# ==============================================================================
SUBTYPE_CONFIGS = {
    'HR+/HER2-': {'scale':  1.00, 'x0': 3.50, 'desc': 'Decreasing weight (down-weights high residual burden)'},
    'HER2+':     {'scale': -0.50, 'x0': 3.50, 'desc': 'Increasing weight (moderately emphasizes higher burden)'},
    'TN':        {'scale': -1.00, 'x0': 2.00, 'desc': 'Steeply increasing weight beyond x0=2.0 (strongly penalizes residual burden)'},
    'Unweighted (Standard)': {'scale': 0.00, 'x0': 0.00, 'desc': 'Uniform observation weighting'}
}

# ==============================================================================
# SYNTHETIC CLINICAL TRIAL GENERATOR
# ==============================================================================
def generate_synthetic_rcb(n=350, pcr_rate=0.25, alpha=2.0, beta_param=3.5, max_rcb=5.0, seed=None):
    """
    Simulates realistic clinical trial RCB values using a zero-point mass (pCR)
    and a scaled Beta distribution for residual disease.
    """
    rng = np.random.default_rng(seed)
    n_pcr = int(n * pcr_rate)
    n_res = n - n_pcr
    
    # Residual cancer burden values scaled to clinical support [0.1, max_rcb]
    res_scores = rng.beta(alpha, beta_param, size=n_res) * (max_rcb - 0.2) + 0.2
    all_scores = np.concatenate([np.zeros(n_pcr), res_scores])
    rng.shuffle(all_scores)
    return np.round(all_scores, 2)

# ==============================================================================
# MARZYCK WEIGHTED ECDF & STATISTICAL ENGINE
# ==============================================================================
def marczyk_wecdf(x, scale=0.0, x0=0.0):
    """Constructs the weighted empirical CDF."""
    if len(x) == 0:
        return lambda grid: np.zeros_like(grid, dtype=float)

    vals, counts = np.unique(np.sort(x), return_counts=True)
    weights = 2.0 / (1.0 + np.exp(scale * (vals - x0)))
    w_pdf = counts * weights
    total_weight = np.sum(w_pdf)
    w_cdf = np.cumsum(w_pdf) / total_weight if total_weight > 0 else np.zeros_like(w_pdf)

    return interp1d(vals, w_cdf, kind='previous', bounds_error=False, fill_value=(0.0, 1.0))

def compute_marczyk_tes(rcb_A, rcb_B, scale=0.0, x0=0.0):
    """Calculates normalized Marczyk TES between Treatment B and Treatment A."""
    if len(rcb_A) == 0 or len(rcb_B) == 0:
        return 0.0

    grid = np.sort(np.unique(np.concatenate(([0.0], rcb_A, rcb_B))))
    max_g = grid[-1]
    if max_g == 0:
        return 0.0

    f_A = marczyk_wecdf(rcb_A, scale=scale, x0=x0)(grid)
    f_B = marczyk_wecdf(rcb_B, scale=scale, x0=x0)(grid)

    delta_f = f_B[:-1] - f_A[:-1]
    dx = np.diff(grid)
    return float(np.sum(delta_f * dx) / max_g)

def run_permutation_test(rcb_A, rcb_B, subtype_key, n_perms=1000):
    """Permutation test under H0 for 1,000 cycles."""
    cfg = SUBTYPE_CONFIGS[subtype_key]
    s, x0 = cfg['scale'], cfg['x0']

    obs_unwt = compute_marczyk_tes(rcb_A, rcb_B, scale=0.0, x0=0.0)
    obs_wt   = compute_marczyk_tes(rcb_A, rcb_B, scale=s, x0=x0)

    pooled = np.concatenate([rcb_A, rcb_B])
    nA = len(rcb_A)

    null_unwt = np.empty(n_perms)
    null_wt   = np.empty(n_perms)

    for i in range(n_perms):
        shuffled = np.random.permutation(pooled)
        p_A, p_B = shuffled[:nA], shuffled[nA:]
        null_unwt[i] = compute_marczyk_tes(p_A, p_B, scale=0.0, x0=0.0)
        null_wt[i]   = compute_marczyk_tes(p_A, p_B, scale=s, x0=x0)

    p_unwt = float(np.mean(np.abs(null_unwt) >= np.abs(obs_unwt)))
    p_wt   = float(np.mean(np.abs(null_wt) >= np.abs(obs_wt)))

    return obs_unwt, p_unwt, obs_wt, p_wt

def parse_input_text(raw_text):
    """Extracts non-negative numbers from string inputs."""
    if not raw_text or not raw_text.strip():
        return np.array([])
    clean = raw_text.replace(',', ' ').replace('\n', ' ')
    out = []
    for tok in clean.split():
        try:
            val = float(tok)
            if val >= 0:
                out.append(val)
        except ValueError:
            continue
    return np.array(out)

def get_arm_metrics(scores):
    n = len(scores)
    k = int(np.sum(scores == 0))
    pcr_rate = k / n if n > 0 else 0.0
    return {
        'n': n,
        'pcr_count': k,
        'pcr_rate': pcr_rate,
        'rcb0': k,
        'rcb1': int(np.sum((scores > 0) & (scores <= 1.36))),
        'rcb2': int(np.sum((scores > 1.36) & (scores <= 3.28))),
        'rcb3': int(np.sum(scores > 3.28)),
        'mean': np.mean(scores) if n > 0 else 0.0,
        'median': np.median(scores) if n > 0 else 0.0
    }

# ==============================================================================
# STATE INITIALIZATION (Default 350 Patients Per Arm)
# ==============================================================================
if 'default_rcb_A' not in st.session_state:
    # Treatment A: 25% pCR rate, typical residual profile
    st.session_state.default_rcb_A = generate_synthetic_rcb(
        n=350, pcr_rate=0.25, alpha=2.2, beta_param=2.8, seed=42
    )

if 'default_rcb_B' not in st.session_state:
    # Treatment B: 42% pCR rate, shifted toward lower residual burden
    st.session_state.default_rcb_B = generate_synthetic_rcb(
        n=350, pcr_rate=0.42, alpha=1.8, beta_param=3.8, seed=101
    )

# ==============================================================================
# UI CONFIGURATION & CONTROLS
# ==============================================================================
st.set_page_config(page_title="RCB & TES Trial Comparator", layout="wide")

st.title("Residual Cancer Burden (RCB) Clinical Trial Comparator")
st.markdown(
    "Benchmark **Treatment Arm A (Control/Reference)** vs. **Treatment Arm B (Experimental)**. "
    "Preloaded with **350 randomized patients per arm**. Calculates $\\Delta$pCR, standard continuous TES, and "
    "**Marczyk Subtype-Specific TES** with a 1,000-permutation test."
)

with st.sidebar:
    st.header("1. Subtype & Permutations")
    selected_subtype = st.selectbox(
        "Clinical Subtype",
        options=list(SUBTYPE_CONFIGS.keys()),
        index=0
    )
    cfg = SUBTYPE_CONFIGS[selected_subtype]
    st.info(f"**Scale ($s$):** `{cfg['scale']:.2f}`\n\n**Shift ($x_0$):** `{cfg['x0']:.2f}`\n\n*{cfg['desc']}*")

    n_perms = st.number_input("Permutations for p-value", min_value=100, max_value=10000, value=1000, step=100)
    
    st.divider()
    st.header("2. Synthetic Sample Generator")
    if st.button("Generate New Random Cohorts (350 each)"):
        st.session_state.default_rcb_A = generate_synthetic_rcb(n=350, pcr_rate=0.25, alpha=2.2, beta_param=2.8)
        st.session_state.default_rcb_B = generate_synthetic_rcb(n=350, pcr_rate=0.42, alpha=1.8, beta_param=3.8)
        st.rerun()

    input_mode = st.radio("Input Mode", ["Pasted / Default Text", "Upload CSV Files"])

col_A, col_B = st.columns(2)

if input_mode == "Pasted / Default Text":
    default_str_A = ", ".join(map(str, st.session_state.default_rcb_A))
    default_str_B = ", ".join(map(str, st.session_state.default_rcb_B))

    with col_A:
        st.subheader("Treatment Arm A (Reference/Control)")
        txt_A = st.text_area(
            "RCB values (350 randomized patients preloaded):",
            value=default_str_A,
            height=200
        )
        rcb_A = parse_input_text(txt_A)

    with col_B:
        st.subheader("Treatment Arm B (Experimental)")
        txt_B = st.text_area(
            "RCB values (350 randomized patients preloaded):",
            value=default_str_B,
            height=200
        )
        rcb_B = parse_input_text(txt_B)

else:
    with col_A:
        st.subheader("Treatment Arm A (Reference/Control)")
        file_A = st.file_uploader("Upload CSV for Arm A (Column named 'RCB')", type=["csv"], key="fileA")
        if file_A:
            dfA = pd.read_csv(file_A)
            col_candidates = [c for c in dfA.columns if 'rcb' in c.lower()]
            rcb_A = dfA[col_candidates[0]].dropna().values if col_candidates else np.array([])
        else:
            rcb_A = st.session_state.default_rcb_A

    with col_B:
        st.subheader("Treatment Arm B (Experimental)")
        file_B = st.file_uploader("Upload CSV for Arm B (Column named 'RCB')", type=["csv"], key="fileB")
        if file_B:
            dfB = pd.read_csv(file_B)
            col_candidates = [c for c in dfB.columns if 'rcb' in c.lower()]
            rcb_B = dfB[col_candidates[0]].dropna().values if col_candidates else np.array([])
        else:
            rcb_B = st.session_state.default_rcb_B

st.divider()

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================
if st.button("Run Full Trial Comparison", type="primary"):
    if len(rcb_A) < 3 or len(rcb_B) < 3:
        st.error("Please supply at least 3 valid non-negative numerical RCB scores for both Arm A and Arm B.")
    else:
        with st.spinner(f"Computing weighted eCDFs and running {n_perms:,} permutations for {len(rcb_A) + len(rcb_B)} patients..."):
            m_A = get_arm_metrics(rcb_A)
            m_B = get_arm_metrics(rcb_B)

            # Categorical statistics
            delta_pcr = m_B['pcr_rate'] - m_A['pcr_rate']
            res_A = m_A['n'] - m_A['pcr_count']
            res_B = m_B['n'] - m_B['pcr_count']
            odds_ratio = (m_B['pcr_count'] / max(1, res_B)) / (m_A['pcr_count'] / max(1, res_A))

            # Contingency Chi-Square test
            ctable = np.array([[m_A['pcr_count'], res_A], [m_B['pcr_count'], res_B]])
            expected = np.outer(ctable.sum(axis=1), ctable.sum(axis=0)) / ctable.sum()
            chi2_val = np.sum(((ctable - expected) ** 2) / (expected + 1e-6))
            pcr_p = float(chi2.sf(chi2_val, df=1))

            # Marczyk TES & 1000 Permutations
            unwt_tes, unwt_p, wt_tes, wt_p = run_permutation_test(
                rcb_A, rcb_B, subtype_key=selected_subtype, n_perms=int(n_perms)
            )

        st.success("Analysis Complete!")

        # High-level metric highlights
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("$\Delta$pCR (Arm B - Arm A)", f"{delta_pcr:+.3f}", f"Chi² p = {pcr_p:.4f}")
        c2.metric("pCR Odds Ratio", f"{odds_ratio:.3f}", f"{m_B['pcr_count']}/{m_B['n']} vs {m_A['pcr_count']}/{m_A['n']}")
        c3.metric("Standard TES", f"{unwt_tes:+.4f}", f"Perm p = {unwt_p:.4f}")
        c4.metric(f"Subtype TES ({selected_subtype})", f"{wt_tes:+.4f}", f"Perm p = {wt_p:.4f}")

        # Summary Table
        st.subheader("Statistical Summary")
        res_df = pd.DataFrame({
            "Metric": ["ΔpCR", "Standard TES", f"Subtype-Specific TES ({selected_subtype})"],
            "Estimate": [f"{delta_pcr:+.4f}", f"{unwt_tes:+.4f}", f"{wt_tes:+.4f}"],
            "Hypothesis Test": ["Pearson Chi-Square", f"{n_perms} Permutations", f"{n_perms} Permutations"],
            "p-value": [f"{pcr_p:.4f}", f"{unwt_p:.4f}", f"{wt_p:.4f}"],
            "Significant (α = 0.05)": [pcr_p < 0.05, unwt_p < 0.05, wt_p < 0.05]
        })
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Categorical Breakdown Table
        st.subheader("Residual Cancer Burden Category Breakdown")
        cat_df = pd.DataFrame([
            {
                "Arm": "Treatment A (Ref)", "Total N": m_A['n'],
                "pCR (RCB 0)": f"{m_A['rcb0']} ({m_A['pcr_rate']*100:.1f}%)",
                "RCB-I (≤ 1.36)": m_A['rcb1'],
                "RCB-II (1.36 - 3.28)": m_A['rcb2'],
                "RCB-III (> 3.28)": m_A['rcb3'],
                "Mean RCB": f"{m_A['mean']:.2f}",
                "Median RCB": f"{m_A['median']:.2f}"
            },
            {
                "Arm": "Treatment B (Exp)", "Total N": m_B['n'],
                "pCR (RCB 0)": f"{m_B['rcb0']} ({m_B['pcr_rate']*100:.1f}%)",
                "RCB-I (≤ 1.36)": m_B['rcb1'],
                "RCB-II (1.36 - 3.28)": m_B['rcb2'],
                "RCB-III (> 3.28)": m_B['rcb3'],
                "Mean RCB": f"{m_B['mean']:.2f}",
                "Median RCB": f"{m_B['median']:.2f}"
            }
        ])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        # eCDF Plots
        st.subheader("Cumulative Distribution Function (eCDF) Curves")
        grid = np.linspace(0, max(np.max(rcb_A), np.max(rcb_B), 4.5), 350)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Standard unweighted plot
        fA_unwt = marczyk_wecdf(rcb_A, scale=0.0, x0=0.0)(grid)
        fB_unwt = marczyk_wecdf(rcb_B, scale=0.0, x0=0.0)(grid)
        ax1.step(grid, fA_unwt, label='Treatment A', color='#1f77b4', lw=2)
        ax1.step(grid, fB_unwt, label='Treatment B', color='#ff7f0e', lw=2)
        ax1.fill_between(grid, fA_unwt, fB_unwt, step='pre', color='#2ca02c', alpha=0.25, label=f'TES = {unwt_tes:.3f}')
        ax1.set_title("Standard eCDF (Unweighted)", fontweight='bold')
        ax1.set_xlabel("RCB Continuous Value")
        ax1.set_ylabel("Empirical Cumulative Probability")
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='lower right')

        # Subtype weighted plot
        fA_wt = marczyk_wecdf(rcb_A, scale=cfg['scale'], x0=cfg['x0'])(grid)
        fB_wt = marczyk_wecdf(rcb_B, scale=cfg['scale'], x0=cfg['x0'])(grid)
        ax2.step(grid, fA_wt, label='Treatment A', color='#1f77b4', lw=2)
        ax2.step(grid, fB_wt, label='Treatment B', color='#ff7f0e', lw=2)
        ax2.fill_between(grid, fA_wt, fB_wt, step='pre', color='#2ca02c', alpha=0.25, label=f'Subtype TES = {wt_tes:.3f}')
        ax2.set_title(f"Subtype-Weighted eCDF ({selected_subtype})", fontweight='bold')
        ax2.set_xlabel("RCB Continuous Value")
        ax2.set_ylabel("Weighted Cumulative Probability")
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right')

        plt.tight_layout()
        st.pyplot(fig)
