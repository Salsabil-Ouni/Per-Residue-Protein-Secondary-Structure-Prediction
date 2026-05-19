import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Protein SS Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── Vocabulary ────────────────────────────────────────────────────────────────
AA_VOCAB   = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX  = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
IDX_TO_SS  = {0: "H", 1: "E", 2: "C"}
VOCAB_SIZE = len(AA_VOCAB) + 1
NUM_CLASSES = 3

SS_COLOR = {"H": "#e53935", "E": "#1e88e5", "C": "#757575"}
SS_NAME  = {"H": "Alpha Helix", "E": "Beta Sheet", "C": "Coil / Loop"}

# ── Model definition (must match training) ────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64,
                 hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.clf  = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out = self.drop(self.emb(x))
        out, _ = self.lstm(out)
        return self.clf(self.drop(out))

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = BiLSTM()
    ckpt  = torch.load(
        r"C:\Users\salsa\protein_ss_project\bilstm_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(seq, model):
    seq = seq.upper().strip()
    x   = torch.tensor([AA_TO_IDX.get(aa, 0) for aa in seq],
                       dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    probs  = torch.softmax(logits.squeeze(0), dim=-1).numpy()
    labels = [IDX_TO_SS[int(i)] for i in probs.argmax(-1)]
    return labels, probs

# ── Sequence figure ───────────────────────────────────────────────────────────
def draw_sequence(seq, labels, cols_per_row=50):
    n      = len(seq)
    n_rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(cols_per_row * 0.32, n_rows * 0.75 + 0.4))
    if n_rows == 1:
        axes = [axes]
    for row in range(n_rows):
        ax = axes[row]
        s  = row * cols_per_row
        e  = min(s + cols_per_row, n)
        ax.set_xlim(0, cols_per_row)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(-0.8, 0.45, str(s + 1), fontsize=7,
                color="#999", va="center", ha="right")
        for i, (aa, ss) in enumerate(zip(seq[s:e], labels[s:e])):
            rect = mpatches.FancyBboxPatch(
                (i + 0.06, 0.08), 0.84, 0.78,
                boxstyle="round,pad=0.04",
                facecolor=SS_COLOR[ss], edgecolor="white", linewidth=0.6,
            )
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.49, aa, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
    plt.tight_layout(pad=0.2)
    return fig

# ── Confidence bar chart ──────────────────────────────────────────────────────
def draw_confidence(probs):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    positions = range(len(probs))
    ax.stackplot(positions,
                 probs[:, 0], probs[:, 1], probs[:, 2],
                 colors=["#e53935", "#1e88e5", "#757575"],
                 labels=["Helix", "Sheet", "Coil"], alpha=0.85)
    ax.set_xlim(0, len(probs) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Residue position", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=10)
    ax.set_title("Per-residue prediction confidence", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧬 Protein Secondary Structure Predictor")
st.caption("BiLSTM model trained on synthetic sequences · 78.23% per-residue accuracy")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts the **secondary structure** of a protein sequence using a
    bidirectional LSTM trained in PyTorch.

    **Labels**
    - 🔴 **H** — Alpha Helix
    - 🔵 **E** — Beta Sheet
    - ⬜ **C** — Coil / Loop

    **Model** BiLSTM · 2 layers · hidden 128 · 310K params
    """)

    st.header("Example sequences")
    examples = {
        "Human TP53 (first 60)": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP",
        "Short helix-rich":      "AELAAELAAELAAELAAELAAELAAELA",
        "Short sheet-rich":      "VIFYVIFYVIFYVIFYVIFYVIVY",
        "Mixed":                 "MEEPQSDAELAAELKLLPENNVIFYVIYVKLLPENNVIFYLSPD",
    }
    for name, seq in examples.items():
        if st.button(name, use_container_width=True):
            st.session_state["input_seq"] = seq

# Main input
default_seq = st.session_state.get("input_seq", "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP")
seq_input = st.text_area(
    "Enter amino acid sequence (single-letter codes, spaces/newlines ignored):",
    value=default_seq,
    height=100,
)

run = st.button("Predict", type="primary", use_container_width=True)

if run or seq_input:
    # Clean input
    seq = "".join(c for c in seq_input.upper() if c in AA_TO_IDX)

    if len(seq) < 4:
        st.warning("Sequence too short — enter at least 4 valid amino acid letters.")
        st.stop()

    invalid = [c for c in seq_input.upper() if c.isalpha() and c not in AA_TO_IDX]
    if invalid:
        st.info(f"Ignored unrecognized characters: {', '.join(set(invalid))}")

    # Load and run
    with st.spinner("Running BiLSTM…"):
        model  = load_model()
        labels, probs = predict(seq, model)

    # ── Stats row ─────────────────────────────────────────────────────────────
    h_pct = labels.count("H") / len(labels) * 100
    e_pct = labels.count("E") / len(labels) * 100
    c_pct = labels.count("C") / len(labels) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Residues", len(seq))
    col2.metric("Alpha Helix", f"{h_pct:.1f}%")
    col3.metric("Beta Sheet",  f"{e_pct:.1f}%")
    col4.metric("Coil / Loop", f"{c_pct:.1f}%")

    st.divider()

    # ── Colored sequence ──────────────────────────────────────────────────────
    st.subheader("Predicted secondary structure")

    legend_html = " &nbsp; ".join(
        f'<span style="background:{SS_COLOR[ss]};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold">{ss} = {SS_NAME[ss]}</span>'
        for ss in ["H", "E", "C"]
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st.pyplot(draw_sequence(seq, labels), use_container_width=True)

    st.divider()

    # ── Confidence plot ───────────────────────────────────────────────────────
    st.subheader("Prediction confidence per residue")
    st.pyplot(draw_confidence(probs), use_container_width=True)

    st.divider()

    # ── Raw label string ──────────────────────────────────────────────────────
    st.subheader("Label string")
    label_str = "".join(labels)
    st.code(f"Sequence: {seq}\nLabels:   {label_str}", language=None)

    # Download
    result_txt = f">prediction\n{seq}\n{label_str}\n"
    st.download_button(
        "Download result (.txt)",
        data=result_txt,
        file_name="prediction.txt",
        mime="text/plain",
    )

# ── TP53 3D structure section (always shown) ──────────────────────────────────
st.divider()
st.subheader("🧬 Human TP53 — Interactive 3D Structure")
st.caption("UniProt P04637 · 393 residues · AlphaFold v6 · Drag to rotate · Scroll to zoom")

@st.cache_data
def load_tp53_views():
    import py3Dmol

    PDB = r"C:\Users\salsa\protein_ss_project\tp53_alphafold.pdb"
    with open(PDB) as f:
        pdb_text = f.read()

    # Parse residues + pLDDT
    aa3 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
           "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
           "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"}
    seen, residues, plddt_map = set(), [], {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            resi = int(line[22:26].strip())
            resn = line[17:20].strip()
            bfac = float(line[60:66].strip())
            if resi not in seen:
                seen.add(resi)
                residues.append((resi, aa3.get(resn, "X")))
                plddt_map[resi] = bfac
    sequence = "".join(aa for _, aa in residues)

    # Run BiLSTM on TP53
    model = load_model()
    x = torch.tensor([AA_TO_IDX.get(aa, 0) for aa in sequence],
                     dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    pred_labels = [IDX_TO_SS[int(i)] for i in logits.squeeze(0).argmax(-1).numpy()]

    def plddt_hex(s):
        if s >= 90: return "0x0053D6"
        if s >= 70: return "0x65CBF3"
        if s >= 50: return "0xFFDB13"
        return "0xFF7D45"

    SS_HEX = {"H": "0xff3333", "E": "0x3399ff", "C": "0x888888"}

    # View 1 — pLDDT
    v1 = py3Dmol.view(width=860, height=500)
    v1.addModel(pdb_text, "pdb")
    v1.setStyle({}, {"cartoon": {"color": "0x888888"}})
    for resi, _ in residues:
        v1.addStyle({"resi": resi}, {"cartoon": {"color": plddt_hex(plddt_map[resi])}})
    v1.setBackgroundColor("0x0f0f1a")
    v1.zoomTo()
    html1 = v1.write_html()

    # View 2 — BiLSTM
    v2 = py3Dmol.view(width=860, height=500)
    v2.addModel(pdb_text, "pdb")
    v2.setStyle({}, {"cartoon": {"color": SS_HEX["C"]}})
    for (resi, _), ss in zip(residues, pred_labels):
        if ss != "C":
            v2.addStyle({"resi": resi}, {"cartoon": {"color": SS_HEX[ss]}})
    v2.setBackgroundColor("0x0f0f1a")
    v2.zoomTo()
    html2 = v2.write_html()

    h_pct = pred_labels.count("H") / len(pred_labels) * 100
    e_pct = pred_labels.count("E") / len(pred_labels) * 100
    c_pct = pred_labels.count("C") / len(pred_labels) * 100

    return html1, html2, h_pct, e_pct, c_pct

try:
    html1, html2, h_pct, e_pct, c_pct = load_tp53_views()

    tab1, tab2 = st.tabs(["View 1 — AlphaFold Confidence (pLDDT)", "View 2 — BiLSTM Prediction"])

    with tab1:
        st.markdown(
            '<span style="background:#0053d6;color:white;padding:3px 10px;border-radius:4px;margin-right:6px">Very high &gt;90</span>'
            '<span style="background:#65cbf3;color:white;padding:3px 10px;border-radius:4px;margin-right:6px">Confident 70–90</span>'
            '<span style="background:#ffdb13;color:#333;padding:3px 10px;border-radius:4px;margin-right:6px">Low 50–70</span>'
            '<span style="background:#ff7d45;color:white;padding:3px 10px;border-radius:4px">Disordered &lt;50</span>',
            unsafe_allow_html=True,
        )
        components.html(html1, height=520, scrolling=False)

    with tab2:
        st.markdown(
            f'<span style="background:#ff3333;color:white;padding:3px 10px;border-radius:4px;margin-right:6px">Helix {h_pct:.1f}%</span>'
            f'<span style="background:#3399ff;color:white;padding:3px 10px;border-radius:4px;margin-right:6px">Sheet {e_pct:.1f}%</span>'
            f'<span style="background:#888;color:white;padding:3px 10px;border-radius:4px">Coil {c_pct:.1f}%</span>',
            unsafe_allow_html=True,
        )
        components.html(html2, height=520, scrolling=False)

except Exception as e:
    st.error(f"Could not load 3D structure: {e}")
