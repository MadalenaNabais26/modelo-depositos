import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Configuração geral
# =========================
st.set_page_config(
    page_title="Previsão de Depósitos",
    page_icon="🧪",
    layout="centered"
)

MODEL_FILE = "modelo_depositos_logit.joblib"
HIST_FILE = "historico_previsoes_depositos.csv"
TRAIN_DATA_FILE = "dados_treino.csv"

COL_DUREZA_TREINO = "mg CaCO3/L"
COL_PH_TREINO = "pH_dia_evento"

LIMIAR_BAIXO_DEFAULT = 0.33
LIMIAR_ALTO_DEFAULT = 0.66

# =========================
# Funções auxiliares
# =========================
@st.cache_resource
def carregar_modelo(path: str):
    bundle = joblib.load(path)
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)
    return model, threshold

@st.cache_data
def carregar_limites_reais(train_file: str):
    if not os.path.exists(train_file):
        return None

    df = pd.read_csv(train_file)

    if COL_DUREZA_TREINO not in df.columns or COL_PH_TREINO not in df.columns:
        return None

    df2 = df[[COL_DUREZA_TREINO, COL_PH_TREINO]].dropna()
    if df2.empty:
        return None

    return {
        "dureza_min": float(df2[COL_DUREZA_TREINO].min()),
        "dureza_max": float(df2[COL_DUREZA_TREINO].max()),
        "ph_min": float(df2[COL_PH_TREINO].min()),
        "ph_max": float(df2[COL_PH_TREINO].max())
    }

def classificar_risco(prob: float, limiar_baixo: float, limiar_alto: float) -> str:
    if prob < limiar_baixo:
        return "Baixo"
    if prob < limiar_alto:
        return "Médio"
    return "Alto"

def guardar_historico(linha: dict):
    df_new = pd.DataFrame([linha])
    if os.path.exists(HIST_FILE):
        df_old = pd.read_csv(HIST_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(HIST_FILE, index=False)

def ler_historico():
    if os.path.exists(HIST_FILE):
        return pd.read_csv(HIST_FILE)
    return pd.DataFrame(columns=[
        "data_hora", "dureza_mg_caco3_l", "ph_evento", "probabilidade", "risco",
        "operador", "observacoes", "fora_intervalo_treino"
    ])

def prever_prob(model, dureza: float, ph: float) -> float:
    X_new = np.array([[float(dureza), float(ph)]], dtype=float)
    return float(model.predict_proba(X_new)[0, 1])

def fora_intervalo(valor: float, vmin: float, vmax: float) -> bool:
    return (valor < vmin) or (valor > vmax)

def plot_mapa_risco_com_contornos(
    model,
    dureza_min, dureza_max,
    ph_min, ph_max,
    n_grid,
    dureza_atual, ph_atual,
    limiar_baixo, limiar_alto,
    rect_treino=None
):
    durezas = np.linspace(dureza_min, dureza_max, n_grid)
    phs = np.linspace(ph_min, ph_max, n_grid)
    D, P = np.meshgrid(durezas, phs)

    grid = np.c_[D.ravel(), P.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(D.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(
        probs,
        origin="lower",
        aspect="auto",
        extent=[dureza_min, dureza_max, ph_min, ph_max]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probabilidade prevista")

    levels = sorted({float(limiar_baixo), float(limiar_alto)})
    cs = ax.contour(D, P, probs, levels=levels)
    ax.clabel(cs, inline=True, fontsize=9, fmt=lambda v: f"{v:.2f}")

    if rect_treino is not None:
        ax.add_patch(rect_treino)

    ax.set_xlabel("Dureza da água (mg CaCO₃/L)")
    ax.set_ylabel("pH no dia do evento")
    ax.set_title("Mapa de risco previsto (dureza × pH)")

    ax.scatter([dureza_atual], [ph_atual], s=90, marker="x")
    ax.text(dureza_atual, ph_atual, "  valor atual", va="center")

    fig.tight_layout()
    return fig

# =========================
# Interface
# =========================
st.title("🧪 Previsão do risco de formação de depósitos")
st.write(
    "Ferramenta para estimar o risco de formação de depósitos "
    "(proxy: recuperação prolongada do pH) a partir de medições de "
    "**dureza da água** e **pH no dia do evento**."
)

if not os.path.exists(MODEL_FILE):
    st.error(f"Não foi encontrado o ficheiro do modelo: '{MODEL_FILE}'.")
    st.stop()

model, threshold = carregar_modelo(MODEL_FILE)
limites = carregar_limites_reais(TRAIN_DATA_FILE)

with st.sidebar:
    st.header("Definições")
    st.caption("Ajustes de visualização.")

    limiar_baixo = st.slider("Limiar risco baixo → médio", 0.05, 0.60, float(LIMIAR_BAIXO_DEFAULT), 0.01)
    limiar_alto = st.slider("Limiar risco médio → alto", 0.40, 0.95, float(LIMIAR_ALTO_DEFAULT), 0.01)

    st.divider()
    st.subheader("Mapa de risco")
    n_grid = st.slider("Resolução do mapa", 25, 160, 90, 5)

    st.caption(f"Limiar binário interno do modelo: **{threshold:.2f}**")

st.subheader("1) Inserir medições")

col1, col2 = st.columns(2)
with col1:
    dureza = st.number_input("Dureza da água (mg CaCO₃/L)", min_value=0.0, value=200.0, step=1.0)
with col2:
    ph_evento = st.number_input("pH no dia do evento", min_value=0.0, max_value=14.0, value=5.5, step=0.1)

fora = False
if limites is not None:
    fora_d = fora_intervalo(float(dureza), limites["dureza_min"], limites["dureza_max"])
    fora_p = fora_intervalo(float(ph_evento), limites["ph_min"], limites["ph_max"])
    fora = fora_d or fora_p

    if fora:
        msgs = []
        if fora_d:
            msgs.append(f"dureza fora do intervalo [{limites['dureza_min']:.1f}, {limites['dureza_max']:.1f}]")
        if fora_p:
            msgs.append(f"pH fora do intervalo [{limites['ph_min']:.2f}, {limites['ph_max']:.2f}]")
        st.warning(
            "Atenção: " + " e ".join(msgs) +
            ". A previsão corresponde a extrapolação e pode ser menos fiável."
        )

with st.expander("Campos opcionais (para histórico)"):
    operador = st.text_input("Operador / Turno (opcional)", value="")
    observacoes = st.text_area("Observações (opcional)", value="", height=80)

st.subheader("2) Prever")

btn = st.button("Calcular previsão", type="primary")

if btn:
    prob = prever_prob(model, dureza, ph_evento)
    risco = classificar_risco(prob, limiar_baixo, limiar_alto)

    st.success("Previsão calculada com sucesso.")
    st.metric("Probabilidade prevista", f"{prob:.3f}")

    if risco == "Baixo":
        st.info("🟢 Risco baixo de formação de depósitos.")
    elif risco == "Médio":
        st.warning("🟡 Risco médio de formação de depósitos.")
    else:
        st.error("🔴 Risco elevado de formação de depósitos.")

    linha = {
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dureza_mg_caco3_l": float(dureza),
        "ph_evento": float(ph_evento),
        "probabilidade": prob,
        "risco": risco,
        "operador": operador,
        "observacoes": observacoes,
        "fora_intervalo_treino": bool(fora)
    }
    guardar_historico(linha)
    st.caption("Registo guardado no histórico.")

st.subheader("3) Mapa de risco")

if limites is None:
    dureza_min = max(0.0, float(dureza) - 150)
    dureza_max = float(dureza) + 150
    ph_min = max(0.0, float(ph_evento) - 2.0)
    ph_max = min(14.0, float(ph_evento) + 2.0)
    rect = None
else:
    margem_d = 0.05 * (limites["dureza_max"] - limites["dureza_min"] if limites["dureza_max"] > limites["dureza_min"] else 1.0)
    margem_p = 0.05 * (limites["ph_max"] - limites["ph_min"] if limites["ph_max"] > limites["ph_min"] else 1.0)

    dureza_min = max(0.0, limites["dureza_min"] - margem_d)
    dureza_max = limites["dureza_max"] + margem_d
    ph_min = max(0.0, limites["ph_min"] - margem_p)
    ph_max = min(14.0, limites["ph_max"] + margem_p)

    rect = Rectangle(
        (limites["dureza_min"], limites["ph_min"]),
        (limites["dureza_max"] - limites["dureza_min"]),
        (limites["ph_max"] - limites["ph_min"]),
        fill=False,
        linewidth=2
    )

fig = plot_mapa_risco_com_contornos(
    model=model,
    dureza_min=dureza_min, dureza_max=dureza_max,
    ph_min=ph_min, ph_max=ph_max,
    n_grid=n_grid,
    dureza_atual=float(dureza),
    ph_atual=float(ph_evento),
    limiar_baixo=float(limiar_baixo),
    limiar_alto=float(limiar_alto),
    rect_treino=rect
)
st.pyplot(fig)

st.subheader("4) Histórico")

hist = ler_historico()
st.dataframe(hist, use_container_width=True)

csv_bytes = hist.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descarregar histórico (CSV)",
    data=csv_bytes,
    file_name="historico_previsoes_depositos.csv",
    mime="text/csv"
)

st.caption("O histórico é guardado localmente em ficheiro CSV.")
