# -*- coding: utf-8 -*-
import os
import time
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.local", override=True)

import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud: carrega secrets se disponíveis
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GDRIVE_FOLDER_ID" in st.secrets:
        os.environ["GDRIVE_FOLDER_ID"] = st.secrets["GDRIVE_FOLDER_ID"]
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
except Exception:
    pass

from drive_sync import sync_folder, download_index_from_drive
from ingest import build_index
from rag import answer

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"

# No Streamlit Cloud /mount/src é read-only — usa /tmp que é gravável
_on_cloud = Path("/mount/src").exists()
DB_DIR = "/tmp/chroma_db" if _on_cloud else str(DATA_DIR / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wounds_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")

st.set_page_config(page_title="Guia PMP- Amor a Pele", layout="wide")

st.markdown("""
<style>
  header {visibility: hidden;}
  [data-testid="stToolbar"] {visibility: hidden;}
  [data-testid="stHeader"] {display:none;}
  [data-testid="stDecoration"] {display:none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container { padding-top: 0.25rem !important; }
  .field-label {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    margin-top: 0.8rem;
    color: #1a1a2e;
  }
  .info-text {
    font-size: 1.0rem;
    line-height: 1.6;
    color: #222;
    margin: 0.3rem 0;
  }
</style>
""", unsafe_allow_html=True)


def img_b64(filename: str) -> str:
    p = BASE_DIR / "assets" / filename
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""




def secret_get(key: str, default: str = ""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)
def _ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _init_state() -> None:
    defaults = {
        "restore_status": "unknown",
        "vectordb_loaded": False,
        "vectordb": None,
        "load_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    st.session_state.load_log.append(f"[{stamp}] {msg}")
    print(f"[app] {msg}")


_ensure_data_dirs()
_init_state()


def ensure_local_index(force_download: bool = False) -> str:
    db_path = Path(DB_DIR)
    already_local = db_path.exists() and any(db_path.iterdir())
    if already_local and not force_download:
        status = "local"
        st.session_state.restore_status = status
        _log("Índice local já existe.")
        return status

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    if not folder_id:
        status = "no_folder_id"
        st.session_state.restore_status = status
        _log("GDRIVE_FOLDER_ID não configurado.")
        return status

    _log("Tentando restaurar índice do Google Drive...")
    t0 = time.perf_counter()
    ok = download_index_from_drive(DB_DIR, folder_id)
    dt = time.perf_counter() - t0
    status = "restored" if ok else "not_found"
    st.session_state.restore_status = status
    _log(f"Restauração concluída em {dt:.1f}s com status: {status}.")
    return status


@st.cache_resource(show_spinner=False)
def _load_vectordb_resource(db_dir: str, embed_model: str, collection_name: str, embed_device: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def load_vectordb_if_needed() -> bool:
    if st.session_state.vectordb is not None:
        return True

    db_path = Path(DB_DIR)
    if not db_path.exists() or not any(db_path.iterdir()):
        _log("Índice local ausente ao tentar carregar vector DB.")
        return False

    try:
        _log("Carregando embeddings/modelo vetorial do disco...")
        t0 = time.perf_counter()
        st.session_state.vectordb = _load_vectordb_resource(
            str(db_path),
            EMBED_MODEL,
            COLLECTION_NAME,
            os.getenv("EMBED_DEVICE", "cpu"),
        )
        dt = time.perf_counter() - t0
        st.session_state.vectordb_loaded = st.session_state.vectordb is not None
        _log(f"Vector DB carregado em {dt:.1f}s.")
        return st.session_state.vectordb_loaded
    except Exception as e:
        st.session_state.vectordb = None
        st.session_state.vectordb_loaded = False
        _log(f"Falha ao carregar índice local: {e}")
        return False




def build_augmented_query(user_text: str, mode: str, temperature: float) -> str:
    if temperature <= 0.25:
        style = "resposta mais objetiva, curta, organizada e direta"
    elif temperature <= 0.55:
        style = "resposta equilibrada, clara, didática e prática"
    else:
        style = "resposta mais explicativa, com mais contexto, exemplos e detalhamento"

    if mode == "Ensino (tutor)":
        mode_block = """
MODO: ENSINO (TUTOR)
Objetivo:
- Ensinar do básico ao clínico.
- Explicar o raciocínio passo a passo.
- Quando faltarem dados, apontar claramente o que ainda precisa saber.
Formato desejado:
1. Resposta curta inicial.
2. Explicação em passos.
3. Perguntas diagnósticas objetivas, se faltarem dados.
4. Mini-roteiro de estudo ou revisão.
5. Quando fizer sentido, exercício curto com gabarito comentado.
"""
    else:
        mode_block = """
MODO: CLÍNICO (OBJETIVO)
Objetivo:
- Responder de forma prática, segura e direta.
- Priorizar condutas, avaliação, sinais de alarme, critérios e próximos passos.
Formato desejado:
1. Resposta objetiva.
2. Checklist ou tópicos curtos.
3. Alertas e encaminhamento presencial quando necessário.
"""
    return f"""{mode_block}

ESTILO DA RESPOSTA:
- Adote {style}.
- Use prioritariamente os documentos recuperados.
- Se precisar complementar com conhecimento geral, deixe isso explícito.
- Não invente fonte.
- Quando citar base documental, mencione os documentos usados.

DÚVIDA / CONSULTA DO USUÁRIO:
{user_text}
""".strip()


def decide_sketch_with_groq(question: str, response_text: str, mode: str) -> dict:
    try:
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return {"need_sketch": False, "reason": "GROQ_API_KEY ausente.", "sketch_prompt": ""}
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=groq_key,
            temperature=0.1,
        )
        prompt = f"""
Você decide se um ESBOÇO/FIGURA didática ajudaria a compreender a resposta.
Responda SOMENTE em JSON válido, sem markdown, no formato:
{{"need_sketch": true/false, "reason": "...", "sketch_prompt": "..."}}

Regras:
- need_sketch = true quando desenho, fluxograma, anatomia simplificada, comparação visual, posicionamento, classificação, passo-a-passo ou esquema ajudarem muito.
- need_sketch = false quando a resposta já estiver suficientemente clara em texto.
- Se need_sketch = false, sketch_prompt deve ser "".
- Se need_sketch = true, crie um prompt curto, claro, didático e sem conteúdo chocante.
- Contexto do app: feridas crônicas, atendimento, ensino e consulta clínica.
- Modo atual: {mode}

PERGUNTA:
{question}

RESPOSTA:
{response_text[:2500]}
"""
        raw = llm.invoke(prompt).content.strip()
        import json, re
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                return {"need_sketch": False, "reason": "Não foi possível interpretar a decisão.", "sketch_prompt": ""}
            data = json.loads(m.group(0))
        return {
            "need_sketch": bool(data.get("need_sketch", False)),
            "reason": str(data.get("reason", "")).strip(),
            "sketch_prompt": str(data.get("sketch_prompt", "")).strip(),
        }
    except Exception as e:
        return {"need_sketch": False, "reason": f"Falha ao decidir esboço: {e}", "sketch_prompt": ""}

banner_b64 = img_b64("banner.png")
if banner_b64:
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:8px;">
      <img src="data:image/png;base64,{banner_b64}"
           style="width:40%; height:auto; border-radius:6px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🩹 Feridas Crônicas - PET G10 UFPel")

insta_b64 = img_b64("instagram.png")
enf_b64 = img_b64("logo.enfermagem.png")
insta_tag = f'<img src="data:image/png;base64,{insta_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if insta_b64 else "📷"
enf_tag = f'<img src="data:image/png;base64,{enf_b64}" width="22" style="vertical-align:middle; margin-right:4px;">' if enf_b64 else "🎓"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;
            margin-top:4px; margin-bottom:10px; font-size:0.95rem;">
  {insta_tag}
  <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
     style="text-decoration:none; font-weight:500;">Amor à Pele</a>
  <span style="color:#aaa;">|</span>
  <a href="https://www.instagram.com/g10petsaude/" target="_blank"
     style="text-decoration:none; font-weight:500;">PET G10</a>
  <span style="color:#aaa;">|</span>
  {enf_tag}
  <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
     style="text-decoration:none; font-weight:500;">Faculdade de Enfermagem – UFPel</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="info-text" style="margin-bottom:0.8rem;">
  Sistema de apoio ao <strong>Guia de Feridas Crônicas</strong> da Prefeitura Municipal de Pelotas,
  desenvolvido em parceria entre a Secretaria Municipal de Saúde e os projetos
  <strong>Amor à Pele</strong> e
  <a href="https://www.instagram.com/g10petsaude/" target="_blank">PET UFPel Saúde Digital</a> &mdash;
  Telemonitoramento de Feridas Crônicas, coordenados pela professora <a href="https://institucional.ufpel.edu.br/servidores/id/2858" target="_blank">Adrize Rutz Porto</a>,
  da Faculdade de Enfermagem &ndash; UFPel.
</p>
""", unsafe_allow_html=True)


def _html_com_imagens_embutidas(html_path: Path) -> str:
    import re
    html = html_path.read_text(encoding="utf-8", errors="replace")
    assets_dir = html_path.parent

    def substituir(match):
        src = match.group(1)
        if src.startswith("http"):
            return match.group(0)
        img_path = assets_dir / src
        if img_path.exists():
            mime = "image/png" if src.lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:{mime};base64,{b64}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', substituir, html)


st.markdown("""
<p class="info-text" style="margin-top:0.8rem;">
  <strong>Retrieval-Augmented Generation (RAG-AI)</strong> melhora a precisão dos modelos de linguagem (LLMs)
  ao recuperar dados externos, como documentos e bases de dados, para responder perguntas,
  reduzindo as chamadas alucinações.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  <strong>Embeddings</strong> são representações numéricas (vetores) de textos que capturam seu significado
  semântico, permitindo que a IA encontre informações relevantes por similaridade de sentido,
  e não apenas por palavras-chave.
</p>
<p class="info-text" style="margin-top:0.2rem;">
  Quer visualizar a representação gráfica do conhecimento?
</p>
""", unsafe_allow_html=True)

_emb_path = BASE_DIR / "assets" / "embeddings_guia.html"
if _emb_path.exists():
    with st.expander("📊 Ver Embeddings da Base de Conhecimento", expanded=False):
        _html_content = _html_com_imagens_embutidas(_emb_path)
        components.html(_html_content, height=700, scrolling=True)
else:
    st.caption("_(arquivo embeddings_guia.html não encontrado em assets/)_")

st.divider()

with st.sidebar:
    st.header("Base de conhecimento (Google Drive)")

    status = st.session_state.restore_status
    if status in ("local", "restored"):
        st.success("✅ Índice local disponível.")
    elif status == "not_found":
        st.error("⚠️ Nenhum índice encontrado no Drive.")
    elif status == "no_folder_id":
        st.error("⚠️ GDRIVE_FOLDER_ID não configurado nos secrets.")
    else:
        st.caption("Índice ainda não verificado nesta sessão.")

    if st.session_state.vectordb_loaded:
        st.success("✅ Vector DB carregado na sessão.")
    else:
        st.caption("Vector DB ainda não carregado na sessão.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Verificar/restaurar índice"):
            with st.spinner("Verificando/restaurando índice..."):
                ensure_local_index(force_download=False)
    with col_b:
        if st.button("Carregar índice"):
            with st.spinner("Carregando índice na sessão..."):
                if not (Path(DB_DIR).exists() and any(Path(DB_DIR).iterdir())):
                    ensure_local_index(force_download=False)
                ok = load_vectordb_if_needed()
                if ok:
                    st.success("Índice carregado na sessão.")
                else:
                    st.error("Não foi possível carregar o índice.")

    if os.getenv("DEBUG_APP", "0") == "1":
        st.caption(f"DB_DIR: {DB_DIR}")
        st.caption(f"RAW_DOCS_DIR: {RAW_DOCS_DIR}")
        st.caption(f"EMBED_MODEL: {EMBED_MODEL}")

    with st.expander("Logs da sessão", expanded=False):
        if st.session_state.load_log:
            for line in st.session_state.load_log[-20:]:
                st.code(line)
        else:
            st.caption("Sem logs ainda.")

    admin_pass = st.text_input("Senha admin", type="password")
    admin_password = secret_get("ADMIN_PASSWORD", "")
    is_admin = bool(admin_password) and (admin_pass == admin_password)

    if is_admin:
        folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
        st.text_input("Folder ID", value=folder_id, key="folder_id")

        if st.button("1) Sincronizar arquivos do Drive"):
            with st.spinner("Baixando..."):
                files = sync_folder(st.session_state.folder_id, str(RAW_DOCS_DIR))
            st.success(f"Baixados {len(files)} arquivos para {RAW_DOCS_DIR}")

        if st.button("2) Recriar índice (embeddings)"):
            with st.spinner("Indexando e salvando no Drive..."):
                t0 = time.perf_counter()
                n, vectordb, upload_ok = build_index(
                    str(RAW_DOCS_DIR),
                    DB_DIR,
                    gdrive_folder_id=st.session_state.folder_id,
                )
                dt = time.perf_counter() - t0
                if vectordb is not None:
                    _load_vectordb_resource.clear()
                    st.session_state.vectordb = vectordb
                    st.session_state.vectordb_loaded = True
                    st.session_state.restore_status = "local"
                    _log(f"Índice recriado em {dt:.1f}s com {n} trechos.")
                    db_check = Path(DB_DIR)
                    db_files = list(db_check.rglob("*")) if db_check.exists() else []
                    db_total = sum(f.stat().st_size for f in db_files if f.is_file())
                    st.info(f"🔍 DB_DIR={DB_DIR} | arquivos={len(db_files)} | tamanho={db_total//1024}KB")
                    st.success(f"✅ Índice criado com {n} trechos e salvo no Drive.")
                else:
                    _log("Nenhum documento encontrado para indexação.")
                    st.error(f"Nenhum documento encontrado em {RAW_DOCS_DIR}. Sincronize primeiro.")

        if Path(DB_DIR).exists():
            st.info("✅ Índice disponível em disco.")
        else:
            st.warning("⚠️ Índice não encontrado. Execute os passos 1 e 2 acima.")
    else:
        if admin_pass:
            st.error("Senha incorreta")


def mic_component(target_label: str, field_index: int):
    key = f"mic_{field_index}"
    components.html(f"""
    <style>
      .mic-row {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
      .mic-btn {{
        background:#0066cc; color:white; border:none;
        width:36px; height:36px; border-radius:50%;
        font-size:1.1rem; cursor:pointer; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        transition:background .2s;
      }}
      .mic-btn.listening {{ background:#e53935; animation:pulse 1s infinite; }}
      @keyframes pulse {{
        0%,100%{{box-shadow:0 0 0 0 rgba(229,57,53,.35);}}
        50%{{box-shadow:0 0 0 8px rgba(229,57,53,0);}}
      }}
      #micStatus_{key} {{ font-size:.80rem; color:#555; }}
    </style>
    <div class="mic-row">
      <button class="mic-btn" id="micBtn_{key}" title="Falar">🎤</button>
      <span id="micStatus_{key}">Toque em 🎤 para falar (Chrome/Edge)</span>
    </div>
    <script>
    (function(){{
      const btn    = document.getElementById('micBtn_{key}');
      const status = document.getElementById('micStatus_{key}');
      let listening = false;

      function fillTextarea(text) {{
        try {{
          const doc = window.parent.document;
          const allTa = doc.querySelectorAll('[data-testid="stTextArea"] textarea');
          const ta = allTa[{field_index}] || doc.querySelector('textarea[aria-label="{target_label}"]');
          if (!ta) {{ status.textContent = '⚠️ Campo não encontrado.'; return; }}
          const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
          setter.call(ta, text);
          ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
          ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
          ta.focus();
          status.textContent = '✅ Texto inserido — edite se quiser.';
        }} catch(e) {{
          status.textContent = '⚠️ Erro: ' + e.message;
        }}
      }}

      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {{
        status.textContent = '⚠️ Voz indisponível — use Chrome ou Edge.';
        btn.style.opacity = '.4';
        return;
      }}
      const rec = new SR();
      rec.lang = 'pt-BR';
      rec.continuous = false;
      rec.interimResults = true;

      rec.onstart = () => {{
        listening = true;
        btn.classList.add('listening');
        btn.innerHTML = '🔴';
        status.textContent = '🎙️ Ouvindo…';
      }};
      rec.onresult = (e) => {{
        let interim = '', final = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {{
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) final += t; else interim += t;
        }}
        if (final) fillTextarea(final.trim());
        else status.textContent = '…' + interim;
      }};
      rec.onerror = (e) => {{
        status.textContent = '❌ Erro: ' + e.error;
        btn.classList.remove('listening'); btn.innerHTML = '🎤'; listening = false;
      }};
      rec.onend = () => {{
        listening = false; btn.classList.remove('listening'); btn.innerHTML = '🎤';
      }};
      btn.addEventListener('click', () => {{ if (listening) rec.stop(); else rec.start(); }});
    }})();
    </script>
    """, height=48, scrolling=False)



col_cfg1, col_cfg2, col_cfg3 = st.columns([1.3, 1.2, 1.2])

with col_cfg1:
    mode = st.radio("Modo", ["Ensino (tutor)", "Clínico (objetivo)"], horizontal=True)

with col_cfg2:
    temperature = st.slider(
        "Estilo da resposta",
        0.0,
        1.0,
        0.25 if mode == "Ensino (tutor)" else 0.20,
        0.05,
    )
    st.caption("Mais baixo = mais direto • Mais alto = mais explicativo")

with col_cfg3:
    auto_sketch = st.checkbox(
        "Sugerir esboço quando fizer sentido",
        value=True,
        help="Usa o Groq para decidir se um esquema/figura ajudaria e, se sim, gera um prompt.",
    )

st.markdown('<p class="field-label">❓ Dúvida / consulta</p>', unsafe_allow_html=True)
mic_component("Dúvida / consulta", 0)
user_query = st.text_area(
    "Dúvida / consulta",
    height=220,
    placeholder=(
        "Escreva aqui sua dúvida clínica ou de estudo.\n"
        "Ex.: Como diferenciar lesão por pressão de úlcera arterial?\n"
        "Ex.: No pé diabético, quais sinais exigem avaliação presencial urgente?\n"
        "Ex.: Explique TIME/TIMERS do básico ao uso prático."
    ),
    label_visibility="collapsed",
)

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    gerar = st.button("🚀 Tirar dúvida", type="primary")
with col_btn2:
    if st.button("🗑️ Limpar cache"):
        _load_vectordb_resource.clear()
        st.session_state.vectordb = None
        st.session_state.vectordb_loaded = False
        st.success("Cache do índice limpo!")

if gerar:
    if not user_query.strip():
        st.error("Escreva sua dúvida primeiro.")
    else:
        with st.spinner("Buscando nos documentos e gerando resposta..."):
            if st.session_state.vectordb is None:
                ensure_local_index(force_download=False)
                load_vectordb_if_needed()

            query_for_rag = build_augmented_query(user_query, mode, temperature)
            resp, hits = answer(query_for_rag, "Não informado.", k=4, vectordb=st.session_state.vectordb)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Resposta")
            st.markdown(resp)

            if auto_sketch:
                with st.spinner("Avaliando se um esboço ajudaria..."):
                    sketch = decide_sketch_with_groq(user_query, resp, mode)

                if sketch.get("need_sketch"):
                    st.markdown("### ✍️ Sugestão de esboço")
                    if sketch.get("reason"):
                        st.info(sketch["reason"])
                    sketch_prompt = (sketch.get("sketch_prompt") or "").strip()
                    if sketch_prompt:
                        st.text_area(
                            "Prompt do esboço (copie e cole no gerador de imagens)",
                            value=sketch_prompt,
                            height=180,
                        )
                        st.download_button(
                            "💾 Baixar prompt do esboço (.txt)",
                            data=sketch_prompt.encode("utf-8"),
                            file_name="prompt_esboco_feridas.txt",
                            mime="text/plain; charset=utf-8",
                        )

        with col2:
            st.markdown("### Fontes recuperadas")
            if hits:
                shown = set()
                for d in hits:
                    src = Path(d.metadata.get("source", "arquivo_desconhecido")).name
                    page = d.metadata.get("page", "trecho")
                    key = f"{src}-{page}"
                    if key in shown:
                        continue
                    shown.add(key)
                    st.write(f"- {src} | p. {page}")
            else:
                st.info("Nenhuma fonte recuperada.")

                st.info("Nenhuma fonte indexada utilizada — resposta baseada em conhecimento geral.")
