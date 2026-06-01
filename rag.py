"""
Módulo de Recuperação e Resposta (RAG) para o projeto PET-Saúde G10.
Focado em feridas crônicas: consulta, estudo e apoio à decisão com base em
documentos indexados (guias, protocolos e casos), com complemento explícito
de conhecimento geral quando necessário.

Versão ajustada:
- remove referências a dieta/nutrição;
- mantém imports pesados sob demanda;
- prioriza documentos indexados;
- usa Groq para gerar respostas em português.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

try:
    import streamlit as st
    try:
        groq_secret = st.secrets.get("GROQ_API_KEY", "")
        if groq_secret:
            os.environ["GROQ_API_KEY"] = groq_secret
    except Exception:
        pass
except Exception:
    st = None  # type: ignore


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_DIR = Path(__file__).resolve().parent
_on_cloud = Path("/mount/src").exists()
DB_DIR_DEFAULT = "/tmp/chroma_db" if _on_cloud else str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wounds_knowledge")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")

SYSTEM_RULES = """Você é um assistente clínico-educacional do projeto PET-Saúde G10,
especialista em feridas crônicas, avaliação clínica, prevenção, cuidado,
classificação, sinais de gravidade, condutas iniciais e educação em saúde.

MODO DE OPERAÇÃO HÍBRIDO:
1. PRIORIDADE MÁXIMA: use os trechos do CONTEXTO DOS DOCUMENTOS como base principal da resposta.
   - Sempre cite as fontes no formato [Nome do Arquivo | Página] quando usar o contexto.
2. COMPLEMENTAÇÃO: se o contexto não cobrir completamente a dúvida, complemente com conhecimento geral.
   - Nesse caso, indique claramente: "(conhecimento geral - não consta nos documentos indexados)".
3. TRANSPARÊNCIA: deixe claro o que veio dos documentos e o que veio do conhecimento geral.
4. NUNCA invente fontes nem cite documentos que não estejam no contexto fornecido.
5. Organize a resposta de forma clara, prática e didática.
6. Se a situação sugerir gravidade, oriente avaliação presencial/encaminhamento apropriado.
7. Não prescreva condutas fora do escopo das informações fornecidas como se fossem ordens médicas definitivas.
"""


def _import_hf_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings


def _import_chroma():
    from langchain_chroma import Chroma
    return Chroma


def _import_chatgroq():
    from langchain_groq import ChatGroq
    return ChatGroq


@functools.lru_cache(maxsize=1)
def _get_embeddings() -> Any:
    HuggingFaceEmbeddings = _import_hf_embeddings()
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


@functools.lru_cache(maxsize=4)
def _get_db_from_disk(db_dir: str = DB_DIR_DEFAULT) -> Any:
    Chroma = _import_chroma()
    return Chroma(
        persist_directory=db_dir,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def answer(
    question: str,
    patient_summary: str = "Não informado.",
    k: int = 5,
    vectordb: Optional[Any] = None,
) -> Tuple[str, List[Any]]:
    """
    Busca nos documentos e gera resposta híbrida via Groq.

    Parâmetros:
      question       – dúvida/consulta do usuário
      patient_summary – resumo opcional (mantido por compatibilidade)
      k              – número de documentos/trechos recuperados
      vectordb       – instância Chroma já criada (session_state). Se None,
                       tenta carregar do disco.

    Retorna:
      (texto_da_resposta, lista_de_documentos_encontrados)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrada. Configure em .streamlit/secrets.toml "
            "ou nas variáveis do ambiente."
        )

    if vectordb is not None:
        db = vectordb
    else:
        db_path = Path(DB_DIR_DEFAULT)
        if not db_path.exists() or not any(db_path.iterdir()):
            return (
                "⚠️ Índice não encontrado. Na Streamlit Cloud, restaure ou recrie o índice "
                "antes de gerar resposta.",
                [],
            )
        db = _get_db_from_disk()

    full_query = f"Resumo clínico: {patient_summary}\nDúvida/consulta: {question}"
    hits_with_score = db.similarity_search_with_relevance_scores(full_query, k=k)

    hits = [doc for doc, score in hits_with_score if score >= RELEVANCE_THRESHOLD]
    all_hits = [doc for doc, _ in hits_with_score]

    if hits_with_score:
        context_parts = []
        for doc, score in hits_with_score:
            source = os.path.basename(doc.metadata.get("source", "desconhecido"))
            page = doc.metadata.get("page", "-")
            content = doc.page_content.strip()
            relevance = "alta" if score >= RELEVANCE_THRESHOLD else "baixa"
            context_parts.append(
                f"FONTE: {source} (p. {page}) [relevância: {relevance}]\n"
                f"CONTEÚDO: {content}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        context_note = "" if hits else (
            "\n⚠️ NOTA: Os documentos indexados têm baixa relevância para esta consulta. "
            "Use conhecimento geral apenas para complementar, deixando isso explícito."
        )
    else:
        context_text = "Nenhum documento relevante encontrado no índice."
        context_note = (
            "\n⚠️ NOTA: Não há documentos indexados para esta consulta. "
            "Responda com conhecimento geral e deixe isso explícito."
        )

    ChatGroq = _import_chatgroq()
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=groq_key,
        temperature=0.2,
    )

    user_message = f"""RESUMO CLÍNICO:
{patient_summary}

DÚVIDA / CONSULTA:
{question}

CONTEXTO DOS DOCUMENTOS:
{context_text}
{context_note}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_message},
    ])

    return response.content, all_hits
