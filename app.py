import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf

# Configuração da página do Streamlit (Deve ser a primeira chamada do Streamlit)
st.set_page_config(
    page_title="Mir - Suporte Técnico Virtual (Ferroviário)",
    page_icon="🚆",
    layout="wide"
)

# --- Configuração de Pastas ---
PASTA_BASE_MANUAIS = Path("./Docs")
PASTA_BASE_MANUAIS.mkdir(exist_ok=True)

# Inicialização dos componentes de IA (com cache para otimizar o carregamento)
@st.cache_resource
def carregar_componentes_ia():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./db_docs")
    collection = chroma_client.get_or_create_collection(name="mir_suporte_tecnico")
    return encoder, chroma_client, collection

encoder, chroma_client, collection = carregar_componentes_ia()

# Configuração da API do OpenRouter
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Mir Web"}
)

def indexar_arquivo(caminho_arquivo):
    """Indexa um arquivo PDF extraindo seu texto e salvando na collection."""
    try:
        nome_arq = caminho_arquivo.name
        
        # Remove entradas antigas para evitar duplicidade
        existing = collection.get(where={"source": nome_arq})
        if existing and len(existing.get('ids', [])) > 0:
            collection.delete(where={"source": nome_arq})
        
        reader = pypdf.PdfReader(str(caminho_arquivo))
        total_paginas = len(reader.pages)

        if total_paginas == 0:
            return False

        for i, page in enumerate(reader.pages):
            texto = page.extract_text()
            if texto and texto.strip():
                page_id = f"{nome_arq}_p{i}"
                embedding = encoder.encode([texto]).tolist()
                collection.upsert(
                    ids=[page_id],
                    embeddings=embedding,
                    documents=[texto],
                    metadatas=[{"source": nome_arq, "page": i}]
                )
        return True
    except Exception as e:
        print(f"Erro ao indexar {caminho_arquivo}: {e}")
        return False

# --- GESTÃO DE SESSÕES E HISTÓRICO DE CHATS ---
if "chats_duvidas" not in st.session_state:
    st.session_state.chats_duvidas = [{
        "id": 1, 
        "titulo": "Nova Dúvida", 
        "mensagens": [{"role": "assistant", "content": "Olá. Use este canal para retirar dúvidas gerais, realizar consultas técnicas ou verificar especificações priorizando os manuais indexados."}]
    }]
if "chat_atual_duvidas" not in st.session_state:
    st.session_state.chat_atual_duvidas = 0

if "chats_falhas" not in st.session_state:
    st.session_state.chats_falhas = [{
        "id": 1, 
        "titulo": "Nova Falha", 
        "mensagens": [{"role": "assistant", "content": "Olá. Sou o agente especialista em manutenção ferroviária. Estou pronto para auxiliar técnicos, mecânicos, eletricistas, inspetores e engenheiros. Descreva a falha ou anomalia observada."}]
    }]
if "chat_atual_falhas" not in st.session_state:
    st.session_state.chat_atual_falhas = 0

# --- SIDEBAR MODERNA ---
with st.sidebar:
    st.markdown("## 🚆 MIR DESKTOP")
    st.caption("Inteligência em Manutenção")
    st.divider()

    aba_selecionada = st.radio(
        "Navegação",
        ["📖 Dúvidas Técnicas", "⚙️ Análise de Falhas", "📂 Adicionar Conhecimento"],
        label_visibility="collapsed"
    )

    st.divider()
    
    # Histórico de conversas baseado na aba ativa
    if aba_selecionada == "📖 Dúvidas Técnicas":
        st.markdown("### 🕒 Histórico de Dúvidas")
        if st.button("➕ Novo Chat de Dúvidas", use_container_width=True):
            novo_id = len(st.session_state.chats_duvidas) + 1
            st.session_state.chats_duvidas.append({
                "id": novo_id, 
                "titulo": f"Dúvida {novo_id}", 
                "mensagens": [{"role": "assistant", "content": "Olá. Qual consulta deseja realizar?"}]
            })
            st.session_state.chat_atual_duvidas = len(st.session_state.chats_duvidas) - 1
            st.rerun()

        for i, chat in enumerate(st.session_state.chats_duvidas[-10:]):
            col_h1, col_h2 = st.columns([0.8, 0.2])
            if col_h1.button(chat["titulo"], key=f"btn_duvida_{i}", use_container_width=True):
                st.session_state.chat_atual_duvidas = i
                st.rerun()
            if col_h2.button("🗑️", key=f"del_duvida_{i}"):
                if len(st.session_state.chats_duvidas) > 1:
                    del st.session_state.chats_duvidas[i]
                    st.session_state.chat_atual_duvidas = max(0, i - 1)
                    st.rerun()
                else:
                    st.warning("Mínimo de 1 chat.")

    elif aba_selecionada == "⚙️ Análise de Falhas":
        st.markdown("### 🕒 Histórico de Falhas")
        if st.button("➕ Novo Chat de Falhas", use_container_width=True):
            novo_id = len(st.session_state.chats_falhas) + 1
            st.session_state.chats_falhas.append({
                "id": novo_id, 
                "titulo": f"Falha {novo_id}", 
                "mensagens": [{"role": "assistant", "content": "Descreva a nova ocorrência ou falha observada."}]
            })
            st.session_state.chat_atual_falhas = len(st.session_state.chats_falhas) - 1
            st.rerun()

        for i, chat in enumerate(st.session_state.chats_falhas[-10:]):
            col_h1, col_h2 = st.columns([0.8, 0.2])
            if col_h1.button(chat["titulo"], key=f"btn_falha_{i}", use_container_width=True):
                st.session_state.chat_atual_falhas = i
                st.rerun()
            if col_h2.button("🗑️", key=f"del_falha_{i}"):
                if len(st.session_state.chats_falhas) > 1:
                    del st.session_state.chats_falhas[i]
                    st.session_state.chat_atual_falhas = max(0, i - 1)
                    st.rerun()
                else:
                    st.warning("Mínimo de 1 chat.")

    st.divider()
    total_pdfs = len(list(PASTA_BASE_MANUAIS.glob("**/*.pdf")))
    try:
        total_trechos = collection.count()
    except Exception:
        total_trechos = 0
        
    st.info(f"**Status:** Sistema Pronto\n\n📁 {total_pdfs} PDFs encontrados\n📚 {total_trechos} trechos ativos")

# --- ABA 1: DÚVIDAS TÉCNICAS (CONSULTA CONSULTIVA) ---
if aba_selecionada == "📖 Dúvidas Técnicas":
    st.markdown("### 📖 Dúvidas Técnicas e Consultas")
    
    if pergunta := st.chat_input("Qual dúvida técnica você tem hoje?"):
        st.session_state.mensagens_duvidas.append({"role": "user", "content": pergunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="🔹"):
            with st.spinner("Realizando busca profunda nos manuais..."):
                try:
                    # Aumento de n_results para ter mais contexto (12 trechos)
                    pergunta_vetor = encoder.encode([pergunta]).tolist()
                    resultados = collection.query(query_embeddings=pergunta_vetor, n_results=12)
                    
                    contexto_partes = []
                    fontes_encontradas = set()
                    if resultados and resultados["documents"]:
                        for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
                            contexto_partes.append(f"Fonte: {meta.get('source', 'Desconhecido')} | Conteúdo: {doc}")
                            fontes_encontradas.add(meta.get('source', 'Manual'))
                    
                    contexto = "\n\n".join(contexto_partes)
                    
                    # Prompt Refinado (Mais autoridade e rigor)
                    system_prompt = (
                        "Você é o Mir, o especialista sênior em manutenção ferroviária. Sua resposta deve ser técnica, direta e baseada APENAS no contexto fornecido.\n"
                        "PASSO A PASSO PARA RESPONDER:\n"
                        "1. Analise todos os trechos fornecidos abaixo.\n"
                        "2. Se a resposta estiver nos trechos, crie uma resposta estruturada (tópicos são preferidos).\n"
                        "3. Se a informação for insuficiente, diga: 'Com base na documentação disponível, não foi possível confirmar este dado' e dê uma sugestão de onde procurar.\n"
                        "4. Sempre cite a fonte (nome do PDF) ao lado da informação técnica.\n\n"
                        f"CONTEXTO EXTRAÍDO:\n{contexto}"
                    )

                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Pergunta: {pergunta}. Responda de forma detalhada e técnica."}
                        ],
                        temperature=0.0 # Temperatura zero para maior precisão técnica
                    )
                    
                    resposta_ia = response.choices[0].message.content
                    st.markdown(resposta_ia)
                    st.session_state.mensagens_duvidas.append({"role": "assistant", "content": resposta_ia})
                    
                except Exception as e:
                    st.error(f"Erro na consulta: {e}")

# --- ABA 2: ANÁLISE DE FALHAS ---
elif aba_selecionada == "⚙️ Análise de Falhas":
    st.markdown("### ⚙️ Análise de Falhas e Diagnósticos")
    
    chat_idx = st.session_state.chat_atual_falhas
    chat_atual = st.session_state.chats_falhas[chat_idx]

    # Exibir histórico de conversas do chat ativo
    for msg in chat_atual["mensagens"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Entrada do usuário
    if pergunta := st.chat_input("Descreva o equipamento, sistema e sintomas da falha..."):
        chat_atual["mensagens"].append({"role": "user", "content": pergunta})
        
        if chat_atual["titulo"].startswith("Nova Falha") or chat_atual["titulo"].startswith("Falha"):
            chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

        with st.chat_message("user", avatar="👤"):
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="🔹"):
            with st.spinner("Analisando falha e manuais..."):
                try:
                    pergunta_vetor = encoder.encode([pergunta]).tolist()
                    resultados = collection.query(query_embeddings=pergunta_vetor, n_results=6)
                    
                    contexto_partes = []
                    fontes_encontradas = set()
                    if resultados and resultados["documents"] and resultados["documents"][0]:
                        docs = resultados["documents"][0]
                        metas = resultados["metadatas"][0] if resultados.get("metadatas") else [{}] * len(docs)
                        for doc, meta in zip(docs, metas):
                            contexto_partes.append(doc)
                            if meta and "source" in meta:
                                fontes_encontradas.add(meta["source"])
                    
                    contexto = "\n\n".join(contexto_partes) if contexto_partes else "Nenhum trecho correspondente encontrado na base de PDFs local."
                    fontes_str = ", ".join(fontes_encontradas) if fontes_encontradas else "Nenhum manual PDF local indexado para citação."

                    system_prompt = (
                        "Você é o Mir, um técnico especialista sênior em manutenção de locomotivas (especialmente sistemas de freio CCBII).\n"
                        "Sua linguagem é técnica, direta e prática, voltada para profissionais de oficina (mecânicos e eletricistas).\n\n"
                        "DIRETRIZES DE RESPOSTA:\n"
                        "1. ANÁLISE DE DADOS: Sempre analise as variáveis (MR, BP, ER, BC, etc.) antes de dar o diagnóstico. Explique o porquê de cada valor ser relevante.\n"
                        "2. CONTEXTUALIZAÇÃO: Se o código de erro tem uma causa raiz comum (ex: 1106 = ERCP/13CP), comece por ela.\n"
                        "3. HIPÓTESES PRIORIZADAS: Liste as causas em ordem de probabilidade (o que é mais fácil/barato de verificar primeiro).\n"
                        "4. AÇÃO PRÁTICA: O que o técnico deve fazer AGORA na oficina? Liste os testes físicos e inspeções (ex: verificar escape, estrangulador, calibração).\n"
                        "5. POSTURA: Aja como um colega sênior de oficina. Se o usuário fornecer logs, monte uma linha de raciocínio. Se faltarem dados, peça os eventos anteriores/posteriores.\n\n"
                        "ESTRUTURA OBRIGATÓRIA DA RESPOSTA:\n"
                        "### 🔎 Interpretação do Evento\n"
                        "### 📊 Análise das Pressões\n"
                        "### 💡 Minha Hipótese de Diagnóstico\n"
                        "### 🛠️ Plano de Ação (Checklist de Oficina)\n\n"
                        f"Contexto técnico extraído dos manuais locais:\n{contexto}\n\n"
                        f"Fontes/Documentos disponíveis para referência: {fontes_str}"
                    )

                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": pergunta}
                        ],
                        temperature=0.1
                    )
                    resposta_ia = response.choices[0].message.content
                    st.markdown(resposta_ia)
                    chat_atual["mensagens"].append({"role": "assistant", "content": resposta_ia})
                except Exception as e:
                    erro_msg = f"Erro no pipeline de IA: {e}"
                    st.error(erro_msg)
                    chat_atual["mensagens"].append({"role": "assistant", "content": erro_msg})

# --- ABA 3: ADICIONAR CONHECIMENTO ---
elif aba_selecionada == "📂 Adicionar Conhecimento":
    st.markdown("# 📂 Adicionar Conhecimento (Gerenciamento de Manuais)")
    st.markdown("Carregue novos manuais em PDF para expandir imediatamente a base de conhecimento do Mir.")

    uploaded_files = st.file_uploader(
        "Carregar manuais técnicos (PDF)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            caminho_salvamento = PASTA_BASE_MANUAIS / uploaded_file.name
            with open(caminho_salvamento, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} arquivo(s) salvo(s) com sucesso!")

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧠 Indexar Novos PDFs da Pasta", use_container_width=True, type="primary"):
            arquivos = list(PASTA_BASE_MANUAIS.glob("**/*.pdf"))
            if not arquivos:
                st.warning("Nenhum arquivo PDF encontrado na pasta /Docs.")
            else:
                barra = st.progress(0, text="Iniciando indexação...")
                total_arq = len(arquivos)
                for idx, arq in enumerate(arquivos):
                    indexar_arquivo(arq)
                    barra.progress((idx + 1) / total_arq, text=f"Indexando: {arq.name}")
                st.success("Aprendizado concluído com sucesso!")
                st.rerun()

    with col_btn2:
        if st.button("🗑️ Limpar Base de Dados Indexada", use_container_width=True):
            try:
                chroma_client.delete_collection("mir_suporte_tecnico")
                collection = chroma_client.get_or_create_collection(name="mir_suporte_tecnico")
                st.success("Base de dados limpa com sucesso!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao limpar base: {e}")