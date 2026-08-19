import os
import base64
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mir - Suporte Técnico Virtual (Ferroviário)",
    page_icon="🚆",
    layout="wide"
)

# --- CSS CUSTOMIZADO PARA LIMPR O FILE UPLOADER ---
st.markdown("""
<style>
    /* Esconde o texto de limite de tamanho e tipos de arquivo do file_uploader */
    [data-testid="stFileUploader"] section small {
        display: none !important;
    }
    [data-testid="stFileUploader"] section div span {
        display: none !important;
    }
    /* Deixa o botão de upload mais compacto e com o texto 'print' */
    [data-testid="stFileUploader"] button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuração de Pastas ---
PASTA_BASE_MANUAIS = Path("./Docs")
PASTA_BASE_MANUAIS.mkdir(exist_ok=True)

# Inicialização dos componentes de IA (com cache)
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

def converter_imagem_para_base64(uploaded_file):
    """Converte arquivo de imagem enviado pelo Streamlit para string base64."""
    bytes_data = uploaded_file.getvalue()
    encoded = base64.b64encode(bytes_data).decode("utf-8")
    extensao = uploaded_file.type.split("/")[-1]
    return f"data:image/{extensao};base64,{encoded}"

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

# --- ABA 1: DÚVIDAS TÉCNICAS ---
if aba_selecionada == "📖 Dúvidas Técnicas":
    st.markdown("### 📖 Dúvidas Técnicas e Consultas")
    
    chat_idx = st.session_state.chat_atual_duvidas
    chat_atual = st.session_state.chats_duvidas[chat_idx]

    for msg in chat_atual["mensagens"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        st.markdown(item["text"])
                    elif item.get("type") == "image_url":
                        st.image(item["image_url"]["url"], width=300)
            else:
                st.markdown(msg["content"])

    col_input, col_file = st.columns([0.86, 0.14])
    with col_input:
        pergunta = st.chat_input("Qual dúvida técnica você tem hoje?", key="input_duvida")
    with col_file:
        imagem_enviada_duvida = st.file_uploader("print", type=["png", "jpg", "jpeg"], key="uploader_duvida")

    if pergunta:
        conteudo_usuario = []
        
        if imagem_enviada_duvida is not None:
            img_url = converter_imagem_para_base64(imagem_enviada_duvida)
            conteudo_usuario.append({"type": "image_url", "image_url": {"url": img_url}})
        
        conteudo_usuario.append({"type": "text", "text": pergunta})

        chat_atual["mensagens"].append({"role": "user", "content": conteudo_usuario})
        
        if chat_atual["titulo"].startswith("Nova Dúvida") or chat_atual["titulo"].startswith("Dúvida"):
            chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

        with st.chat_message("user", avatar="👤"):
            if imagem_enviada_duvida is not None:
                st.image(imagem_enviada_duvida, width=300)
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="🔹"):
            with st.spinner("Realizando busca profunda nos manuais..."):
                try:
                    pergunta_vetor = encoder.encode([pergunta]).tolist()
                    resultados = collection.query(query_embeddings=pergunta_vetor, n_results=12)
                    
                    contexto_partes = []
                    if resultados and resultados["documents"] and resultados["documents"][0]:
                        for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
                            contexto_partes.append(f"Fonte: {meta.get('source', 'Desconhecido')} | Conteúdo: {doc}")
                    
                    contexto = "\n\n".join(contexto_partes) if contexto_partes else "Nenhum trecho correspondente encontrado."
                    
                    system_prompt = (
                        "Você é o Mir, um assistente especialista sênior em engenharia e manutenção ferroviária. "
                        "Sua função principal é atuar como um consultor técnico de suporte, respondendo a dúvidas, "
                        "explicando conceitos e detalhando especificações com base estrita nos manuais e documentos indexados na base de dados.\n\n"
                        "DIRETRIZES DE ATUAÇÃO:\n"
                        "1. Fidelidade ao Contexto: Utilize os trechos dos manuais fornecidos abaixo como sua fonte primária de verdade técnica.\n"
                        "2. Clareza e Estrutura: Explique os conceitos de forma didática, organizada em tópicos (bullet points) ou passos.\n"
                        "3. Transparência em Caso de Omissão: Se a resposta exata não constar, informe educadamente e dê uma orientação geral ressalvando que não consta no manual.\n"
                        "4. Citação de Fontes: Sempre cite o nome do documento (PDF) correspondente ao lado da informação.\n\n"
                        f"CONTEXTO TÉCNICO EXTRAÍDO DOS MANUAIS:\n{contexto}"
                    )

                    # Limpa histórico para a API: envia apenas texto nas mensagens anteriores para evitar reenvio de imagens antigas
                    messages_payload = [{"role": "system", "content": system_prompt}]
                    for m in chat_atual["mensagens"][:-1]:
                        texto_limpo = m["content"]
                        if isinstance(texto_limpo, list):
                            texto_limpo = next((item["text"] for item in texto_limpo if item.get("type") == "text"), "")
                        messages_payload.append({"role": m["role"], "content": texto_limpo})
                    
                    # Mensagem atual com imagem (se houver)
                    messages_payload.append({"role": "user", "content": chat_atual["mensagens"][-1]["content"]})

                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=messages_payload,
                        temperature=0.0
                    )
                    
                    resposta_ia = response.choices[0].message.content
                    st.markdown(resposta_ia)
                    chat_atual["mensagens"].append({"role": "assistant", "content": resposta_ia})
                    st.rerun()
                    
                except Exception as e:
                    erro_msg = f"Erro na consulta: {e}"
                    st.error(erro_msg)
                    chat_atual["mensagens"].append({"role": "assistant", "content": erro_msg})

# --- ABA 2: ANÁLISE DE FALHAS ---
elif aba_selecionada == "⚙️ Análise de Falhas":
    st.markdown("### ⚙️ Análise de Falhas e Diagnósticos")
    
    chat_idx = st.session_state.chat_atual_falhas
    chat_atual = st.session_state.chats_falhas[chat_idx]

    for msg in chat_atual["mensagens"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        st.markdown(item["text"])
                    elif item.get("type") == "image_url":
                        st.image(item["image_url"]["url"], width=300)
            else:
                st.markdown(msg["content"])

    col_input, col_file = st.columns([0.86, 0.14])
    with col_input:
        pergunta = st.chat_input("Descreva o equipamento, sistema e sintomas da falha...", key="input_falha")
    with col_file:
        imagem_enviada_falha = st.file_uploader("print", type=["png", "jpg", "jpeg"], key="uploader_falha")

    if pergunta:
        conteudo_usuario = []
        
        if imagem_enviada_falha is not None:
            img_url = converter_imagem_para_base64(imagem_enviada_falha)
            conteudo_usuario.append({"type": "image_url", "image_url": {"url": img_url}})
        
        conteudo_usuario.append({"type": "text", "text": pergunta})

        chat_atual["mensagens"].append({"role": "user", "content": conteudo_usuario})
        
        if chat_atual["titulo"].startswith("Nova Falha") or chat_atual["titulo"].startswith("Falha"):
            chat_atual["titulo"] = pergunta[:25] + "..." if len(pergunta) > 25 else pergunta

        with st.chat_message("user", avatar="👤"):
            if imagem_enviada_falha is not None:
                st.image(imagem_enviada_falha, width=300)
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="🔹"):
            with st.spinner("Analisando falha, parâmetros e manuais..."):
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
                        "Você é o Mir, um técnico especialista sênior em manutenção de locomotivas (especialmente sistemas de freio CCBII e elétrica ferroviária).\n"
                        "Sua linguagem é técnica, direta, prática e corporativa de oficina (voltada para mecânicos e eletricistas).\n\n"
                        "DIRETRIZES DE RESPOSTA:\n"
                        "1. ANÁLISE DE DADOS E IMAGENS: Se o usuário enviou uma imagem ou texto com pressões (MR, BP, ER, BC), analise detalhadamente. Caso faltem informações essenciais, aponte o que falta.\n"
                        "2. CONTEXTUALIZAÇÃO: Se o código de erro ou sintoma possui uma causa raiz recorrente, comece por ela.\n"
                        "3. HIPÓTESES PRIORIZADAS: Liste as prováveis causas em ordem decrescente de probabilidade.\n"
                        "4. AÇÃO PRÁTICA: O que o mecânico/eletricista deve fazer AGORA na oficina?\n\n"
                        "ESTRUTURA OBRIGATÓRIA DA RESPOSTA:\n"
                        "### 🔎 Interpretação do Evento\n"
                        "### 📊 Análise das Variáveis / Imagem\n"
                        "### 💡 Minha Hipótese de Diagnóstico\n"
                        "### 🛠️ Plano de Ação (Checklist de Oficina)\n\n"
                        f"Contexto técnico extraído dos manuais locais:\n{contexto}\n\n"
                        f"Fontes/Documentos disponíveis para referência: {fontes_str}"
                    )

                    # Limpa histórico para a API: envia apenas texto nas mensagens anteriores para evitar reenvio de imagens antigas
                    messages_payload = [{"role": "system", "content": system_prompt}]
                    for m in chat_atual["mensagens"][:-1]:
                        texto_limpo = m["content"]
                        if isinstance(texto_limpo, list):
                            texto_limpo = next((item["text"] for item in texto_limpo if item.get("type") == "text"), "")
                        messages_payload.append({"role": m["role"], "content": texto_limpo})
                    
                    # Mensagem atual com imagem (se houver)
                    messages_payload.append({"role": "user", "content": chat_atual["mensagens"][-1]["content"]})

                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=messages_payload,
                        temperature=0.1
                    )
                    resposta_ia = response.choices[0].message.content
                    st.markdown(resposta_ia)
                    chat_atual["mensagens"].append({"role": "assistant", "content": resposta_ia})
                    st.rerun()
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