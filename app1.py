import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf
from pathlib import Path

# --- Configuração de Pasta ---
PASTA_DOCS = Path("./Docs")
PASTA_DOCS.mkdir(exist_ok=True)

st.sidebar.header("📂 Administração do Mir")

# Botão de Upload
uploaded_files = st.sidebar.file_uploader(
    "Carregar manuais técnicos (PDF)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        caminho_salvamento = PASTA_DOCS / uploaded_file.name
        
        # Salva o arquivo na pasta Docs do servidor
        with open(caminho_salvamento, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"{len(uploaded_files)} arquivo(s) carregado(s)!")
    
    # DICA: Se você tiver uma função que indexa os PDFs no ChromaDB, 
    # chame ela aqui para atualizar a memória do Mir automaticamente.
    # Exemplo: indexar_manuais_pasta(PASTA_DOCS)

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mir - Suporte Técnico Virtual (Ferroviário)",
    page_icon="🚆",
    layout="wide"
)

# Diretório de Manuais Locais
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
# Carrega a chave de forma segura
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

# Inicialização do histórico de mensagens no Session State do Streamlit
if "mensagens_falhas" not in st.session_state:
    st.session_state.mensagens_falhas = [
        {"role": "assistant", "content": "Olá. Sou o agente especialista em manutenção ferroviária. Estou pronto para auxiliar técnicos, mecânicos, eletricistas, inspetores e engenheiros. Descreva a falha ou anomalia observada."}
    ]

if "mensagens_duvidas" not in st.session_state:
    st.session_state.mensagens_duvidas = [
        {"role": "assistant", "content": "Olá. Use este canal para retirar dúvidas gerais, realizar consultas técnicas ou verificar especificações priorizando os manuais indexados."}
    ]

# --- SIDEBAR MODERNA ---
with st.sidebar:
    st.markdown("## 🚆 MIR DESKTOP")
    st.caption("Inteligência em Manutenção")
    st.divider()

    aba_selecionada = st.radio(
        "Navegação",
        ["🏠 Início", "💬 Análise de Falhas", "📖 Dúvidas Técnicas"],
        label_visibility="collapsed"
    )

    st.divider()
    total_pdfs = len(list(PASTA_BASE_MANUAIS.glob("**/*.pdf")))
    try:
        total_trechos = collection.count()
    except Exception:
        total_trechos = 0
        
    st.info(f"**Status:** Sistema Pronto\n\n📁 {total_pdfs} PDFs encontrados\n📚 {total_trechos} trechos ativos")

# --- ABA 1: INÍCIO ---
if aba_selecionada == "🏠 Início":
    st.markdown("# Painel de Controle Técnico - Mir 🚆")
    st.markdown("Sistema inteligente de suporte à manutenção ferroviária com Agente Especialista e priorização de PDFs locais.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 PDFs no Diretório (/Docs)", f"{total_pdfs} arquivos")
    with col2:
        st.metric("📚 Trechos Indexados", f"{total_trechos} trechos")

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧠 Aprender (Indexar Novos PDFs)", use_container_width=True, type="primary"):
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
       if st.button("🗑️ Limpar Arquivos Indexados", use_container_width=True):
            try:
                chroma_client.delete_collection("mir_suporte_tecnico")
                collection = chroma_client.get_or_create_collection(name="mir_suporte_tecnico")
                st.success("Base de dados limpa com sucesso!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao limpar base: {e}")

    st.markdown("### 💡 Diretrizes do Agente Especialista")
    st.info(
        "📌 **Perfil da IA:** Especialista em manutenção ferroviária dedicado a auxiliar técnicos, mecânicos, eletricistas, inspetores e engenheiros.\n\n"
        "📌 **Prioridade de Consulta:** O sistema busca primeiramente nos manuais em PDF indexados na pasta `./Docs`. Caso não encontre especificações locais suficientes, utiliza o conhecimento técnico geral para complementar, sem inventar valores ou dados fictícios.\n\n"
        "⚠️ **Restrições Rígidas:** Priorize procedimentos documentados. Nunca invente valores de torque, pressão ou ajustes. Quando não existir informação na documentação, informe isso claramente."
    )

# --- ABA 2: ANÁLISE DE FALHAS ---
elif aba_selecionada == "💬 Análise de Falhas":
    col_titulo, col_limpar = st.columns([0.8, 0.2])
    with col_titulo:
        st.markdown("### 💬 Análise de Falhas e Diagnósticos")
    with col_limpar:
        if st.button("🗑️ Limpar Chat", use_container_width=True):
            st.session_state.mensagens_falhas = [
                {"role": "assistant", "content": "Modo de Análise de Falhas limpo. Insira a nova ocorrência."}
            ]
            st.rerun()

    # Exibir histórico de conversas
    for msg in st.session_state.mensagens_falhas:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user", avatar="🔹" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Entrada do usuário
    if pergunta := st.chat_input("Descreva o equipamento, sistema e sintomas da falha..."):
        st.session_state.mensagens_falhas.append({"role": "user", "content": pergunta})
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
                        "Você é o Mir, um agente especialista sênior em manutenção ferroviária. Sua missão é auxiliar técnicos ferroviários, "
                        "mecânicos, eletricistas, inspetores e engenheiros.\n\n"
                        "DIRETRIZES OBRIGATÓRIAS:\n"
                        "1. PRIORIDADE LOCAL: Consulte rigorosamente os trechos extraídos dos manuais em PDF fornecidos abaixo. Baseie suas respostas primordialmente neles.\n"
                        "2. SUPLEMENTAÇÃO EXTERNA: Caso os manuais locais não contenham informações completas sobre o assunto, você pode utilizar seu conhecimento geral técnico da indústria ferroviária para complementar a resposta, mas ressalte quando a informação vier de fora dos PDFs locais.\n"
                        "3. RESTRIÇÕES RÍGIDAS: Nunca invente valores de torque, pressão ou ajustes. Se não houver informação específica nos manuais ou se for incerto, informe claramente essa limitação.\n"
                        "Ao analisar falhas, siga obrigatoriamente a seguinte estrutura em sua resposta:\n"
                        "RESUMO\n"
                        "SINTOMA\n"
                        "POSSÍVEIS CAUSAS\n"
                        "VERIFICAÇÕES RECOMENDADAS\n"
                        "TESTES RECOMENDADOS\n"
                        "AÇÃO CORRETIVA SUGERIDA\n"
                        "DOCUMENTOS CONSULTADOS\n\n"
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
                    st.session_state.mensagens_falhas.append({"role": "assistant", "content": resposta_ia})
                except Exception as e:
                    erro_msg = f"Erro no pipeline de IA: {e}"
                    st.error(erro_msg)
                    st.session_state.mensagens_falhas.append({"role": "assistant", "content": erro_msg})

# --- ABA 3: DÚVIDAS TÉCNICAS (OTIMIZADA) ---
elif aba_selecionada == "📖 Dúvidas Técnicas":
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