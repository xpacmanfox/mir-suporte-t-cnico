import os
from pathlib import Path
import streamlit as st
from openai import OpenAI
import pypdf
import pandas as pd

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mir - Central de Gestão e Catálogos",
    page_icon="📦",
    layout="wide"
)

# --- CSS CUSTOMIZADO PARA AUMENTAR AS FONTES ---
st.markdown("""
    <style>
        /* Aumenta a fonte do menu lateral (sidebar) */
        section[data-testid="stSidebar"] * {
            font-size: 16px !important;
        }
        
        /* Aumenta a fonte dos textos gerais e tabelas */
        .stMarkdown * {
            font-size: 16px !important;
        }
        
        /* Aumenta a fonte dos campos de texto e inputs */
        .stTextInput input {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Inicializa o estado da escolha do sistema
if "sistema_ativo" not in st.session_state:
    st.session_state.sistema_ativo = None

# Configuração da API do OpenRouter / OpenAI
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Mir Materiais"}
)

# Função de busca inteligente e flexível para planilhas
def buscar_materiais(df, termo_busca):
    if not termo_busca or df.empty:
        return pd.DataFrame()
    
    termos = termo_busca.lower().split()
    
    mascara = pd.Series(False, index=df.index)
    for col in df.columns:
        col_str = df[col].astype(str).str.lower()
        sub_mascara = pd.Series(True, index=df.index)
        for termo in termos:
            sub_mascara &= col_str.str.contains(termo, na=False, regex=False)
        mascara |= sub_mascara
        
    return df[mascara]

# --- FUNÇÃO DE BUSCA E GERENCIAMENTO DE CATÁLOGOS EM PDF ---
def gerenciar_modulo_catalogo(titulo_modulo, pasta_destino):
    if st.button("🏠 Voltar ao Menu Principal", type="secondary"):
        st.session_state.sistema_ativo = None
        st.rerun()
        
    st.title(titulo_modulo)
    st.markdown("Consulte peças, descrições e Part Numbers diretamente nos catálogos de fornecedores carregados na base.")
    
    pasta_catalogo = Path(pasta_destino)
    pasta_catalogo.mkdir(exist_ok=True, parents=True)
    
    aba_cat = st.radio("Navegação Catálogo", ["🔍 Realizar Busca", "📂 Gerenciar Base de PDFs"], horizontal=True, key=f"radio_{pasta_destino}")
    
    if aba_cat == "🔍 Realizar Busca":
        termo_busca = st.text_input("Digite o nome da peça, descrição ou Part Number:", key=f"input_{pasta_destino}")
        
        if st.button("Buscar no Catálogo", type="primary", key=f"btn_busca_{pasta_destino}"):
            if not termo_busca:
                st.warning("Por favor, digite um termo para buscar.")
            else:
                with st.spinner("Consultando catálogos e analisando com IA..."):
                    arquivos_pdf = list(pasta_catalogo.glob("**/*.pdf"))
                    if not arquivos_pdf:
                        st.warning(f"Nenhum arquivo PDF encontrado na pasta `{pasta_catalogo}`. Vá na aba 'Gerenciar Base de PDFs' para adicionar.")
                    else:
                        contexto_partes = []
                        for pdf_path in arquivos_pdf:
                            nome_modelo = pdf_path.stem.replace("_", " ").replace("-", " ")
                            try:
                                reader = pypdf.PdfReader(str(pdf_path))
                                for i, page in enumerate(reader.pages):
                                    texto = page.extract_text()
                                    if texto and termo_busca.lower() in texto.lower():
                                        contexto_partes.append(f"--- Equipamento/Arquivo: {nome_modelo} | Página: {i+1} ---\n{texto[:1200]}")
                            except Exception as e:
                                st.warning(f"Não foi possível ler o arquivo '{pdf_path.name}': {e}")
                                continue
                        
                        if not contexto_partes:
                            st.info("Nenhum trecho direto correspondente foi encontrado nos arquivos. Tente um termo mais genérico.")
                        else:
                            contexto = "\n\n".join(contexto_partes[:10])
                            
                            prompt_sistema = """
                            Você é um agente especialista em suprimentos, engenharia e peças industriais ferroviárias.
                            Sua tarefa é analisar os trechos de catálogos fornecidos, identificar **todos** os itens correspondentes à busca do usuário e agrupar separadamente por modelo (cujo nome consta no início de cada trecho do contexto).
                            
                            Para CADA item encontrado, retorne em um formato limpo, estruturado e fácil de ler (em bullet points ou cards):
                            - **🚆 Equipamento / Modelo:** [Nome extraído do arquivo]
                            - **🔩 Peça / Material:** [Nome claro do item]
                            - **🔢 Part Number / Código:** [Código encontrado no catálogo]
                            - **📄 Página de Referência:** [Número da página]
                            - **📝 Descrição / Detalhes:** [Breve resumo técnico]
                            
                            Exiba todos os resultados encontrados divididos por modelo. Seja organizado e objetivo.
                            """
                            
                            response = client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": prompt_sistema},
                                    {"role": "user", "content": f"Busca do usuário: {termo_busca}\n\nTrechos dos Catálogos:\n{contexto}"}
                                ],
                                temperature=0.0
                            )
                            st.markdown("### 📋 Resultados Encontrados:")
                            st.markdown(response.choices[0].message.content)
    
    else:
        st.markdown("### 📂 Gerenciamento do Banco de Dados de Catálogos")
        st.markdown("Faça o upload de novos catálogos em PDF para atualizar a base. **Dica:** Nomeie o arquivo PDF com o modelo correspondente.")
        
        uploaded_pdfs = st.file_uploader("Carregar novos catálogos (PDF)", type=["pdf"], accept_multiple_files=True, key=f"upload_pdf_{pasta_destino}")
        
        if uploaded_pdfs:
            for up_file in uploaded_pdfs:
                caminho_salvamento = pasta_catalogo / up_file.name
                with open(caminho_salvamento, "wb") as f:
                    f.write(up_file.getbuffer())
            st.success(f"{len(uploaded_pdfs)} catálogo(s) salvo(s) com sucesso!")
            
        st.divider()
        st.markdown("#### PDFs atualmente na base:")
        arquivos_atuais = list(pasta_catalogo.glob("**/*.pdf"))
        if arquivos_atuais:
            for arq in arquivos_atuais:
                col_p1, col_p2 = st.columns([0.8, 0.2])
                col_p1.text(f"📄 {arq.name}")
                if col_p2.button("Excluir", key=f"del_{pasta_destino}_{arq.name}"):
                    arq.unlink()
                    st.success(f"Arquivo {arq.name} removido!")
                    st.rerun()
        else:
            st.info("Nenhum PDF cadastrado no momento.")


# --- TELA INICIAL: ESCOLHA DE SISTEMA ---
if st.session_state.sistema_ativo is None:
    st.title("📦 MIR - Central de Gestão de Catálogos e Materiais")
    st.markdown("### Selecione qual sistema você deseja acessar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 🚆 Catálogo de Locomotivas")
        st.markdown("Busca inteligente de peças e Part Numbers em catálogos em PDF de locomotivas.")
        if st.button("Acessar Catálogo de Locomotivas", type="primary"):
            st.session_state.sistema_ativo = "catalogo_locomotivas"
            st.rerun()
            
        st.info("### 🛤️ Catálogo de Máquinas de Via")
        st.markdown("Busca inteligente de peças e Part Numbers em catálogos de máquinas de via permanente.")
        if st.button("Acessar Catálogo de Máquinas de Via", type="primary"):
            st.session_state.sistema_ativo = "catalogo_maquinas_via"
            st.rerun()        
            
    with col2:
        st.info("### 📋 Código de Materiais")
        st.markdown("Consulta em planilha com códigos internos da empresa para requisição de materiais.")
        if st.button("Acessar Código de Materiais", type="primary"):
            st.session_state.sistema_ativo = "codigo_materiais"
            st.rerun()

# --- ROTEAMENTO DOS MÓDULOS SELECIONADOS ---
elif st.session_state.sistema_ativo == "catalogo_locomotivas":
    gerenciar_modulo_catalogo("🚆 Catálogo de Peças - Locomotivas", "./Docs_Catalogos_Locomotivas")

elif st.session_state.sistema_ativo == "catalogo_maquinas_via":
    gerenciar_modulo_catalogo("🛤️ Catálogo de Peças - Máquinas de Via", "./Docs_Catalogos_MaquinasVia")

elif st.session_state.sistema_ativo == "codigo_materiais":
    if st.button("🏠 Voltar ao Menu Principal", type="secondary"):
        st.session_state.sistema_ativo = None
        st.rerun()
        
    st.title("📋 Buscador de Código de Materiais (Planilha Interna)")
    st.markdown("Consulte rapidamente os códigos internos de materiais da empresa para requisições e compras.")
    
    pasta_excel = Path("./Docs_Planilhas")
    pasta_excel.mkdir(exist_ok=True, parents=True)
    caminho_excel = pasta_excel / "materiais_internos.xlsx"
    
    aba_mat = st.radio("Navegação Materiais", ["📋 Realizar Consulta", "📂 Gerenciar Planilha de Dados"], horizontal=True, key="radio_materiais_interno")
    
    if aba_mat == "📋 Realizar Consulta":
        termo_interno = st.text_input("Digite o nome ou código interno do material (ex: Sensor DSS):", key="input_termo_material")
        
        if st.button("Consultar Materiais", type="primary", key="btn_consulta_materiais"):
            if not termo_interno:
                st.warning("Informe um termo para a consulta.")
            else:
                with st.spinner("Buscando na planilha interna..."):
                    if not caminho_excel.exists():
                        st.error(f"Nenhuma planilha encontrada em `{caminho_excel}`. Vá na aba 'Gerenciar Planilha de Dados' para fazer o upload.")
                    else:
                        df = pd.read_excel(caminho_excel)
                        df.columns = df.columns.str.strip()
                        
                        resultado = buscar_materiais(df, termo_interno)
                        
                        if not resultado.empty:
                            st.success(f"Foram encontrados {len(resultado)} item(ns):")
                            st.dataframe(resultado, width='stretch')
                        else:
                            st.warning(f"Nenhum item correspondente a '{termo_interno}' foi encontrado na planilha.")
    else:
        st.markdown("### 📂 Gerenciamento da Planilha de Materiais")
        st.markdown("Faça o upload da planilha atualizada contendo os códigos internos (`.xlsx` ou `.xls`).")
        
        uploaded_excel = st.file_uploader("Carregar planilha de materiais", type=["xlsx", "xls"], key="upload_excel_mat")
        
        if uploaded_excel:
            with open(caminho_excel, "wb") as f:
                f.write(uploaded_excel.getbuffer())
            st.success("Planilha de materiais atualizada com sucesso!")
            
        st.divider()
        if caminho_excel.exists():
            st.success(f"✅ Planilha ativa no sistema: `{caminho_excel.name}`")
            try:
                df_preview = pd.read_excel(caminho_excel)
                st.markdown("#### Pré-visualização dos dados:")
                st.dataframe(df_preview.head(10), width='stretch')
            except Exception as e:
                st.error(f"Erro ao ler a planilha: {e}")
        else:
            st.warning("⚠️ Nenhuma planilha de materiais carregada ainda.")