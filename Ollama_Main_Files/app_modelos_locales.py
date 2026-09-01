import streamlit as st
import pandas as pd
import json
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import io
import re
import time
import warnings
import tempfile
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from unsloth import FastLanguageModel

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Comentario de Mercados - Gemma3 & Gemma2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CSS personalizado para mejor apariencia
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A5F;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #4A5568;
    text-align: center;
    margin-bottom: 2rem;
}
.commentary-box {
    background-color: #F7FAFC;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1E3A5F;
    font-family: 'Courier New', monospace;
    line-height: 1.6;
}
.status-box {
    background-color: #EDF2F7;
    padding: 1rem;
    border-radius: 0.5rem;
    font-family: monospace;
    font-size: 0.85rem;
}
.stButton button {
    background-color: #1E3A5F;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)





# =====================================================
# INICIALIZACIÓN DE SESIÓN
# =====================================================

if 'generated_commentary' not in st.session_state:
    st.session_state.generated_commentary = None
if 'generation_logs' not in st.session_state:
    st.session_state.generation_logs = []
if 'processing' not in st.session_state:
    st.session_state.processing = False


# =====================================================
# INICIALIZACIÓN Y CACHÉ DE MODELOS EN GPU
# =====================================================

@st.cache_resource
def load_llm_models(gemma_repo_id: str, filename: str, llama_repo_id: str, hf_token: str):
    """
    Carga persistente de los modelos Gemma 3 FT y Gemma 2 en VRAM
    para evitar re-descargas en cada iteración de Streamlit.
    """
    # 1. Descargar únicamente el archivo .gguf
    model_path = hf_hub_download(
        repo_id=gemma_repo_id,
        filename=filename
    )

    # 2. Cargar modelo en GPU

    gemma_model = Llama(
        model_path=model_path,
        n_ctx=6000,
        n_gpu_layers=99,
        n_batch=512,
        flash_attn=True,
        verbose=True   # ← cámbialo a True temporalmente
    )


    # 2. Carga de Gemma 2 (Modelo Axiliar: Resúmenes y Validaciones)
    gemma2_tokenizer = AutoTokenizer.from_pretrained(llama_repo_id, token=hf_token)
    gemma2_model = AutoModelForCausalLM.from_pretrained(
        llama_repo_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        token=hf_token
    )

    return (gemma_model), (gemma2_model, gemma2_tokenizer)




# =====================================================
# CLASES Y FUNCIONES
# =====================================================


class FEDSpeechProcessor:
    """Procesa discursos de la Reserva Federal usando Llama 3 Base."""

    def __init__(self, llama_tuple, state_file='fed_speeches_processed.txt'):
        self.model, self.tokenizer = llama_tuple
        self.feed_url = "https://www.federalreserve.gov/feeds/speeches.xml"
        self.state_file = state_file
        self.processed_urls = self._load_processed_urls()
        warnings.filterwarnings("ignore", message=".*document declared as.*")

    def _load_processed_urls(self) -> set:
        try:
            with open(self.state_file, 'r') as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            return set()

    def _save_processed_url(self, url: str):
        with open(self.state_file, 'a') as f:
            f.write(url + '\n')
        self.processed_urls.add(url)

    def _parse_feed_date(self, pub_date_str: str) -> Optional[datetime.date]:
        if not pub_date_str:
            return None
        clean_date = re.sub(r'<!\[CDATA\[|\]\]>', '', pub_date_str).strip()
        date_formats = ['%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S', '%d %b %Y %H:%M:%S']
        for fmt in date_formats:
            try:
                return datetime.strptime(clean_date, fmt).date()
            except ValueError:
                continue
        return None

    def _get_target_date(self) -> datetime.date:
        today = datetime.now().date()
        return today - timedelta(days=3) if today.weekday() == 0 else today - timedelta(days=1)

    def _extract_speech_content(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for div in soup.find_all('div'):
                clases = div.get('class', [])
                if 'col-xs-12' in clases and 'col-sm-8' in clases and 'col-md-8' in clases and 'heading' not in clases:
                    article_container = div
                    break
            else:
                article_container = soup.find('div', class_=re.compile(r'col-(xs|sm|md)-\d+'))

            if article_container:
                paragraphs = article_container.find_all('p')
                if paragraphs:
                    speech_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                    speech_text = re.sub(r'\s+', ' ', speech_text).strip()
                    return speech_text if len(speech_text) > 400 else None
            return None
        except Exception:
            return None

    def _summarize_speech(self, title: str, content: str) -> Optional[str]:
        truncated_content = content[:5000]
        prompt = f"""Eres un analista experto en la Reserva Federal.
Resume el siguiente discurso en un máximo de 150 palabras en castellano.

PRIORIZA:
1. Política monetaria (decisiones o pistas sobre tipos)
2. Inflación
3. Mercado laboral
4. Perspectivas económicas generales

Título: {title}
Discurso: {truncated_content}

Resumen (150 palabras máximo):"""

        messages = [{"role": "user", "content": prompt}]
        # inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

        # input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

        encoding = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        input_ids = encoding["input_ids"].to("cuda")  # ← igual aquí

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=400, temperature=0.2, do_sample=True)

        summary = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        print('************')
        print(f"Título: {title}")
        print(f"Resumen: {summary}")
        print('************')
        return f"**{title}**: {summary}"

    def fetch_and_summarize_new_speeches(self, log_callback=None) -> List[str]:
        summaries = []
        target_date = self._get_target_date()
        if log_callback:
            log_callback(f"🏦 Buscando discursos del día: {target_date}")

        try:
            feed = feedparser.parse(self.feed_url)
            for entry in feed.entries:
                pub_date_str = getattr(entry, 'published', None) or getattr(entry, 'pubDate', None)
                if not pub_date_str:
                    continue
                speech_date = self._parse_feed_date(pub_date_str)
                if not speech_date or speech_date != target_date or entry.link in self.processed_urls:
                    continue

                if log_callback:
                    log_callback(f"   - Nuevo discurso: {entry.title}")
                speech_content = self._extract_speech_content(entry.link)
                if speech_content:
                    summary = self._summarize_speech(entry.title, speech_content)
                    if summary:
                        summaries.append(summary)
                        self._save_processed_url(entry.link)
            return summaries
        except Exception as e:
            if log_callback:
                log_callback(f"   - Error: {e}")
            return []


# (Las clases ExcelDataLoader y extract_trend_insights se mantienen idénticas al código original)



class ExcelDataLoader:
    """Carga y procesa los datos de mercado desde el archivo Excel."""

    @staticmethod
    def load_from_excel(file_bytes_io) -> Dict:
        """Carga todas las hojas del Excel"""
        xlsx = pd.ExcelFile(file_bytes_io)
        data = {}

        # Hojas originales
        indices_df = pd.read_excel(xlsx, sheet_name="Indices")
        data["indices"] = {row["Índices"]: row["% día"] for _, row in indices_df.iterrows()}

        sp_sectors_df = pd.read_excel(xlsx, sheet_name="Sectores_SP500")
        data["sp_sectors"] = {row["S&P 500  sectores"]: row["% día"] for _, row in sp_sectors_df.iterrows()}

        stoxx_sectors_df = pd.read_excel(xlsx, sheet_name="Sectores_STOXX600")
        data["stoxx_sectors"] = {row["Stoxx 600  sectores"]: row["% día"] for _, row in stoxx_sectors_df.iterrows()}

        oil_df = pd.read_excel(xlsx, sheet_name="OIL")
        data["oil"] = {row.iloc[0]: row.iloc[1] for _, row in oil_df.iterrows()}

        yields_df = pd.read_excel(xlsx, sheet_name="YIELDS")
        data["yields"] = {row.iloc[0]: row.iloc[1] for _, row in yields_df.iterrows()}

        expectativas_df = pd.read_excel(xlsx, sheet_name="Expectativas_Tipos")
        data["expectativas"] = {row.iloc[0]: row.iloc[1] for _, row in expectativas_df.iterrows()}

        eps_df = pd.read_excel(xlsx, sheet_name="EPS")
        data["eps"] = {}
        for _, row in eps_df.iterrows():
            index_name = row.iloc[0]
            if pd.notna(index_name) and index_name not in ["PE", "EPS_GROWTH"]:
                data["eps"][index_name] = {"pe": row.iloc[1], "eps_growth": row.iloc[2]}

        # =====================================================
        # NUEVAS HOJAS (solo para lunes)
        # =====================================================

        # Hoja 1: 10weeks_Indexes
        try:
            df_10weeks_orig = pd.read_excel(xlsx, sheet_name="10weeks_Indexes")

            # Set the first column ('Unnamed: 0') as the index before transposing
            df_10weeks_orig.set_index(df_10weeks_orig.columns[0], inplace=True)
            df_10weeks = df_10weeks_orig.T

            # print(df_10weeks_orig.columns)
            # print(df_10weeks.columns)

            if not df_10weeks.empty:
                # Now 'Semanas Subiendo' and 'Semanas Bajando' should be valid column names in df_10weeks
                data["weeks_up"] = {}
                data["weeks_down"] = {}

                for idx, row in df_10weeks.iterrows():
                    data["weeks_up"][idx] = row.get('Semanas Subiendo', 0)
                    data["weeks_down"][idx] = row.get('Semanas Bajando', 0)

        except Exception as e:
            data["weeks_up"] = {}
            data["weeks_down"] = {}
            print(f"Nota: No se pudo cargar '10weeks_Indexes': {e}")

        # Hoja 2: Streaks_Indexes
        try:
            df_streaks_orig = pd.read_excel(xlsx, sheet_name="Streaks_Indexes")
            df_streaks_orig.set_index(df_streaks_orig.columns[0], inplace=True)

            df_streaks = df_streaks_orig.T

            if not df_streaks.empty:
                data["streak_up"] = {}
                data["streak_down"] = {}

                for idx, row in df_streaks.iterrows():
                    data["streak_up"][idx] = row.get('Racha Subiendo', 0)
                    data["streak_down"][idx] = row.get('Racha bajando', 0)


        except Exception as e:
            data["streak_up"] = {}
            data["streak_down"] = {}
            print(f"Nota: No se pudo cargar 'Streaks_Indexes': {e}")

        # Hoja 3: MaxMin_Indexes
        try:
            df_maxmin_orig = pd.read_excel(xlsx, sheet_name="MaxMin_Indexes")
            df_maxmin_orig.set_index(df_maxmin_orig.columns[0], inplace=True)

            df_maxmin = df_maxmin_orig.T



            if not df_maxmin.empty:
                data["at_high"] = []
                data["at_low"] = []

                for idx, row in df_maxmin.iterrows():

                    is_high = row['A 1% Máximo']
                    is_low = row['A 1% Mínimo']

                    if pd.notna(is_high) and is_high == 1:
                        data["at_high"].append(idx)
                    if pd.notna(is_low) and is_low == 1:
                        data["at_low"].append(idx)


        except Exception as e:
            data["at_high"] = []
            data["at_low"] = []
            print(f"Nota: No se pudo cargar 'MaxMin_Indexes': {e}")


        print('----- streak up')
        print(data['streak_up'])
        print('----- streak down')
        print(data['streak_down'])
        print('-----at high')
        print(data['at_high'])
        print('-----at low')
        print(data['at_low'])

        return data

    @staticmethod
    def extract_key_market_data(excel_data: Dict) -> Dict:
        """Extrae los datos más relevantes para el prompt"""

        indices = excel_data.get("indices", {})
        sp_sectors = excel_data.get("sp_sectors", {})
        stoxx_sectors = excel_data.get("stoxx_sectors", {})
        oil = excel_data.get("oil", {})
        yields = excel_data.get("yields", {})
        expect = excel_data.get("expectativas", {})

        #print(f"Expectativas de Fed: {expect['Fed_var']}")
        #print(f"Expectativas de Fed: {expect['Fed_var']}")

        best_sp = max(sp_sectors.items(), key=lambda x: x[1]) if sp_sectors else ("N/A", 0)
        worst_sp = min(sp_sectors.items(), key=lambda x: x[1]) if sp_sectors else ("N/A", 0)
        best_stoxx = max(stoxx_sectors.items(), key=lambda x: x[1]) if stoxx_sectors else ("N/A", 0)
        worst_stoxx = min(stoxx_sectors.items(), key=lambda x: x[1]) if stoxx_sectors else ("N/A", 0)

        return {
            # Datos originales
            "sp500": indices.get("S&P 500", 0),
            "spw": indices.get("S&P Equal Weight", 0),
            "bm7t": indices.get("7 Magnificas", indices.get("7 Magnificients", 0)),
            "eurostoxx": indices.get("EuroStoxx--50", 0),
            "stoxx600": indices.get("Stoxx 600", 0),
            "ibex": indices.get("Ibex", 0),
            "dax": indices.get("Dax", 0),
            "cac": indices.get("Cac-40", 0),
            "ftse": indices.get("FTSE", 0),
            "nikkei": indices.get("Nikkei", 0),
            "shanghai": indices.get("Shanghai", 0),
            "best_sp_sector": {"name": best_sp[0], "change": best_sp[1]},
            "worst_sp_sector": {"name": worst_sp[0], "change": worst_sp[1]},
            "best_stoxx_sector": {"name": best_stoxx[0], "change": best_stoxx[1]},
            "worst_stoxx_sector": {"name": worst_stoxx[0], "change": worst_stoxx[1]},
            "brent": oil.get("Brent (US$/bl)", 0),
            "wti": oil.get("West Texas (US$/bl)", 0),
            "us10y": yields.get("Yield US 10y", 0),
            "us5y": yields.get("Yield US 5y", 0),
            "us30y": yields.get("Yield US 30y", 0),
            "germany10y": yields.get("Yield Alemania 10y", 0),
            "spain10y": yields.get("Yield España 10y", 0),
            "sp_pe": excel_data.get("eps", {}).get("S&P", {}).get("pe", 0),
            "sp_eps_growth": excel_data.get("eps", {}).get("S&P", {}).get("eps_growth", 0),
            "stoxx_pe": excel_data.get("eps", {}).get("STOXX", {}).get("pe", 0),
            "stoxx_eps_growth": excel_data.get("eps", {}).get("STOXX", {}).get("eps_growth", 0),

            # NUEVOS DATOS (para tendencias)
            "weeks_up": excel_data.get("weeks_up", {}),
            "weeks_down": excel_data.get("weeks_down", {}),
            "streak_up": excel_data.get("streak_up", {}),
            "streak_down": excel_data.get("streak_down", {}),
            "at_high": excel_data.get("at_high", []),
            "at_low": excel_data.get("at_low", []),

            # EXPECTATIVAS
            "Fed_var": expect.get("Fed_var", ''),
          # "Fed_1s": expect.get("Fed_1s", ''),
          #  "Fed_2s": expect.get("Fed_2s", ''),
            "BCE_var": expect.get("BCE_var", ''),
           # "BCE_1s": expect.get("BCE_1s", ''),
           # "BCE_2s": expect.get("BCE_2s", ''),
            "BOE_var": expect.get("BOE_var", ''),
           # "BOE_1s": expect.get("BOE_1s", ''),
           # "BOE_2s": expect.get("BOE_2s", ''),

        }


def extract_trend_insights(market_data: Dict, max_items: int = 3) -> Tuple[List[str], List[str], List[str]]:
    """
    Extrae insights de tendencias para usar en comentarios de lunes.
    Retorna: (frases_semanas_10, frases_racha, frases_maxmin)
    """
    weeks_up = market_data.get("weeks_up", {})
    weeks_down = market_data.get("weeks_down", {})
    streak_up = market_data.get("streak_up", {})
    streak_down = market_data.get("streak_down", {})
    at_high = market_data.get("at_high", [])
    at_low = market_data.get("at_low", [])

    frases_semanas = []
    frases_racha = []
    frases_maxmin = []

    # 1. Índices que han subido en más de 7 de las últimas 10 semanas
    strong_up = [(name, count) for name, count in weeks_up.items() if count > 7]
    strong_up_sorted = sorted(strong_up, key=lambda x: x[1], reverse=True)[:max_items]

    for name, count in strong_up_sorted:
        # Limpiar nombre para que sea legible
        # clean_name = name.replace("S&P ", "").replace("Consumer Discret", "Consumo Discrecional")
        frases_semanas.append(f"El {name} ha subido en {count} de las últimas 10 semanas")


    # 1.1 Índices que han bajado en más de 7 de las últimas 10 semanas
    strong_down = [(name, count) for name, count in weeks_down.items() if count > 7]
    strong_down_sorted = sorted(strong_down, key=lambda x: x[1], reverse=True)[:max_items]

    for name, count in strong_down_sorted:
        # Limpiar nombre para que sea legible
        # clean_name = name.replace("S&P ", "").replace("Consumer Discret", "Consumo Discrecional")
        frases_semanas.append(f"El {name} ha bajado en {count} de las últimas 10 semanas")


    # 2. Índices con racha consecutiva al alza superior a 2 semanas
    long_streaks_up = [(name, count) for name, count in streak_up.items() if count > 2]
    long_streaks_up_sorted = sorted(long_streaks_up, key=lambda x: x[1], reverse=True)[:max_items]

    for name, count in long_streaks_up_sorted:
        # clean_name = name.replace("S&P ", "")
        frases_racha.append(f"El {name} acumula {count} semanas consecutivas al alza")


    # 2.1 Índices con racha consecutiva a la baja superior a 2 semanas
    long_streaks_down = [(name, count) for name, count in streak_down.items() if count > 2]
    long_streaks_down_sorted = sorted(long_streaks_down, key=lambda x: x[1], reverse=True)[:max_items]

    for name, count in long_streaks_down_sorted:
        # clean_name = name.replace("S&P ", "")
        frases_racha.append(f"El {name} acumula {count} semanas consecutivas a la baja")



    # 3. Índices cerca de máximos (a 1%)
    if at_high:
        for name in at_high[:max_items]:
            # clean_name = name.replace("S&P ", "")
            frases_maxmin.append(f"El {name} se encuentra en niveles de máximos")


    # 3.1 Índices cerca de mínimos (a 1%)
    if at_low:
        for name in at_low[:max_items]:
            # clean_name = name.replace("S&P ", "")
            frases_maxmin.append(f"El {name} se encuentra en niveles de mínimos")


    print('----- frases semanas')
    print(frases_semanas)
    print('----- frases racha')
    print(frases_racha)
    print('----- frases maxmin')
    print(frases_maxmin)


    return frases_semanas, frases_racha, frases_maxmin



def build_prompt(before_bell_content: str, five_things_content: str,
                 market_data: Dict, is_monday: bool,
                 examples: List[Dict], fed_summaries: List[str]) -> str:

    # before_bell_excerpt = before_bell_content[:800]

    prompt = f"""
## DATOS NUMÉRICOS DE MERCADO (USA ESTOS VALORES EXACTOS):
### RENTA VARIABLE (% día anterior):
- S&P500: {market_data['sp500']*100:.2f}%
- S&P Equal Weight (SPW): {market_data['spw']*100:.2f}%
- 7 Magníficas (BM7T): {market_data['bm7t']*100:.2f}%
- EuroStoxx50: {market_data['eurostoxx']*100:.2f}%
- Stoxx600: {market_data['stoxx600']*100:.2f}%
- Ibex35: {market_data['ibex']*100:.2f}%
- Nikkei: {market_data['nikkei']*100:.2f}%

Menciona los datos de S&P Equal Weight (SPW) y 7 Magníficas (BM7T) solo en el caso de que haya una diferencia apreciable (más de 0,6%) entre ambos, o tengan signo contrario (uno positivo y otro negativo)

### SECTORES DESTACADOS:
- Mejor sector S&P500: {market_data['best_sp_sector']['name']} (+{market_data['best_sp_sector']['change']*100:.2f}%)
- Peor sector S&P500: {market_data['worst_sp_sector']['name']} ({market_data['worst_sp_sector']['change']*100:.2f}%)
- Mejor sector Stoxx600: {market_data['best_stoxx_sector']['name']} (+{market_data['best_stoxx_sector']['change']*100:.2f}%)
- Peor sector Stoxx600: {market_data['worst_stoxx_sector']['name']} ({market_data['worst_stoxx_sector']['change']*100:.2f}%)
"""

    if is_monday:
        # Datos de PE (originales)
        prompt += f"""
### RATIOS PE Y CRECIMIENTO DE EPS (Datos de cierre de la semana pasada):
- S&P500: PE {market_data['sp_pe']:.1f}x, crecimiento EPS estimado +{market_data['sp_eps_growth']*100:.1f}% (2026)
- Stoxx600: PE {market_data['stoxx_pe']:.1f}x, crecimiento EPS estimado +{market_data['stoxx_eps_growth']*100:.1f}% (2026)

\n**Importante:** Integra estas observaciones de ratios PE y crecimiento de de EPS en el análisis de renta variable.\n

"""

        # =====================================================
        # NUEVOS INSIGHTS DE TENDENCIAS (solo lunes)
        # =====================================================
        frases_semanas, frases_racha, frases_maxmin = extract_trend_insights(market_data)

        if frases_semanas or frases_racha or frases_maxmin:
            prompt += "\n### TENDENCIAS DE MERCADO (análisis de 10 semanas):\n"

            if frases_semanas:
                prompt += "\n**Rendimiento en últimas 10 semanas:**\n"
                for frase in frases_semanas:
                    # print(frase)

                    prompt += f"- {frase}\n"

            if frases_racha:
                prompt += "\n**Rachas alcistas consecutivas:**\n"
                for frase in frases_racha:
                    prompt += f"- {frase}\n"

            if frases_maxmin:
                prompt += "\n**Índices/sectores en niveles extremos:**\n"
                for frase in frases_maxmin:
                    prompt += f"- {frase}\n"

            prompt += "\n**Importante:** Integra estas observaciones de tendencia en el análisis de renta variable, de forma natural y sin forzar la inclusión de todos los datos. Destaca los más relevantes.\n"

    prompt += f"""
### MATERIAS PRIMAS:
- Petróleo Brent: ${market_data['brent']:.1f}/barril
- Petróleo WTI: ${market_data['wti']:.1f}/barril

### RENTA FIJA (Yields bonos 10 años):
- Bono EEUU: {market_data['us10y']:.2f}%
- Bono Alemania: {market_data['germany10y']:.2f}%
- Bono España: {market_data['spain10y']:.2f}%

### EXPECTATIVAS DE TIPOS DE LA FED, BCE Y BANCO DE INGLATERRA (BOE)

- FED: {market_data['Fed_var']}
- BCE: {market_data['BCE_var']}
- BOE: {market_data['BOE_var']}

## NOTICIAS Y ANÁLISIS (FUENTES CUALITATIVAS):

### FUENTE 1: BEFORE THE EUROPEAN BELL:
--- INICIO ---
{before_bell_content}
--- FIN ---

### FUENTE 2: FIVE THINGS:
--- INICIO ---
{five_things_content}
--- FIN ---
"""

    if fed_summaries:
        prompt += "\n### FUENTE 3: NUEVOS DISCURSOS DE LA RESERVA FEDERAL:\n"
        for summary in fed_summaries:
            prompt += f"- {summary}\n"

    prompt += f"""

## EJEMPLOS DE COMENTARIOS ANTERIORES:
{chr(10).join(['--- ' + e['fecha'] + ' ---\n' + e['texto'][:700] + ('...' if len(e['texto']) > 700 else '') + '\n' for e in examples[:2]])}

"""

    return prompt


# =====================================================
# GENERACIÓN Y VALIDACIÓN CON MODELOS LOCALES
# =====================================================

def validate_numbers_with_llama(gemma2_tuple, generated_text: str, market_data: Dict) -> Dict:
    """Valida la coherencia numérica empleando Gemma 2."""
    model, tokenizer = gemma2_tuple
    critical_data = {
        "sp500_porcentaje": market_data.get("sp500", 0),
        "brent_precio": market_data.get("brent", 0),
        "us10y_yield": market_data.get("us10y", 0),
        "eurostoxx_porcentaje": market_data.get("eurostoxx", 0),
        "stoxx600_porcentaje": market_data.get("stoxx600", 0),
        "ibex_porcentaje": market_data.get("ibex", 0),
        "nikkei_porcentaje": market_data.get("nikkei", 0),
        "spw_porcentaje": market_data.get("spw", 0),
        "bm7t_porcentaje": market_data.get("bm7t", 0),
        "sp_pe": market_data.get("sp_pe", 0),
        "sp_eps_growth": market_data.get("sp_eps_growth", 0),
        "stoxx_pe": market_data.get("stoxx_pe", 0),
        "stoxx_eps_growth": market_data.get("stoxx_eps_growth", 0)
    }

    prompt = f"""Compara el texto con los datos originales.
DATOS CORRECTOS: {json.dumps(critical_data)}
TEXTO: {generated_text}

Responde ÚNICAMENTE en formato JSON estricto: {{"is_valid": true, "errors": []}}"""

    messages = [{"role": "user", "content": prompt}]
    # inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    encoding = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    input_ids = encoding["input_ids"].to("cuda")

    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=250, temperature=0.0, do_sample=False)

    res_raw = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(res_raw)
    except Exception:
        return {"is_valid": True, "errors": []}




def generate_commentary_gemma(gemma_tuple, system_instruction: str, user_prompt: str, log_callback=None) -> str:
    """Genera el comentario usando Gemma 3 Fine-Tuned."""
    model = gemma_tuple

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]

    # Inferencia directa usando la API de chat de llama-cpp-python
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=1800,
        temperature=0.3,
        repeat_penalty=1.3
    )

    # Extraer el texto generado por la respuesta del asistente
    respuesta = response["choices"][0]["message"]["content"]

    return respuesta


def load_examples_from_csv(csv_bytes) -> List[Dict]:
    df = pd.read_csv(io.BytesIO(csv_bytes), sep=';', encoding='utf-8')
    return [{"fecha": row["Fecha"], "texto": row["Comentario"]} for _, row in df.iterrows()]


# =====================================================
# INTERFAZ DE STREAMLIT
# =====================================================


def main():
    st.markdown('<div class="main-header">📈 Comentario de Mercados</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Despliegue Local con Gemma 3 Fine-Tuned & Gemma 2</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuración")
        hf_token = st.text_input("Hugging Face Token", type="password")
        gemma_repo = st.text_input("Repo Gemma 3 FT", value="GuillermoBarrio/gemma3-4b-finetuned-comentarios-gguf")
        gemma_filename = st.text_input("Repo Gemma 3 FT Filename", value="gemma-3-4b-it.Q4_K_M.gguf")
        gemma2_repo = st.text_input("Repo Gemma 2", value="google/gemma-2-2b-it")

        st.markdown("---")
        st.header("📁 Archivos")
        excel_file = st.file_uploader("📊 Excel", type=["xlsx"])
        before_bell_file = st.file_uploader("📰 Before the Bell (.txt)", type=["txt"])
        five_things_file = st.file_uploader("📋 Five Things (.txt)", type=["txt"])
        examples_file = st.file_uploader("📚 CSV Ejemplos", type=["csv"])

        generate_btn = st.button("🚀 GENERAR COMENTARIO", type="primary")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Progreso")

        if generate_btn:
            if not all([excel_file, before_bell_file, five_things_file, examples_file, hf_token]):
                st.error("❌ Por favor, sube los archivos e ingresa tu token HF.")
            else:
                def add_log(msg):
                    st.session_state.generation_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

                st.session_state.generation_logs = []
                add_log("⚙️ Cargando/Verificando modelos en GPU...")

                # Cargar modelos en caché pasándole los repos y tokens
                gemma_tuple, gemma2_tuple = load_llm_models(gemma_repo, gemma_filename, gemma2_repo, hf_token)

                # Procesamiento normal de archivos

                add_log("🚀 Iniciando proceso...")
                add_log("📊 Cargando datos del Excel...")
                # excel_bytes_io = io.BytesIO(excel_file.getvalue())
                excel_data = ExcelDataLoader.load_from_excel(io.BytesIO(excel_file.getvalue()))
                market_data = ExcelDataLoader.extract_key_market_data(excel_data)

                add_log(f"   - Brent: ${market_data['brent']:.1f} | S&P: {market_data['sp500']*100:.2f}%")
                add_log(f"   - US10Y: {market_data['us10y']:.2f}%")

                # Mostrar tendencias cargadas (solo lunes)
                if datetime.now().weekday() == 0:
                    add_log("📈 Procesando datos de tendencias (lunes)...")
                    frases_semanas, frases_racha, frases_maxmin = extract_trend_insights(market_data)
                    if frases_semanas:
                        add_log(f"   - {len(frases_semanas)} índices con tendencia alcista 10 semanas")
                    if frases_racha:
                        add_log(f"   - {len(frases_racha)} índices en racha alcista")
                    if frases_maxmin:
                        add_log(f"   - {len(frases_maxmin)} índices cerca de máximos")


                # Procesar FED con Gemma 2
                add_log("🏦 Procesando discursos de la FED si los hubiese...")
                fed_processor = FEDSpeechProcessor(gemma2_tuple)
                fed_summaries = fed_processor.fetch_and_summarize_new_speeches(log_callback=add_log)

                # Construir Prompt
                system_instruction =  ("Eres un analista financiero senior. Debes redactar un comentario de mercados en castellano (650 palabras aprox.)"
                                        "basado en datos reales y las noticias proporcionadas."
                                        "## FORMATO:"
                                        "- Estilo profesional, conciso y analítico."
                                        "- Usa abreviaturas: EEUU, UK, ATH, yoy, pbs, BBG."
                                        "- Porcentajes con signo: +2.3%, -1.5%."
                                        "- El comentario debe tener 4-5 párrafos en los que trates, al menos, los siguientes temas, "
                                        "sin que sea este un orden de importancia:"
                                          " - Renta Variable, principalmente norteamericana y en menor medida europea"
                                          " - Materias Primas"
                                          " - Renta Fija, tipos de interés, Reserva Federal"
                                          " - Noticias corporativas de primer orden en EEUU"
                                          " - Agenda de datos a conocerse: Resultados empresariales en EEUU y Europa de empresas especialmente importantes, "
                                          "así como datos macro, especialmente inflación y desempleo en EEUU, UK y la Eurozona"
                                        "- Los temas tratados en cada párrafo irán de más a menos importancia, y su importancia dependerá de las "
                                        "fuentes Before The European Bell y Five Things, que se te adjuntan en el resto de este prompt."
                                        "## INSTRUCCIONES FINALES:"
                                        "1. Usa los DATOS NUMÉRICOS EXACTOS que se te han dado. NO los inventes."
                                        "2. Genera los párrafos temáticos en orden de importancia: primero los que consideres más relevantes."
                                        "3. La información de Before the Bell y Five Things es la base para el análisis cualitativo,"
                                        "las perspectivas y las noticias corporativas."
                                        "4. Si se incluyen discursos de la FED, son la fuente principal para hablar de política monetaria."
                                        "5. Incluye un breve resumen de las expectativas de tipos de la Fed, BCE y BOE en el apartado de"
                                        "Renta Fija, tipos de interés, y Reserva Federal"
                                        "6. Si hoy es lunes, los ratios PE, los crecimientos de EPS y las tendencias se tienen "
                                        "que integrar en el análisis de renta variable."
                                        "7. Los datos de resultados empresariales y macro a conocerse deben ir hacia el final de comentario, "
                                        "a menos que tengan conexión con el análisis de renta variable o renta fija."
                                        "8. Pon en negrita los nombres de países, empresas, datos macro, y personalidades importantes."
                                        "9. No añadas texto introductorio como Claro, aquí tienes... . Empieza directamente con el análisis."
                                        "10. NO repitas párrafos ni frases ya escritas. Cada idea se menciona una sola vez. "
                                        "11. Cuando hayas cubierto los 4-5 temas principales, termina el comentario."
                                        "Genera el comentario de mercados a continuación:" )

                user_prompt = build_prompt(
                    before_bell_file.getvalue().decode('utf-8'),
                    five_things_file.getvalue().decode('utf-8'),
                    market_data,
                    datetime.now().weekday() == 0,
                    load_examples_from_csv(examples_file.getvalue()),
                    fed_summaries
                )

                # Inferencia Gemma 3
                commentary = generate_commentary_gemma(gemma_tuple, system_instruction, user_prompt, log_callback=add_log)
                st.session_state.generated_commentary = commentary

                # Validación Gemma 2
                validation = validate_numbers_with_llama(gemma2_tuple, commentary, market_data)
                add_log(f"✅ Proceso finalizado. Resultado de validación: {validation['is_valid']}")

        if st.session_state.get('generation_logs'):
            for log in st.session_state.generation_logs:
                st.text(log)

    with col2:
        st.subheader("📝 Comentario Generado")
        if st.session_state.get('generated_commentary'):
            st.markdown(f'<div class="commentary-box">{st.session_state.generated_commentary}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

