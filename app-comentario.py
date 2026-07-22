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
from google import genai
import warnings
import tempfile
import os
from google.genai.types import GenerateContentConfig

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Comentario de Mercados - AutoBloomberg",
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



# =====================================================
# CLASES Y FUNCIONES
# =====================================================

class FEDSpeechProcessor:
    """Procesa discursos de la Reserva Federal desde su feed RSS oficial."""

    def __init__(self, genai_client, state_file=None):
        self.client = genai_client
        self.feed_url = "https://www.federalreserve.gov/feeds/speeches.xml"
        
        # Usar el directorio temporal del sistema operativo (/tmp en Cloud Run)
        if state_file is None:
            self.state_file = os.path.join(tempfile.gettempdir(), 'fed_speeches_processed.txt')
        else:
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
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S',
            '%d %b %Y %H:%M:%S',
        ]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                return dt.date()
            except ValueError:
                continue
        return None

    def _get_target_date(self) -> datetime.date:
        """Determina el día objetivo: día anterior, pero si es lunes, busca viernes"""
        today = datetime.now().date()
        if today.weekday() == 0:  # Lunes
            return today - timedelta(days=3)
        else:
            return today - timedelta(days=1)

    def _extract_speech_content(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            article_containers = soup.find_all('div')

            for div in article_containers:
              clases = div.get('class', [])

              # Verificamos si tiene las 3 clases que quieres Y NO tiene 'heading'
              if 'col-xs-12' in clases and 'col-sm-8' in clases and 'col-md-8' in clases and 'heading' not in clases:
                print("¡Encontrado el div correcto!")
                article_container = div
                break
            else:
                print("No se encontró el div correcto.")
                article_container = None



            # article_container = soup.find('div', class_=re.compile(r'col-xs-12 col-sm-8 col-md-8'))
            # article_containers = soup.select('div.col-xs-12.col-sm-8.col-md-8:not(.heading)')
            # article_containers = soup.find('div', class_=['col-xs-12', 'col-sm-8', 'col-md-8'])
            # article_container = soup.select('div.col-xs-12.col-sm-8.col-md-8')
            if not article_container:
                article_container = soup.find('div', class_=re.compile(r'col-(xs|sm|md)-\d+'))
            if article_container:
                paragraphs = article_container.find_all('p')
                print('paragraphs')
                print(paragraphs)
                if paragraphs:
                    speech_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                    speech_text = re.sub(r'\s+', ' ', speech_text).strip()
                    print('text')
                    print(speech_text)
                    return speech_text if len(speech_text) > 400 else None
            return None
        except Exception:
            return None

    def _summarize_speech(self, title: str, content: str) -> Optional[str]:
        max_chars = 6000
        truncated_content = content[:max_chars]
        prompt = f"""Eres un analista experto en la Reserva Federal.
Resume el siguiente discurso en un máximo de 150 palabras en castellano.

PRIORIZA en este orden:
1. Política monetaria (decisiones o pistas sobre tipos de interés)
2. Inflación (perspectivas y riesgos)
3. Mercado laboral (situación y perspectivas)
4. Perspectivas económicas generales

Ignora agradecimientos y formalidades. Sé directo y profesional.

Título: {title}

Discurso:
{truncated_content}

Resumen (150 palabras máximo):"""
        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=350,
                    temperature=0.3,
                )
            )
            summary = response.text.strip()
            return f"**{title}**: {summary}"
        except Exception as e:

            print(f"❌ ERROR REAL EN FED SPEECH: {str(e)}") # Esto saldrá en tus logs de Google Cloud Run
            return None

    def fetch_and_summarize_new_speeches(self, log_callback=None) -> List[str]:
        """Devuelve lista de resúmenes de discursos nuevos"""
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
                if not speech_date or speech_date != target_date:
                    continue
                if entry.link in self.processed_urls:
                    continue
                if log_callback:
                    log_callback(f"   - Nuevo discurso: {entry.title}")
                speech_content = self._extract_speech_content(entry.link)
                if speech_content:
                    summary = self._summarize_speech(entry.title, speech_content)
                    if summary:
                        summaries.append(summary)
                        self._save_processed_url(entry.link)
                        if log_callback:
                            log_callback(f"      - Resumen generado")

                print('Speeches *****')
                print(summaries)
                print('*****')
            return summaries
        except Exception as e:
            if log_callback:
                log_callback(f"   - Error: {e}")
            return []


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
    strong_down_sorted = sorted(strong_up, key=lambda x: x[1], reverse=True)[:max_items]

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

    before_bell_excerpt = before_bell_content[:800]

    prompt = f"""Eres un analista financiero senior. Debes redactar un comentario de mercados en castellano (550 palabras aprox.) basado en datos reales y las noticias proporcionadas.

## FORMATO:
- Estilo profesional, conciso y analítico.
- Usa abreviaturas: EEUU, ATH, yoy, pbs, BBG.
- Porcentajes con signo: +2.3%, -1.5%.
- El comentario debe tener 3-4 párrafos en los que trates, al menos, los siguientes temas, sin que sea este un orden de importancia:
    - Renta Variable, principalmente norteamericana y en menor medida europea
    - Materias Primas
    - Renta Fija, tipos de interés, Reserva Federal
    - Noticias corporativas de primer orden en EEUU

- Los temas tratados en cada párrafo irán de más a menos importancia, y su importancia dependerá de las fuentes Before The European Bell y Five Things, que se te adjuntan en el resto de este prompt.

## DATOS NUMÉRICOS DE MERCADO (USA ESTOS VALORES EXACTOS):
### RENTA VARIABLE (% día anterior):
- S&P500: {market_data['sp500']*100:.2f}%
- S&P Equal Weight (SPW): {market_data['spw']*100:.2f}%
- 7 Magníficas (BM7T): {market_data['bm7t']*100:.2f}%
- EuroStoxx50: {market_data['eurostoxx']*100:.2f}%
- Stoxx600: {market_data['stoxx600']*100:.2f}%
- Ibex35: {market_data['ibex']*100:.2f}%
- Nikkei: {market_data['nikkei']*100:.2f}%

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

## NOTICIAS Y ANÁLISIS (FUENTES CUALITATIVAS):

### FUENTE 1: BEFORE THE EUROPEAN BELL:
--- INICIO ---
{before_bell_excerpt}
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


    # Sacamos la lógica de formato fuera del f-string para compatibilidad con Python 3.11
    ejemplos_formateados = "\n".join(
        [f"--- {e['fecha']} ---\n{e['texto'][:700]}{'...' if len(e['texto']) > 700 else ''}\n" for e in examples[:2]]
    )

    prompt += f"""

## EJEMPLOS DE COMENTARIOS ANTERIORES:

{ejemplos_formateados}

## INSTRUCCIONES FINALES:
1. Hoy es {datetime.now().strftime('%d/%m/%Y')}.
2. Es imperativo que el análisis sea extenso y desarrolle todos los puntos, con una longitud cercana a las 550 palabras. No dejes el texto inacabado.
3. Usa los DATOS NUMÉRICOS EXACTOS que se te han dado. NO los inventes.
4. Genera los párrafos temáticos en orden de importancia: primero los que consideres más relevantes.
5. La información de "Before the Bell" y "Five Things" es la base para el análisis cualitativo, las perspectivas y las noticias corporativas.
6. Si se incluyen discursos de la FED, son la fuente principal para hablar de política monetaria.
7. Si hoy es lunes, los ratios PE, los crecimientos de EPS y las tendencias se tienen que integrar en el análisis de renta variable.
8. No añadas texto introductorio como "Claro, aquí tienes...". Empieza directamente con el análisis.

Genera el comentario de mercados a continuación:"""

    return prompt



# =====================================================
# 6. VALIDACIÓN NUMÉRICA (CON GEMINI 3.5 FLASH)
# =====================================================

def validate_numbers_with_llm(client, generated_text: str, market_data: Dict) -> Dict:
    """Usa Gemini 3.5 Flash para verificar que los números clave son correctos."""

    # Seleccionar los datos más críticos para la validación
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


    # Escapamos/limpiamos el texto para no romper el prompt del  JSON
    clean_generated_text = generated_text.replace('"', "'")
    safe_text_json = json.dumps(generated_text, ensure_ascii=False)



    validation_prompt = f"""Eres un verificador de datos. Compara el texto generado con los datos originales.

DATOS ORIGINALES CORRECTOS:
{json.dumps(critical_data, indent=2, ensure_ascii=False)}

TEXTO GENERADO:

"{safe_text_json}"

Reglas:
- Tolerancia: ±0.05 para precios, ±0.1% para porcentajes.
- Si un número no aparece en el texto, no lo consideres un error.
- Solo reporta discrepancias claras y significativas.

RESPONDE ÚNICAMENTE CON UN JSON EN ESTE FORMATO:
{{"is_valid": true, "errors": []}}
Donde "errors" es una lista de strings describiendo cada error encontrado."""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=validation_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
            )
        )
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(raw_text)
        return result
    except Exception as e:
        print(f"   - Error en el validador de números: {e}")
        return {"is_valid": True, "errors": []}





# =====================================================
# 7. GENERACIÓN DEL COMENTARIO (CON GEMINI 3.5 FLASH)
# =====================================================


def generate_commentary(client, before_bell, five_things, market_data, examples, fed_summaries, log_callback=None):
    """Genera el comentario usando Gemini 3.5 Flash"""
    is_monday = datetime.now().weekday() == 0
    prompt = build_prompt(before_bell, five_things, market_data, is_monday, examples, fed_summaries)

    if log_callback:
        log_callback("🤖 Generando comentario con Gemini 3.5 Flash...")

    try:

	# Configuración de seguridad para evitar cortes por noticias geopolíticas/bélicas
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]


	config_setup = types.GenerateContentConfig(
    		system_instruction=(
        	"Eres un analista financiero senior. Redactas comentarios de mercado detallados y fluidos en castellano. "
        	"IMPORTANTE: Sé conciso y directo en el análisis para no exceder los límites. Mantén tu respuesta "
        	"estrictamente alrededor de las 550 palabras y concluye siempre con un párrafo de cierre claro."
   		 ),
    		temperature=1.0, 
    		# 1. REMOVE or DRAMATICALLY INCREASE the token cap
    		max_output_tokens=8192, 
    		# 2. SEPARATE thinking tokens from your final visible text budget
    		thinking_config=types.ThinkingConfig(thinking_budget=2048),
    		safety_settings=safety_settings,
	)



        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=config_setup,  # <-- Pasamos el diccionario directo
        )


	# Inspección del motivo de finalización
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if log_callback:
                log_callback(f"ℹ️ Motivo de finalización del LLM: {finish_reason}")
            print(f"DEBUG Finish Reason: {finish_reason}")

        generated_text = response.text

        # validation = validate_numbers_with_llm(client, generated_text, market_data)

        if log_callback:
            log_callback(f"✅ Comentario generado ({len(generated_text)} caracteres)")

        # validation = validate_numbers_with_llm(client, generated_text, market_data)

        if log_callback:
          log_callback(f"✅ Validación en marcha!")

        validation = validate_numbers_with_llm(client, generated_text, market_data)


        return {
            "comentario": generated_text,
            "validation": validation
        }


        # return generated_text


    except Exception as e:
        if log_callback:
            log_callback(f"❌ Error: {e}")

        print(f"❌ ERROR REAL EN GENERATE: {str(e)}") # Esto saldrá en tus logs de Google Cloud Run
        return None


def load_examples_from_csv(csv_bytes) -> List[Dict]:
    df = pd.read_csv(io.BytesIO(csv_bytes), sep=';', encoding='utf-8')
    return [{"fecha": row["Fecha"], "texto": row["Comentario"]} for _, row in df.iterrows()]


# =====================================================
# INTERFAZ DE STREAMLIT
# =====================================================

def main():
    if 'generated_commentary' not in st.session_state:
        st.session_state.generated_commentary = None
    if 'generation_logs' not in st.session_state:
        st.session_state.generation_logs = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Header
    # st.markdown('<div class="main-header">📈 Comentario de Mercados</div>', unsafe_allow_html=True)
    # st.markdown('<div class="sub-header">Generación automática con IA (Gemini 2.5 Pro)</div>', unsafe_allow_html=True)


    st.markdown('<div class="main-header">📈 Comentario de Mercados</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generación automática con IA (Gemini 2.5 Pro)</div>', unsafe_allow_html=True)



    # Sidebar: Configuración
    with st.sidebar:
        # st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Bloomberg_Logo.svg/256px-Bloomberg_Logo.svg.png", use_container_width=True)
        st.markdown("---")
        st.header("⚙️ Configuración")

        # api_key = st.text_input("Gemini API Key", type="password",
        #                     help="Introduce tu API key de Google AI Studio")
        
        api_key = os.environ.get("GEMINI_API_KEY", st.text_input("Gemini API Key", type="password", help="Introduce tu API key de Google AI Studio"))

        st.markdown("---")
        st.header("📁 Archivos")
        st.info("Sube los archivos necesarios:")

        excel_file = st.file_uploader("📊 Excel (Enlace_Diario_Pedro_valores.xlsx)", type=["xlsx"])
        before_bell_file = st.file_uploader("📰 Before the European Bell (.txt)", type=["txt"])
        five_things_file = st.file_uploader("📋 Five Things (.txt)", type=["txt"])
        examples_file = st.file_uploader("📚 CSV Ejemplos (Comentario_Mercados_Pedro_csv.csv)", type=["csv"])

        st.markdown("---")
        generate_btn = st.button("🚀 GENERAR COMENTARIO", use_container_width=True, type="primary")

    # Área principal
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Progreso")

        if generate_btn and not api_key:
            st.error("❌ Por favor, introduce tu API key de Gemini")
        elif generate_btn:
            if not all([excel_file, before_bell_file, five_things_file, examples_file]):
                st.error("❌ Por favor, sube todos los archivos necesarios")
            else:
                st.session_state.processing = True
                st.session_state.generation_logs = []

                # Función para capturar logs
                def add_log(msg):
                    st.session_state.generation_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

                try:
                    add_log("🚀 Iniciando proceso...")
                    add_log("📊 Cargando datos del Excel...")
                    client = genai.Client(api_key=api_key)


                    # Cargar datos
                    excel_bytes_io = io.BytesIO(excel_file.getvalue())
                    excel_data = ExcelDataLoader.load_from_excel(excel_bytes_io)
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

                    add_log("📚 Cargando ejemplos...")
                    examples = load_examples_from_csv(examples_file.getvalue())
                    add_log(f"   - {len(examples)} ejemplos cargados")

                    # Procesar FED
                    add_log("🏦 Procesando discursos de la FED si los hubiese...")
                    fed_processor = FEDSpeechProcessor(client)
                    fed_summaries = fed_processor.fetch_and_summarize_new_speeches(log_callback=add_log)

                    # Generar comentario
                    before_bell_content = before_bell_file.getvalue().decode('utf-8')
                    five_things_content = five_things_file.getvalue().decode('utf-8')

                    add_log("🤖 Generando comentario...")
                    commentary = generate_commentary(
                        client, before_bell_content, five_things_content,
                        market_data, examples, fed_summaries, log_callback=add_log
                    )

                    if commentary:
                        st.session_state.generated_commentary = commentary['comentario']

                        print(f"Validación: {commentary['validation']}")
                        if commentary['validation']['is_valid']:
                            add_log("✅ Comentario válido")
                        else:
                            add_log("❌ Comentario inválido por errores de validación")
                            add_log(f"   - Errores: {commentary['validation']['errors']}")
                    else:
                        add_log("❌ Error en la generación del comentario")

                except Exception as e:
                    add_log(f"❌ Error: {str(e)}")

                st.session_state.processing = False

        # Mostrar logs
        with st.container():
            if st.session_state.generation_logs:
                for log in st.session_state.generation_logs:
                    st.text(log)
            elif not st.session_state.processing:
                st.info("Listo para generar. Sube los archivos y haz clic en 'GENERAR COMENTARIO'")

    with col2:
        st.subheader("📝 Comentario Generado")

        if st.session_state.generated_commentary:
            st.markdown(f'<div class="commentary-box">{st.session_state.generated_commentary}</div>',
                       unsafe_allow_html=True)

            # Botones de acción
            col_download, col_copy = st.columns(2)
            with col_download:
                st.download_button(
                    label="📥 Descargar .txt",
                    data=st.session_state.generated_commentary,
                    file_name=f"comentario_mercados_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
            with col_copy:
                if st.button("📋 Copiar al portapapeles"):
                    st.write("✅ Copiado (selecciona manualmente)")
        else:
            st.info("El comentario generado aparecerá aquí...")


if __name__ == "__main__":
    main()
