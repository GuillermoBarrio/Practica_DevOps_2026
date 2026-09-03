

# 📈 AutoBloomberg — Comentario Automático de Mercados

[![Deploy to Cloud Run](https://github.com/GuillermoBarrio/Practica_DevOps_2026/actions/workflows/deploy.yml/badge.svg)](https://github.com/GuillermoBarrio/Practica_DevOps_2026/actions/workflows/deploy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

Proyecto de fin de curso de **LLMOps** que implementa una aplicación de generación automática de comentarios financieros de mercado, desplegada en **Google Cloud Run** con CI/CD mediante **GitHub Actions**.

🚀 **App en producción:** [https://autobloomberg-yn34vb66rq-ez.a.run.app](https://autobloomberg-yn34vb66rq-ez.a.run.app)

**Vídeo explicativo de la app:** [https://www.youtube.com/watch?v=Aei2hv44NdA](https://www.youtube.com/watch?v=Aei2hv44NdA)

---

## 🏗️ Arquitectura

Excel con datos Before the Bell Five Things
de mercado (Bloomberg) (Bloomberg)
│ │ │
└────────────────────────┴──────────────────────┘
│
┌──────────▼──────────┐
│ Agente FED │
│ ┌───────────────┐ │
│ │ MCP Fetch │ │ ← Discursos oficiales
│ │ (RSS Fed) │ │ federalreserve.gov
│ └──────┬────────┘ │
│ │ Sin │
│ ▼ discursos │
│ ┌───────────────┐ │
│ │ Google Search │ │ ← Fallback: noticias
│ │ Grounding │ │ últimas 24h
│ └───────────────┘ │
└──────────┬──────────┘
│
┌──────────▼──────────┐
│ Gemini Flash │ ← Generación comentario
│ (Google AI) │ ~650 palabras
└──────────┬──────────┘
│
┌──────────▼──────────┐
│ LangSmith │ ← Trazabilidad completa
└─────────────────────┘


---

## ✨ Características principales

- **Generación automática** de comentarios financieros diarios en castellano (~650 palabras)
- **Agente FED con lógica de fallback**: busca primero discursos oficiales de la Reserva Federal vía MCP Fetch; si no los hay, activa búsqueda web con Google Search Grounding
- **MCP Fetch** para scraping robusto del RSS de la Fed (`federalreserve.gov`), convirtiendo HTML a Markdown sin dependencia de selectores DOM
- **Validación numérica** de los datos de mercado incluidos en el comentario generado
- **Trazabilidad completa** con LangSmith: jerarquía de runs con pipeline → agente → LLM
- **CI/CD automático** con GitHub Actions: cada push a `main` despliega en Cloud Run
- **Interfaz web** con Streamlit, accesible públicamente

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM principal | Gemini 3.5 Flash (Google AI) |
| Agente y orquestación | Python + `@traceable` (LangSmith) |
| Scraping FED | MCP Fetch (`mcp-server-fetch`) |
| Búsqueda web fallback | Google Search Grounding (Gemini) |
| Trazabilidad | LangSmith |
| Interfaz | Streamlit |
| Contenedor | Docker (python:3.11-slim) |
| Despliegue | Google Cloud Run (europe-west4) |
| CI/CD | GitHub Actions + Workload Identity Federation |
| Secretos | Google Cloud Secret Manager |

---

## 📁 Estructura del repositorio

Practica_DevOps_2026/
│
├── app-comentario.py # App principal (Cloud Run + Gemini)
├── Dockerfile # Imagen Docker para Cloud Run
├── requirements.txt # Dependencias Python
├── Comentario_Mercados_Pedro_csv.csv # Ejemplos de comentarios (few-shot)
├── Proyecto_LLMOps_IA_Memoria.pdf # Memoria del proyecto
│
├── .github/
│ └── workflows/
│ └── deploy.yml # Pipeline CI/CD → Cloud Run
│
├── Ejemplo_Ficheros_Necesarios/ # Ficheros de ejemplo para ejecutar la app
│ └── ... # Excel de mercado, Before the Bell, Five Things
│
├── Ollama_Main_Files/ # App con modelos locales (Gemma 3 + Gemma 2)
│ └── app_modelos_locales.py
│
├── Ollama_Datasets/ # Dataset de fine-tuning (18 pares prompt/comentario)
│ └── ...
│
├── Ollama_Notebooks/ # Notebooks de fine-tuning Gemma 3 4B con Unsloth
│ └── ...
│
└── Ollama_Ragas_Files/ # Evaluación del modelo con RAGAS + DeepSeek
└── ...


---

## 🚀 Despliegue en Cloud Run

El despliegue es completamente automático mediante GitHub Actions. Cada push a `main` ejecuta el pipeline definido en `.github/workflows/deploy.yml`, que:

1. Autentica con Google Cloud vía **Workload Identity Federation** (sin claves de servicio)
2. Construye la imagen Docker directamente en Cloud Run (`--source .`)
3. Despliega el servicio con los secretos inyectados desde **Secret Manager**

Las variables de entorno y secretos necesarios son:
- `GEMINI_API_KEY` — API key de Google AI
- `DEEPSEEK_API_KEY` — API key de DeepSeek (juez en evaluación RAGAS)
- `LANGCHAIN_API_KEY` — API key de LangSmith

---

## 💻 Ejecución local (app Cloud Run)

```bash
# 1. Clonar el repositorio
git clone https://github.com/GuillermoBarrio/Practica_DevOps_2026.git
cd Practica_DevOps_2026

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
export GEMINI_API_KEY="tu_api_key"
export LANGCHAIN_API_KEY="tu_api_key"

# 5. Ejecutar
streamlit run app-comentario.py
```

Los ficheros de entrada necesarios (Excel, Before the Bell, Five Things) están disponibles como ejemplos en la carpeta `Ejemplo_Ficheros_Necesarios/`.

---

## 🤖 Modelos locales (Gemma 3 Fine-Tuned)

Como parte complementaria del proyecto se realizó un **fine-tuning de Gemma 3 4B** con LoRA/Unsloth para generar comentarios sin depender de APIs externas.

El modelo fine-tuneado está disponible en Hugging Face:
**[GuillermoBarrio/gemma3-4b-finetuned-comentarios-gguf](https://huggingface.co/GuillermoBarrio/gemma3-4b-finetuned-comentarios-gguf)**

Para ejecutar la app con modelos locales (requiere GPU):

```bash
# Instalar dependencias adicionales
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Ejecutar
streamlit run Ollama_Main_Files/app_modelos_locales.py
```

### Evaluación con RAGAS

Se evaluó el modelo fine-tuneado con **RAGAS** usando DeepSeek Chat como juez. Resultados principales sobre 7 ejemplos:

| Métrica | Puntuación |
|---|---|
| `context_precision` | 1.00 ✅ |
| `context_recall` | 0.92 ✅ |
| `faithfulness` | 0.46 ⚠️ |
| `answer_relevancy` | 0.44 ⚠️ |

La `faithfulness` moderada refleja overfitting por dataset reducido (18 ejemplos, 14 épocas). La siguiente iteración incrementará el dataset y reducirá las épocas de entrenamiento.

---


## 📊 Trazabilidad con LangSmith

Cada ejecución genera una traza completa en LangSmith con la siguiente jerarquía:

Pipeline comentario financiero (chain)
├── Agente de discursos o noticias Fed (agent)
│ ├── MCP Fetch → RSS federalreserve.gov
│ └── Google Search Grounding (fallback)
└── generate_commentary (llm)
└── Gemini 3.5 Flash

---

## 📝 Licencia

Proyecto académico desarrollado como práctica del curso de LLMOps 2026.