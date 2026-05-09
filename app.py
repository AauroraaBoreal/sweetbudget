import json
from datetime import datetime

from datetime import datetime
from pymongo import MongoClient

import pandas as pd
import requests
import streamlit as st

from google import genai
from google.genai import types

import psycopg2
from psycopg2.extras import Json

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="SweetBudget",
    page_icon="🍰",
    layout="wide"
)

MODELO_GEMINI = "gemini-2.5-flash"


# =====================================================
# CLIENTES Y CONEXIONES
# =====================================================

def obtener_cliente_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.error("Falta GEMINI_API_KEY en .streamlit/secrets.toml")
        st.stop()

    return genai.Client(api_key=api_key)


def obtener_conexion_postgres():
    config = st.secrets["postgres"]

    conexion = psycopg2.connect(
        user=config["USER"],
        password=config["PASSWORD"],
        host=config["HOST"],
        port=config["PORT"],
        dbname=config["DBNAME"],
        sslmode="require"
    )

    return conexion


def obtener_cliente_mongo():
    mongo_uri = st.secrets.get("MONGO_URI", "")

    if not mongo_uri or MongoClient is None:
        return None

    return MongoClient(mongo_uri)


# =====================================================
# BASE DE DATOS SUPABASE POSTGRESQL
# =====================================================

def crear_tabla_recetas():
    conexion = obtener_conexion_postgres()
    cursor = conexion.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recetas_postres (
            id BIGSERIAL PRIMARY KEY,
            busqueda TEXT,
            nombre_postre TEXT,
            fuente TEXT,
            link TEXT,
            ingredientes_probables JSONB,
            costo_estimado_soles NUMERIC,
            rango_precio TEXT,
            dificultad TEXT,
            recomendacion TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    conexion.commit()
    cursor.close()
    conexion.close()


def guardar_en_postgres(postres, busqueda):
    conexion = obtener_conexion_postgres()
    cursor = conexion.cursor()

    for postre in postres:
        cursor.execute("""
            INSERT INTO recetas_postres (
                busqueda,
                nombre_postre,
                fuente,
                link,
                ingredientes_probables,
                costo_estimado_soles,
                rango_precio,
                dificultad,
                recomendacion
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            busqueda,
            postre.get("nombre_postre", ""),
            postre.get("fuente", ""),
            postre.get("link", ""),
            Json(postre.get("ingredientes_probables", [])),
            postre.get("costo_estimado_soles", 0),
            postre.get("rango_precio", ""),
            postre.get("dificultad", ""),
            postre.get("recomendacion", "")
        ))

    conexion.commit()
    cursor.close()
    conexion.close()


def obtener_historial():
    conexion = obtener_conexion_postgres()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT 
            id,
            busqueda,
            nombre_postre,
            fuente,
            link,
            ingredientes_probables,
            costo_estimado_soles,
            rango_precio,
            dificultad,
            recomendacion,
            created_at
        FROM recetas_postres
        ORDER BY created_at DESC
        LIMIT 50;
    """)

    columnas = [
        "id",
        "busqueda",
        "nombre_postre",
        "fuente",
        "link",
        "ingredientes_probables",
        "costo_estimado_soles",
        "rango_precio",
        "dificultad",
        "recomendacion",
        "created_at"
    ]

    filas = cursor.fetchall()

    cursor.close()
    conexion.close()

    return pd.DataFrame(filas, columns=columnas)


# =====================================================
# MONGODB ATLAS OPCIONAL
# =====================================================

def guardar_busqueda_raw_en_mongo(busqueda, resultados_web):
    cliente = obtener_cliente_mongo()

    if cliente is None:
        return False

    db_name = st.secrets.get("MONGO_DB", "sweetbudget")
    collection_name = st.secrets.get("MONGO_COLLECTION", "busquedas_raw")

    db = cliente[db_name]
    collection = db[collection_name]

    documento = {
        "busqueda": busqueda,
        "resultados_web": resultados_web,
        "fecha": datetime.utcnow()
    }

    collection.insert_one(documento)
    cliente.close()

    return True


# =====================================================
# SERPER: BÚSQUEDA EN GOOGLE
# =====================================================

def buscar_recetas_con_serper(consulta, cantidad_resultados):
    serper_key = st.secrets.get("SERPER_API_KEY", "")

    if not serper_key:
        st.error("Falta SERPER_API_KEY en .streamlit/secrets.toml")
        st.stop()

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": f"recetas de postres {consulta} ingredientes",
        "gl": "pe",
        "hl": "es",
        "num": cantidad_resultados
    }

    response = requests.post(url, headers=headers, json=payload, timeout=20)

    if response.status_code != 200:
        st.error("Error al buscar con Serper.dev")
        st.code(response.text)
        return []

    data = response.json()

    resultados = []

    for item in data.get("organic", []):
        resultados.append({
            "titulo": item.get("title", ""),
            "link": item.get("link", ""),
            "descripcion": item.get("snippet", "")
        })

    return resultados


# =====================================================
# GEMINI: ANÁLISIS Y CLASIFICACIÓN
# =====================================================

def limpiar_json_respuesta(texto):
    texto = texto.strip()

    if texto.startswith("```json"):
        texto = texto.replace("```json", "").replace("```", "").strip()
    elif texto.startswith("```"):
        texto = texto.replace("```", "").strip()

    return texto


def analizar_postres_con_gemini(consulta, resultados_web):
    client = obtener_cliente_gemini()

    contexto_web = json.dumps(resultados_web, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un asistente experto en recetas de postres, costos de ingredientes y precios aproximados en Perú.

    El usuario buscó:
    {consulta}

    Estos son resultados obtenidos desde Google mediante Serper.dev:
    {contexto_web}

    Tu tarea:
    1. Identificar posibles recetas de postres a partir de los resultados.
    2. Estimar ingredientes probables.
    3. Calcular un costo aproximado de preparación en soles peruanos.
    4. Clasificar cada postre por rango de precio.
    5. Dar una recomendación útil para venta o preparación.

    Usa estos rangos:
    - Económico: S/ 0 a S/ 15
    - Medio: S/ 16 a S/ 35
    - Premium: S/ 36 a más

    Devuelve únicamente JSON válido.
    No agregues texto fuera del JSON.

    Estructura exacta:

    {{
      "postres": [
        {{
          "nombre_postre": "Brownie de chocolate",
          "fuente": "Nombre de la página o resultado",
          "link": "https://...",
          "ingredientes_probables": ["harina", "cacao", "azúcar", "huevo"],
          "costo_estimado_soles": 18,
          "rango_precio": "Medio",
          "dificultad": "Fácil",
          "recomendacion": "Ideal para vender en porciones pequeñas."
        }}
      ]
    }}
    """

    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    texto_limpio = limpiar_json_respuesta(response.text)

    try:
        data = json.loads(texto_limpio)
        return data.get("postres", [])
    except json.JSONDecodeError:
        st.error("Gemini respondió, pero no se pudo leer el JSON.")
        st.code(response.text)
        return []


# =====================================================
# GRÁFICO DEL STACK TECNOLÓGICO
# =====================================================

def mostrar_stack_tecnologico():
    st.subheader("Stack tecnológico usado")

    st.graphviz_chart("""
        digraph {
            rankdir=LR;

            Usuario [shape=box, style=filled, fillcolor="#FFF2CC"];
            Streamlit [label="Streamlit Cloud\\nInterfaz web + URL pública", shape=box, style=filled, fillcolor="#D9EAD3"];
            Serper [label="Serper.dev\\nBúsqueda en Google", shape=box, style=filled, fillcolor="#D9EAF7"];
            Gemini [label="Gemini API\\nAnálisis con IA", shape=box, style=filled, fillcolor="#EADCF8"];
            Supabase [label="Supabase PostgreSQL\\nBase de datos", shape=box, style=filled, fillcolor="#D9EAD3"];
            Mongo [label="MongoDB Atlas\\nBúsqueda cruda opcional", shape=box, style=filled, fillcolor="#E2F0D9"];
            Github [label="GitHub\\nRepositorio", shape=box, style=filled, fillcolor="#F4CCCC"];

            Usuario -> Streamlit;
            Streamlit -> Serper;
            Serper -> Gemini;
            Gemini -> Supabase;
            Serper -> Mongo;
            Supabase -> Streamlit;
            Github -> Streamlit;
        }
    """)

    st.markdown("""
    **Descripción del flujo:**

    1. El usuario ingresa una búsqueda de postres en Streamlit.
    2. Streamlit envía la consulta a Serper.dev.
    3. Serper.dev obtiene resultados de Google.
    4. Gemini analiza los resultados y estima el costo del postre.
    5. Supabase PostgreSQL guarda los resultados estructurados.
    6. MongoDB Atlas puede guardar la búsqueda cruda como respaldo.
    7. Streamlit muestra la tabla, gráficos e historial.
    """)

def obtener_cliente_mongo():
    mongo_uri = st.secrets.get("MONGO_URI", "")

    if not mongo_uri:
        return None

    return MongoClient(mongo_uri)


def guardar_busqueda_raw_en_mongo(busqueda, resultados_web):
    cliente = obtener_cliente_mongo()

    if cliente is None:
        st.warning("MongoDB no está configurado.")
        return False

    db_name = st.secrets.get("MONGO_DB", "sweetbudget")
    collection_name = st.secrets.get("MONGO_COLLECTION", "busquedas_raw")

    db = cliente[db_name]
    collection = db[collection_name]

    documento = {
        "busqueda": busqueda,
        "resultados_web": resultados_web,
        "fecha": datetime.utcnow()
    }

    collection.insert_one(documento)
    cliente.close()

    return True


# =====================================================
# INTERFAZ STREAMLIT
# =====================================================

st.title("🍰 SweetBudget")
st.caption("Buscador inteligente de recetas de postres por rango de precio")

try:
    crear_tabla_recetas()
except Exception as e:
    st.warning("Todavía no se pudo conectar a Supabase PostgreSQL.")
    st.caption("Revisa tus datos en .streamlit/secrets.toml")
    st.exception(e)


tab_busqueda, tab_historial, tab_stack = st.tabs([
    "🔎 Buscar postres",
    "📋 Historial",
    "🧩 Stack tecnológico"
])


with tab_busqueda:
    st.subheader("Buscar recetas de postres")

    consulta = st.text_input(
        "¿Qué tipo de postre quieres buscar?",
        placeholder="Ejemplo: postres con chocolate económicos"
    )

    cantidad_resultados = st.slider(
        "Cantidad de resultados a buscar",
        min_value=3,
        max_value=10,
        value=5
    )

    guardar_mongo = st.checkbox(
        "Guardar búsqueda cruda en MongoDB Atlas si está configurado",
        value=False
    )

    boton_buscar = st.button("Buscar y analizar", type="primary")

    if boton_buscar:
        if not consulta.strip():
            st.warning("Primero escribe qué postre quieres buscar.")
        else:
            with st.spinner("Buscando recetas en Google con Serper.dev..."):
                resultados_web = buscar_recetas_con_serper(
                    consulta,
                    cantidad_resultados
                )
            try:
                guardado_mongo = guardar_busqueda_raw_en_mongo(
                    consulta,
                    resultados_web
                )

                if guardado_mongo:
                    st.success("Búsqueda cruda guardada en MongoDB Atlas.")

            except Exception as e:
                st.warning("No se pudo guardar en MongoDB Atlas.")
                st.exception(e)

            if resultados_web:
                st.success("Resultados encontrados.")

                st.subheader("Resultados web encontrados")
                st.dataframe(
                    pd.DataFrame(resultados_web),
                    use_container_width=True
                )

                if guardar_mongo:
                    try:
                        guardado_mongo = guardar_busqueda_raw_en_mongo(
                            consulta,
                            resultados_web
                        )

                        if guardado_mongo:
                            st.success("Búsqueda cruda guardada en MongoDB Atlas.")
                        else:
                            st.info("MongoDB Atlas no está configurado. Se omitió ese guardado.")
                    except Exception as e:
                        st.warning("No se pudo guardar en MongoDB Atlas.")
                        st.exception(e)

                with st.spinner("Analizando recetas con Gemini..."):
                    postres = analizar_postres_con_gemini(
                        consulta,
                        resultados_web
                    )

                if postres:
                    st.subheader("Postres clasificados por precio")

                    df_postres = pd.DataFrame(postres)
                    st.dataframe(df_postres, use_container_width=True)

                    st.subheader("Gráfico por rango de precio")

                    if "rango_precio" in df_postres.columns:
                        conteo = df_postres["rango_precio"].value_counts()
                        st.bar_chart(conteo)

                    st.subheader("Recomendaciones")

                    for postre in postres:
                        st.markdown(f"""
                        ### 🍮 {postre.get("nombre_postre", "Postre")}

                        **Fuente:** {postre.get("fuente", "No especificada")}  
                        **Costo estimado:** S/ {postre.get("costo_estimado_soles", "N/A")}  
                        **Rango:** {postre.get("rango_precio", "N/A")}  
                        **Dificultad:** {postre.get("dificultad", "N/A")}  
                        **Link:** {postre.get("link", "N/A")}  

                        **Recomendación:**  
                        {postre.get("recomendacion", "N/A")}
                        """)

                    try:
                        guardar_en_postgres(postres, consulta)
                        st.success("Resultados guardados en Supabase PostgreSQL.")
                    except Exception as e:
                        st.error("No se pudo guardar en Supabase PostgreSQL.")
                        st.exception(e)


with tab_historial:
    st.subheader("Historial guardado en Supabase PostgreSQL")

    if st.button("Actualizar historial"):
        try:
            historial = obtener_historial()

            if historial.empty:
                st.info("Todavía no hay resultados guardados.")
            else:
                st.dataframe(historial, use_container_width=True)

        except Exception as e:
            st.error("No se pudo obtener el historial.")
            st.exception(e)


with tab_stack:
    mostrar_stack_tecnologico()