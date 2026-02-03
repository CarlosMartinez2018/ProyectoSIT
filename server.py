import os
import glob
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from Agente_analista import AmadeusDataLoader, MedellinTourismAgent
except ImportError:

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "MedellinTourismAgent", 
        "Agente_analista.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    AmadeusDataLoader = module.AmadeusDataLoader
    MedellinTourismAgent = module.MedellinTourismAgent

# Configuración
load_dotenv()
app = Flask(__name__)
CORS(app)

# Variables globales para el agente
agent = None

def init_agent():
    global agent
    print("Iniciando servidor y cargando agente...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY no encontrada")
        return

    # Rutas de datos
    search_pattern = os.path.join("D:/Otros Contratos/Eafit/Turismo/Agente1/S3 Amadeus/Salida", "search_completo*.parquet")
    searches_files = glob.glob(search_pattern)
    
    searches_df = None
    if searches_files:
        print(f"Cargando {len(searches_files)} archivos de búsquedas...")
        try:
            searches_df = pd.concat([pd.read_parquet(f) for f in searches_files], ignore_index=True)
            print(f"[OK] Searches cargadas: {len(searches_df):,}")
        except Exception as e:
            print(f"Error cargando searches: {e}")
    
    bookings_path = os.getenv("AMADEUS_BOOKINGS_PATH", "D:/Otros Contratos/Eafit/Turismo/Agente1/S3 Amadeus/Salida/bookings_completo.parquet")
    
    loader = AmadeusDataLoader(
        searches_path=searches_df,
        bookings_path=bookings_path
    )
    
    # Crear vector store
    print("Creando índice vectorial...")
    loader.create_vector_store()
    
    # Crear Agente
    print("Instanciando agente...")
    agent = MedellinTourismAgent(
        api_key=api_key,
        data_loader=loader,
        model="gpt-4"
    )
    print("[OK] Agente listo")
    
@app.route('/api/reset', methods=['POST'])
def reset_agent_memory():
    global agent
    if not agent:
        return jsonify({"error": "El agente no está inicializado"}), 500
        
    try:
        msg = agent.reset_memory()
        print(f"{msg}")
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_agent():
    global agent
    if not agent:
        return jsonify({"error": "El agente no se ha inicializado correctamente"}), 500
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Falta la pregunta"}), 400
    
    try:
        print(f"Pregunta recibida: {question}")
        # El agente devuelve un string JSON
        response_str = agent.query(question)
        
        # Parsear ese string a JSON real para enviarlo limpio al front
        import json
        import re
        
        # Log raw response for debugging
        try:
            with open("server_debug.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- PREGUNTA: {question} ---\n")
                f.write(f"RAW RESPONSE:\n{response_str}\n")
        except Exception:
            pass

        response_data = None
        
        # 1. Intentar limpiar bloques de código Markdown
        clean_str = response_str.strip()
        if "```" in clean_str:
            # Eliminar ```json ... ``` ou ``` ... ```
            clean_str = re.sub(r'^```\w*\s*', '', clean_str)
            clean_str = re.sub(r'\s*```$', '', clean_str)
        
        try:
            # Intento directo
            response_data = json.loads(clean_str)
        except json.JSONDecodeError:
            # 2. Búsqueda heurística del JSON
            try:
                # Buscar el primer '{' y el último '}'
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                if start != -1 and end != -1:
                    json_candidate = response_str[start:end]
                    response_data = json.loads(json_candidate)
                    
            except json.JSONDecodeError:
                # 3. Fallback: Construir JSON manual si falla todo
                print("[WARN] No se pudo parsear el JSON. Entregando texto plano.")
                response_data = {"respuesta_texto": response_str}
        
        if not response_data:
             response_data = {"respuesta_texto": response_str}
            
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Error procesando pregunta: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_agent()
    print("Servidor corriendo en http://localhost:5000")
    app.run(debug=False, port=5000)