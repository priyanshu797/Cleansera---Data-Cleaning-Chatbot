from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, json, re, time, uuid, math
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq
import os
from dotenv import load_dotenv
app = Flask(__name__)
CORS(app)

load_dotenv()

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("cleaned", exist_ok=True)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    HAS_GROQ = True
except Exception:
    client = None
    HAS_GROQ = False

# Sessions now include conversation history
sessions = {}

def call_groq_api(prompt, conversation_history=None):
    """Call Groq API with conversation history support."""
    if not client:
        raise Exception("Groq client not configured")
    
    messages = []
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")

def safe_to_records(df: pd.DataFrame, limit=None, page=0):
    """Convert DataFrame to JSON-serializable list-of-dicts with pagination."""
    if limit is None:
        txt = df.to_json(orient="records", date_format="iso")
        return json.loads(txt)
    else:
        start = page * limit
        end = start + limit
        sub = df.iloc[start:end]
        txt = sub.to_json(orient="records", date_format="iso")
        return json.loads(txt)

def get_data_profile(df: pd.DataFrame):
    """Return comprehensive data profile."""
    profile = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "rows": int(len(df)),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    return profile

def analyze_with_ai(profile: dict):
    """Ask Groq to analyze dataset with enhanced context."""
    if not client:
        return json.dumps({
            "summary": "AI analysis unavailable (Groq not configured)",
            "suggestions": ["Upload data to see basic statistics"],
            "recommended_plots": []
        })
    
    prompt = f"""You are a professional data analyst. Analyze this dataset and provide actionable insights.

Dataset Profile:
{json.dumps(profile, indent=2)}

Provide a JSON response with:
1. "summary": Brief overview of the dataset
2. "suggestions": List of 3-5 data cleaning/analysis recommendations
3. "recommended_plots": List of visualization suggestions with format:
   [{{"type": "bar", "x": "column1", "y": "column2", "title": "Description"}}, ...]

Supported plot types: bar, pie, histogram, line, scatter, bubble, box, heatmap

Return ONLY valid JSON, no markdown formatting."""

    try:
        text = call_groq_api(prompt)
        # Clean markdown formatting
        text = text.replace("```json", "").replace("```", "").strip()
        return text
    except Exception as e:
        return json.dumps({
            "summary": f"AI analysis error: {e}",
            "suggestions": [],
            "recommended_plots": []
        })

def parse_json_response(text):
    """Parse JSON from AI response with robust error handling."""
    if not text:
        return {"type": "message", "content": "No response"}
    
    txt = text.strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(txt)
    except Exception:
        # Try to fix common JSON issues
        try:
            fixed = re.sub(r'(\w+):', r'"\1":', txt)
            return json.loads(fixed)
        except Exception:
            return {"type": "message", "content": txt}

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/api/health")
def health():
    api_status = "configured" if HAS_GROQ and client else "not_configured"
    return jsonify({
        "status": "ok", 
        "model": GROQ_MODEL,
        "api_status": api_status,
        "features": ["pagination", "search", "all_visualizations", "conversation_memory"]
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"})
    
    session_id = str(uuid.uuid4())
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed reading file: {e}"})

    # Auto-detect and convert datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(10).tolist()
            if any(re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', s) for s in sample):
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    pass

    # Initialize session with dataframe history and conversation history
    sessions[session_id] = {
        "dataframes": [df.copy()],
        "conversation": [],
        "filename": file.filename
    }

    profile = get_data_profile(df)
    preview = safe_to_records(df, limit=100, page=0)
    ai_analysis = analyze_with_ai(profile)

    return jsonify({
        "success": True,
        "session_id": session_id,
        "preview": preview,
        "profile": profile,
        "ai_analysis": ai_analysis,
        "total_rows": len(df)
    })

@app.route("/api/get_page", methods=["POST"])
def get_page():
    """Server-side pagination with search support."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    page = data.get("page", 1)
    rows_per_page = data.get("rows_per_page", 50)
    search_query = data.get("search_query", "").strip()
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    
    df = sessions[session_id]["dataframes"][-1]
    
    if search_query:
        mask = df.astype(str).apply(
            lambda row: row.str.contains(search_query, case=False, na=False).any(), 
            axis=1
        )
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    total_rows = len(filtered_df)
    total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
    
    page_idx = page - 1
    start = page_idx * rows_per_page
    end = start + rows_per_page
    
    page_data = filtered_df.iloc[start:end]
    preview = safe_to_records(page_data, limit=None)
    
    return jsonify({
        "success": True,
        "preview": preview,
        "page": page,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "rows_per_page": rows_per_page
    })

@app.route("/api/search", methods=["POST"])
def search():
    """Enhanced server-side search with column-specific filtering."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    query = data.get("query", "").strip()
    page = data.get("page", 1)
    rows_per_page = data.get("rows_per_page", 50)
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session_id"}), 400
    
    if not query:
        return jsonify({"success": False, "error": "Empty search query"}), 400
    
    df = sessions[session_id]["dataframes"][-1]
    
    try:
        column_searches = re.findall(r'(\w+):([^\s]+)', query)
        
        if column_searches:
            mask = pd.Series([True] * len(df))
            for col_name, search_value in column_searches:
                if col_name in df.columns:
                    col_data = df[col_name]
                    if pd.api.types.is_numeric_dtype(col_data):
                        try:
                            numeric_val = float(search_value)
                            mask &= (col_data == numeric_val)
                        except ValueError:
                            mask &= col_data.astype(str).str.contains(search_value, case=False, na=False)
                    else:
                        mask &= col_data.astype(str).str.contains(search_value, case=False, na=False, regex=False)
            filtered_df = df[mask]
        else:
            mask = df.astype(str).apply(
                lambda row: row.str.contains(query, case=False, na=False, regex=False).any(), 
                axis=1
            )
            filtered_df = df[mask]
        
        total_results = len(filtered_df)
        total_pages = math.ceil(total_results / rows_per_page) if total_results > 0 else 1
        
        page_idx = page - 1
        start = page_idx * rows_per_page
        end = start + rows_per_page
        
        results_page = filtered_df.iloc[start:end]
        results = safe_to_records(results_page, limit=None)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_results": total_results,
            "total_pages": total_pages,
            "page": page,
            "rows_per_page": rows_per_page,
            "query": query
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Search failed: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Enhanced chat with conversation memory and all visualization types."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    message = data.get("message", "")
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    df = sessions[session_id]["dataframes"][-1].copy()
    conversation = sessions[session_id]["conversation"]
    filename = sessions[session_id].get("filename", "data")
    
    # Add user message to conversation history
    conversation.append({"role": "user", "content": message})
    
    profile = get_data_profile(df)
    sample = df.head(5).to_string()

    system_prompt = f"""You are an advanced data analysis assistant with conversation memory.

DATASET INFO:
- Filename: {filename}
- Columns: {profile['columns']}
- Numeric columns: {profile['numeric_columns']}
- Categorical columns: {profile['categorical_columns']}
- Datetime columns: {profile['datetime_columns']}
- Total rows: {profile['rows']}
- Data types: {profile['dtypes']}

SAMPLE DATA:
{sample}

CAPABILITIES:
1. Data transformations (filter, sort, group, aggregate)
2. Statistical analysis (mean, median, sum, correlation)
3. All visualization types: bar, pie, histogram, line, scatter, bubble, box, heatmap, treemap, sunburst, violin, strip, density

RESPONSE FORMAT (JSON only):
{{
  "type": "code|analysis|plot|question|message",
  "content": "<python code or message>",
  "explanation": "<what you're doing>",
  "visualization_type": "bar|pie|histogram|..." (for plots)
}}

PLOT EXAMPLES:
- Bar: fig = px.bar(df, x='category', y='value', title='Title')
- Pie: fig = px.pie(df, names='category', values='value')
- Histogram: fig = px.histogram(df, x='column', nbins=20)
- Line: fig = px.line(df, x='date', y='value')
- Scatter: fig = px.scatter(df, x='col1', y='col2', color='category')
- Bubble: fig = px.scatter(df, x='col1', y='col2', size='col3', color='category')
- Box: fig = px.box(df, x='category', y='value')
- Heatmap: fig = px.imshow(df.corr(), text_auto=True)
- Treemap: fig = px.treemap(df, path=['category'], values='value')
- Violin: fig = px.violin(df, x='category', y='value')

USER REQUEST: {message}

Respond with JSON only. Consider previous conversation context."""

    try:
        response_text = call_groq_api(system_prompt, conversation[:-1])  # Exclude the just-added user message
        parsed = parse_json_response(response_text)
        
        # Add assistant response to conversation
        conversation.append({
            "role": "assistant", 
            "content": parsed.get("explanation", parsed.get("content", ""))
        })
        
    except Exception as e:
        # Fallback to basic operations if Groq fails
        if not client:
            return pd.handle_basic_operations(df, message, sessions[session_id])
        return jsonify({"error": f"AI call failed: {e}"}), 500

    rtype = parsed.get("type", "message")
    content = parsed.get("content", "")
    explanation = parsed.get("explanation", "")

    try:
        if rtype == "code":
            exec_globals = {"pd": pd, "np": np}
            exec_locals = {"df": df}
            exec(content, exec_globals, exec_locals)
            new_df = exec_locals.get("df", df)
            sessions[session_id]["dataframes"].append(new_df.copy())
            
            return jsonify({
                "type": "code",
                "success": True,
                "content": content,
                "explanation": explanation,
                "preview": safe_to_records(new_df, limit=100, page=0),
                "profile": get_data_profile(new_df)
            })

        elif rtype == "analysis":
            local_vars = {"df": df, "np": np, "pd": pd}
            exec(content, {}, local_vars)
            result = local_vars.get("result", None)
            
            return jsonify({
                "type": "analysis",
                "success": True,
                "content": content,
                "result": str(result),
                "explanation": explanation
            })

        elif rtype == "plot":
            local_vars = {"df": df, "px": px, "go": go, "pd": pd, "np": np}
            exec(content, {}, local_vars)
            fig = local_vars.get("fig")
            
            if fig:
                plot_json = fig.to_json()
                return jsonify({
                    "type": "plot",
                    "success": True,
                    "plot": plot_json,
                    "explanation": explanation,
                    "visualization_type": parsed.get("visualization_type", "unknown")
                })
            else:
                return jsonify({"type": "message", "content": "No figure generated"})

        elif rtype == "question":
            return jsonify({
                "type": "question",
                "success": True,
                "content": content
            })

        else:
            return jsonify({"type": "message", "content": content})
            
    except Exception as e:
        return jsonify({
            "type": rtype,
            "success": False,
            "error": str(e),
            "content": content
        })

@app.route("/api/undo", methods=["POST"])
def undo():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "Invalid session"})
    
    if len(sessions[session_id]["dataframes"]) <= 1:
        return jsonify({"success": False, "error": "No undo available"})
    
    sessions[session_id]["dataframes"].pop()
    df = sessions[session_id]["dataframes"][-1]
    
    return jsonify({
        "success": True,
        "preview": safe_to_records(df, limit=100, page=0),
        "profile": get_data_profile(df)
    })

@app.route("/api/download", methods=["POST"])
def download():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    df = sessions[session_id]["dataframes"][-1]
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name="cleaned_data.csv",
        mimetype="text/csv"
    )

if __name__ == "__main__":
    print("Cleansera AI Backend v2.0 (Groq-Powered)")
    print(f"Model: {GROQ_MODEL}")
    print("Features: All visualizations, conversation memory, enhanced search")
    print("Server: http://localhost:5000")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
