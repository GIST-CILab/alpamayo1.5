import os
import sys
import json

print("[\u23F3] Starting web server... Loading libraries. This may take a few seconds.")

import sqlite3
import requests
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    return RedirectResponse(url="/")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

USER_ID = "default"
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    USER_ID = sys.argv[1].strip()

DB_FILENAME = f"experiments-{USER_ID}.db" if USER_ID != "default" else "experiments.db"
DB_PATH = os.path.join(DATA_DIR, DB_FILENAME)
print(f"[\u2699\uFE0F] Using database: {DB_FILENAME}")
SAMPLES_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "notebooks", "nav_demo_samples.json"))

def get_sample_clips():
    try:
        if os.path.exists(SAMPLES_JSON_PATH):
            with open(SAMPLES_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[\u274C] Failed to load samples json: {e}")
    return []

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            obs_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            nav_text TEXT,
            temperature REAL,
            top_p REAL,
            num_traj_samples INTEGER,
            FOREIGN KEY(obs_id) REFERENCES observations(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id TEXT,
            t0_us INTEGER,
            gt_trajectory TEXT,
            UNIQUE(clip_id, t0_us)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            run_index INTEGER,
            cot TEXT,
            pred_trajectory TEXT,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        )
    """)
        
    # Sync observations from JSON
    clips = get_sample_clips()
    for clip in clips:
        cursor.execute("INSERT OR IGNORE INTO observations (clip_id, t0_us) VALUES (?, ?)", 
                       (clip["clip_id"], clip["t0_relative"]))
        
    conn.commit()
    conn.close()

init_db()

MODEL_SERVER_URL = "http://127.0.0.1:8000"
DATA_SERVER_URL = "http://127.0.0.1:8002"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="integrated.html", context={"clips": get_sample_clips()})

class ApiDataCheckRequest(BaseModel):
    clip_id: str
    t0_relative: int = 0
    duration_s: float = 20.0
    max_width: int = 0
    quality: int = 85
    delete_fullres: bool = False
    cameras: list[str] = None
    as_video: bool = False

class ApiInferenceRequest(BaseModel):
    clip_id: str
    t0_relative: int
    nav_text: str
    temperature: float = 0.6
    top_p: float = 0.98
    num_runs: int = 1

@app.delete("/api/experiment/{exp_id}")
def api_delete_experiment(exp_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete the experiment and cascade will handle the runs if PRAGMA foreign_keys is on,
        # but to be safe we can manually delete runs first
        cursor.execute("DELETE FROM runs WHERE experiment_id=?", (exp_id,))
        cursor.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
        conn.commit()
        
        # Also clean up empty observations
        cursor.execute("DELETE FROM observations WHERE id NOT IN (SELECT obs_id FROM experiments)")
        conn.commit()
        
        conn.close()
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/data_check")
def api_data_check(req: ApiDataCheckRequest):
    try:
        res = requests.post(f"{DATA_SERVER_URL}/data_check", json={
            "clip_id": req.clip_id.strip(),
            "t0_relative": req.t0_relative,
            "duration_s": req.duration_s,
            "max_width": req.max_width,
            "quality": req.quality,
            "delete_fullres": req.delete_fullres,
            "cameras": req.cameras,
            "as_video": req.as_video
        })
        res.raise_for_status()
        return JSONResponse(content=res.json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/inference")
def api_inference(req: ApiInferenceRequest):
    try:
        res = requests.post(f"{MODEL_SERVER_URL}/predict", json={
            "clip_id": req.clip_id.strip(),
            "t0_relative": req.t0_relative,
            "nav_text": req.nav_text,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "num_runs": req.num_runs
        })
        res.raise_for_status()
        data = res.json()
        
        pred_xy = np.array(data["pred_xy"])
        gt_xy = np.array(data["gt_xy"])
        cot = data["cot"]
        
        all_run_means = []
        for run_idx, run_trajs in enumerate(pred_xy):
            run_trajs = np.array(run_trajs)
            run_mean = run_trajs.mean(axis=0)
            all_run_means.append(run_mean.tolist())
            
        total_mean = []
        if len(all_run_means) > 0:
            total_mean = np.array(all_run_means).mean(axis=0).tolist()
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        gt_xy_json = json.dumps(gt_xy.tolist())
        cursor.execute("INSERT OR IGNORE INTO observations (clip_id, t0_us) VALUES (?, ?)", 
                       (req.clip_id, req.t0_relative))
        cursor.execute("UPDATE observations SET gt_trajectory=? WHERE clip_id=? AND t0_us=?", 
                       (gt_xy_json, req.clip_id, req.t0_relative))
                       
        cursor.execute("SELECT id FROM observations WHERE clip_id=? AND t0_us=?", (req.clip_id, req.t0_relative))
        obs_id = cursor.fetchone()[0]
        
        cursor.execute("""
            INSERT INTO experiments (obs_id, nav_text, temperature, top_p, num_traj_samples)
            VALUES (?, ?, ?, ?, ?)
        """, (obs_id, req.nav_text, req.temperature, req.top_p, req.num_runs))
        experiment_id = cursor.lastrowid
        
        for i in range(pred_xy.shape[0]):
            pred_traj_json = json.dumps(pred_xy[i].tolist())
            cot_str = cot[i] if isinstance(cot, list) else cot
            cursor.execute("""
                INSERT INTO runs (experiment_id, run_index, cot, pred_trajectory)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, i, cot_str, pred_traj_json))
        conn.commit()
        conn.close()
        
        print(f"[\u2705] API Inference successful. Saved to DB: experiment_id={experiment_id}")
            
        return JSONResponse(content={
            "status": "success",
            "pred_xy": pred_xy.tolist(),
            "gt_xy": gt_xy.tolist(),
            "run_means": all_run_means,
            "total_mean": total_mean,
            "cot": cot[0] if isinstance(cot, list) else cot
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/data_check", response_class=HTMLResponse)
def data_check_page(request: Request):
    return templates.TemplateResponse(request=request, name="data_check.html", context={"clips": get_sample_clips()})

@app.post("/data_check", response_class=HTMLResponse)
def data_check_post(request: Request, clip_id: str = Form(...), t0_relative: int = Form(...), duration_s: float = Form(5.0)):
    print(f"[\u26A1] Forwarding data_check to data server: {clip_id}, {t0_relative}, {duration_s}s")
    try:
        res = requests.post(f"{DATA_SERVER_URL}/data_check", json={
            "clip_id": clip_id.strip(),
            "t0_relative": t0_relative,
            "duration_s": duration_s
        })
        res.raise_for_status()
        data = res.json()
        frames = data.get("frames", [])
        num_frames = data.get("num_frames", 0)
        
        print(f"[\u2705] Received data_check response from data server: {num_frames} frames.")
        return templates.TemplateResponse(request=request, name="data_check.html", context={
            "clip_id": clip_id, 
            "t0_relative": t0_relative, 
            "duration_s": duration_s,
            "frames": frames,
            "num_frames": num_frames,
            "clips": get_sample_clips()
        })
    except Exception as e:
        print(f"[\u274C] Error in data_check: {e}")
        return templates.TemplateResponse(request=request, name="data_check.html", context={
            "error": str(e)
        })

@app.get("/clip_viewer", response_class=HTMLResponse)
def clip_viewer_page(request: Request):
    return templates.TemplateResponse(request=request, name="clip_viewer.html", context={"clips": get_sample_clips()})

@app.get("/inference", response_class=HTMLResponse)
def inference_page(request: Request):
    return templates.TemplateResponse(request=request, name="inference.html", context={"clips": get_sample_clips()})

@app.post("/inference", response_class=HTMLResponse)
def inference_post(
    request: Request,
    clip_id: str = Form(...),
    t0_relative: int = Form(...),
    nav_text: str = Form(...),
    temperature: float = Form(0.6),
    top_p: float = Form(0.98),
    num_runs: int = Form(1)
):
    print(f"[\u26A1] Forwarding inference request to model server: {clip_id}, runs={num_runs}")
    try:
        # We request num_runs from the model server, which uses num_traj_samples to generate multiple trajectories
        res = requests.post(f"{MODEL_SERVER_URL}/predict", json={
            "clip_id": clip_id.strip(),
            "t0_relative": t0_relative,
            "nav_text": nav_text,
            "temperature": temperature,
            "top_p": top_p,
            "num_runs": num_runs
        })
        res.raise_for_status()
        data = res.json()
        
        pred_xy = np.array(data["pred_xy"]) # shape [num_runs, 256, 2]
        gt_xy = np.array(data["gt_xy"]) # shape [256, 2]
        cot = data["cot"]
        
        all_run_means = []
        for run_idx, run_trajs in enumerate(pred_xy):
            run_trajs = np.array(run_trajs)
            run_mean = run_trajs.mean(axis=0)
            all_run_means.append(run_mean.tolist())
            
        total_mean = []
        if len(all_run_means) > 0:
            total_mean = np.array(all_run_means).mean(axis=0).tolist()
        
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        gt_xy_json = json.dumps(gt_xy.tolist())
        cursor.execute("INSERT OR IGNORE INTO observations (clip_id, t0_us) VALUES (?, ?)", 
                       (clip_id, t0_relative))
        cursor.execute("UPDATE observations SET gt_trajectory=? WHERE clip_id=? AND t0_us=?", 
                       (gt_xy_json, clip_id, t0_relative))
                       
        cursor.execute("SELECT id FROM observations WHERE clip_id=? AND t0_us=?", (clip_id, t0_relative))
        obs_id = cursor.fetchone()[0]
        
        cursor.execute("""
            INSERT INTO experiments (obs_id, nav_text, temperature, top_p, num_traj_samples)
            VALUES (?, ?, ?, ?, ?)
        """, (obs_id, nav_text, temperature, top_p, num_runs))
        experiment_id = cursor.lastrowid
        
        for i in range(pred_xy.shape[0]):
            pred_traj_json = json.dumps(pred_xy[i].tolist())
            cot_str = cot[i] if isinstance(cot, list) else cot
            cursor.execute("""
                INSERT INTO runs (experiment_id, run_index, cot, pred_trajectory)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, i, cot_str, pred_traj_json))
        conn.commit()
        conn.close()
            
        print("[\u2705] Inference successful. Plot saved to DB.")
            
        return templates.TemplateResponse(request=request, name="inference.html", context={
            "success": True,
            "clips": get_sample_clips(),
            "pred_xy_json": json.dumps(pred_xy.tolist()),
            "gt_xy_json": json.dumps(gt_xy.tolist()),
            "run_means_json": json.dumps(all_run_means),
            "total_mean_json": json.dumps(total_mean),
            "cot": cot[0] if isinstance(cot, list) else cot,
            "clip_id": clip_id,
            "t0_relative": t0_relative,
            "nav_text": nav_text,
            "temperature": temperature,
            "top_p": top_p,
            "num_runs": num_runs
        })

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[\u274C] Error in inference_post: {err_msg}")
        return templates.TemplateResponse(request=request, name="inference.html", context={
            "error": err_msg,
            "clips": get_sample_clips()
        })

@app.get("/api/history")
def api_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM observations ORDER BY id DESC")
        obs_rows = cursor.fetchall()
        
        history = []
        for obs in obs_rows:
            obs_dict = dict(obs)
            cursor.execute("SELECT id, nav_text, num_traj_samples as num_runs, timestamp FROM experiments WHERE obs_id=? ORDER BY id DESC", (obs["id"],))
            exps = [dict(row) for row in cursor.fetchall()]
            if exps:
                obs_dict["experiments"] = exps
                history.append(obs_dict)
                
        conn.close()
        return JSONResponse(content={"status": "success", "history": history})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/experiment_results")
def api_experiment_results(exp_ids: str):
    try:
        if not exp_ids:
            return JSONResponse(content={"status": "success", "experiments": []})
            
        exp_id_list = [int(x.strip()) for x in exp_ids.split(",")]
        if not exp_id_list:
            return JSONResponse(content={"status": "success", "experiments": []})
            
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get obs_id from first exp to get gt_trajectory
        cursor.execute("SELECT obs_id FROM experiments WHERE id=?", (exp_id_list[0],))
        obs_id_row = cursor.fetchone()
        if not obs_id_row:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Experiment not found"})
            
        cursor.execute("SELECT gt_trajectory FROM observations WHERE id=?", (obs_id_row["obs_id"],))
        gt_traj_row = cursor.fetchone()
        gt_xy = json.loads(gt_traj_row["gt_trajectory"]) if gt_traj_row and gt_traj_row["gt_trajectory"] else []
        
        results = []
        for exp_id in exp_id_list:
            cursor.execute("SELECT nav_text FROM experiments WHERE id=?", (exp_id,))
            exp_row = cursor.fetchone()
            if not exp_row: continue
            
            cursor.execute("SELECT run_index, cot, pred_trajectory FROM runs WHERE experiment_id=? ORDER BY run_index", (exp_id,))
            run_rows = cursor.fetchall()
            
            runs_data = []
            for r in run_rows:
                runs_data.append({
                    "run_index": r["run_index"],
                    "cot": r["cot"],
                    "pred_xy": json.loads(r["pred_trajectory"]) if r["pred_trajectory"] else []
                })
                
            results.append({
                "id": exp_id,
                "nav_text": exp_row["nav_text"],
                "runs": runs_data
            })
            
        conn.close()
        return JSONResponse(content={"status": "success", "gt_xy": gt_xy, "experiments": results})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    import subprocess
    import os
    import signal
    
    # Automatically kill any process using port 8888
    try:
        print("[⚡] Attempting to free port 8888...")
        result = subprocess.run(["ss", "-lptn", "sport = :8888"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "pid=" in line:
                pid_str = line.split("pid=")[1].split(",")[0]
                pid = int(pid_str)
                print(f"     Found process {pid} using port 8888. Killing it...")
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"[⚠️] Failed to free port 8888: {e}")

    # 0.0.0.0:8888 binds to the RunPod proxy
    uvicorn.run(app, host="0.0.0.0", port=8888)
