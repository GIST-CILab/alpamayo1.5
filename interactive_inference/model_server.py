import os
import sys

print("[\u23F3] Starting model server... Loading heavy libraries (PyTorch, Alpamayo, etc.). This may take up to a minute.")

import io
import base64
import torch
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.path.append(os.path.abspath("/workspace/Projects/alpamayo1.5"))

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper, nav_utils

app = FastAPI()

# Global variables for model and processor
model = None
processor = None

@app.on_event("startup")
def load_model():
    global model, processor
    print("Loading Alpamayo 1.5 model...")
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    cache_path = "/workspace/.cache/huggingface/hub"
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, cache_dir=cache_path).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded successfully!")

class InferenceRequest(BaseModel):
    clip_id: str
    t0_relative: int
    nav_text: str = "Turn right in 30m"
    temperature: float = 0.6
    top_p: float = 0.98
    num_runs: int = 1
    num_traj_samples: int = 16
    max_generation_length: int = 256
    # You can add CFG parameters here if needed later

@app.post("/predict")
def predict(req: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    
    print(f"[\u26A1] Received prediction request: clip_id={req.clip_id}, t0={req.t0_relative}")
    print(f"      Instruction: '{req.nav_text}' (Runs: {req.num_traj_samples}, Temp: {req.temperature}, Top-p: {req.top_p})")
    
    try:
        with torch.inference_mode():
            print("      [1/4] Loading physical dataset...")
            data_scene = load_physical_aiavdataset(req.clip_id, t0_us=req.t0_relative)
            frames = data_scene["image_frames"].flatten(0, 1)
    
            print("      [2/4] Formatting inputs with chat template...")
            messages_nav = helper.create_message(
                frames,
                camera_indices=data_scene.get("camera_indices"),
                nav_text=req.nav_text,
            )
            
            inputs_nav = processor.apply_chat_template(
                messages_nav,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            model_inputs_nav = helper.to_device(
                {
                    "tokenized_data": inputs_nav,
                    "ego_history_xyz": data_scene["ego_history_xyz"],
                    "ego_history_rot": data_scene["ego_history_rot"],
                },
                "cuda",
            )
            
            # Free CPU memory where possible
            del frames
            del inputs_nav
    
            print(f"      [3/4] Running {req.num_runs} independent inferences (Chunking applied to prevent OOM)...")
            
            pred_xy_list = []
            cot_list = []
            for run_idx in range(req.num_runs):
                print(f"            -> Run {run_idx+1}/{req.num_runs}")
                torch.cuda.manual_seed_all(42 + run_idx)
                
                run_pred_xy = []
                chunk_size = 4
                import math
                num_chunks = math.ceil(req.num_traj_samples / chunk_size)
                
                run_cots = []
                
                for c in range(num_chunks):
                    samples = min(chunk_size, req.num_traj_samples - c * chunk_size)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred_xyz_nav, pred_rot_nav, extra_nav = model.sample_trajectories_from_data_with_vlm_rollout(
                            data=model_inputs_nav,
                            top_p=req.top_p,
                            temperature=req.temperature,
                            num_traj_samples=samples,
                            max_generation_length=req.max_generation_length,
                            return_extra=True,
                            diffusion_kwargs={
                                "temperature": req.temperature,
                            }
                        )
                    
                    for i in range(pred_xyz_nav.shape[2]):
                        run_pred_xy.append(pred_xyz_nav.cpu()[0, 0, i, :, :2].tolist())
                    
                    if extra_nav is not None and "cot" in extra_nav:
                        cots = extra_nav["cot"]
                        if isinstance(cots, list):
                            run_cots.extend(cots)
                        else:
                            run_cots.append(cots)
                        
                    # Cleanup GPU memory for next chunk
                    del pred_xyz_nav, pred_rot_nav, extra_nav
                    torch.cuda.empty_cache()
                    
                pred_xy_list.append(run_pred_xy)
                cot_list.append(str(run_cots))
    
            print("      [4/4] Inference completed. Processing output tensors...")
                
            gt_xy = data_scene["ego_future_xyz"].cpu()[0, 0, :, :2].tolist()
            del data_scene
            del model_inputs_nav
            torch.cuda.empty_cache()

        print(f"[\u2705] Success! Returning {len(pred_xy_list)} independent trajectories to client.\n")
        return {
            "status": "success",
            "pred_xy": pred_xy_list,
            "gt_xy": gt_xy,
            "cot": cot_list
        }
    except Exception as e:
        print(f"[\u274C] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
