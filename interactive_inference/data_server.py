import os
import sys
import shutil

print("[\u23F3] Starting data server... Loading libraries. This may take a few seconds.")

import numpy as np
import physical_ai_av
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

class DataCheckRequest(BaseModel):
    clip_id: str
    t0_relative: int = 0
    duration_s: float = 20.0  # Default to 20s for full clip
    fps: int = 10
    max_width: int = 0
    quality: int = 85
    delete_fullres: bool = False
    cameras: list[str] = None # Filter specific cameras if provided
    as_video: bool = False    # If True, compile to mp4 and return video URLs

avdi = None

@app.on_event("startup")
def load_avdi():
    global avdi
    print("[\u2699] Initializing PhysicalAIAVDatasetInterface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print("[\u2705] Data Server Ready on port 8002!")

@app.post("/data_check")
def data_check(req: DataCheckRequest):
    print(f"[\u26A1] data_check: clip={req.clip_id[:8]}… t0={req.t0_relative} dur={req.duration_s}s max_w={req.max_width} q={req.quality} del_fullres={req.delete_fullres}")
    try:
        if req.duration_s <= 0:
            # If duration_s is 0, it means fetch the entire available clip length.
            print("      [\u23F3] duration_s=0, figuring out total clip duration...")
            first_cam_feature = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
            first_cam = avdi.get_clip_feature(req.clip_id, first_cam_feature, maybe_stream=True)
            if hasattr(first_cam, 'timestamps') and first_cam.timestamps is not None and len(first_cam.timestamps) > 0:
                ts = first_cam.timestamps
                req.duration_s = float(ts[-1] - ts[0]) / 1_000_000.0
                print(f"      [\u2705] Dynamic clip duration: {req.duration_s:.1f}s")
            else:
                req.duration_s = 20.0 # fallback

        num_frames = int(req.duration_s * req.fps)
        time_step_us = int((1.0 / req.fps) * 1_000_000)

        image_timestamps = np.array(
            [req.t0_relative + i * time_step_us for i in range(num_frames)],
            dtype=np.int64,
        )

        all_cameras = {
            "front_wide":  avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            "front_tele":  avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
            "cross_left":  avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            "cross_right": avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            "rear_left":   avdi.features.CAMERA.CAMERA_REAR_LEFT_70FOV,
            "rear_tele":   avdi.features.CAMERA.CAMERA_REAR_TELE_30FOV,
            "rear_right":  avdi.features.CAMERA.CAMERA_REAR_RIGHT_70FOV,
        }
        
        if req.cameras:
            cameras = [(c, all_cameras[c]) for c in req.cameras if c in all_cameras]
        else:
            cameras = list(all_cameras.items())

        # Full-res base dir name (no size/quality tags)
        base_dir_name  = f"{req.clip_id}_{req.t0_relative}_{req.duration_s}s"

        # Actual target dir — includes size/quality suffix if applicable
        size_tag = f"_w{req.max_width}" if req.max_width > 0 else ""
        qual_tag = f"_q{req.quality}"   if req.quality != 85  else ""
        cam_tag = "_" + "-".join(req.cameras) if req.cameras else ""
        req_dir_name = f"{base_dir_name}{size_tag}{qual_tag}{cam_tag}"
        req_dir_path = os.path.join(FRAMES_DIR, req_dir_name)
        os.makedirs(req_dir_path, exist_ok=True)

        frames_data = []
        video_urls = {}
        for i in range(num_frames):
            frames_data.append({
                "frame_idx": i,
                "timestamp": int(image_timestamps[i]),
                "cameras": {}
            })

        if req.as_video:
            expected_files = len(cameras)
            current_files = len([f for f in os.listdir(req_dir_path) if f.endswith(".mp4")]) if os.path.exists(req_dir_path) else 0
        else:
            expected_files = num_frames * len(cameras)
            current_files = len([f for f in os.listdir(req_dir_path) if f.endswith(".webp")]) if os.path.exists(req_dir_path) else 0

        is_cached = (current_files == expected_files)

        if is_cached:
            print(f"      [\u26A1] Already cached: {req_dir_name}")
        else:
            print(f"      Extracting {num_frames} frames × {len(cameras)} cameras → {req_dir_name} ...")

        for cam_name, cam_feature in cameras:
            if req.as_video:
                video_filename = f"{cam_name}.mp4"
                video_filepath = os.path.join(req_dir_path, video_filename)
                video_urls[cam_name] = f"/static/frames/{req_dir_name}/{video_filename}"
                
                if not is_cached:
                    camera = avdi.get_clip_feature(req.clip_id, cam_feature, maybe_stream=True)
                    safe_timestamps = np.clip(image_timestamps, camera.timestamps.min(), camera.timestamps.max())
                    frames_arr, _ = camera.decode_images_from_timestamps(safe_timestamps)
                    for i in range(num_frames):
                        filename = f"tmp_{cam_name}_{i:04d}.webp"
                        filepath = os.path.join(req_dir_path, filename)
                        img_array = frames_arr[i]
                        pil_img = Image.fromarray(img_array)
                        if req.max_width > 0 and pil_img.width > req.max_width:
                            ratio = req.max_width / pil_img.width
                            new_w = req.max_width
                            new_h = int(pil_img.height * ratio)
                            if new_w % 2 != 0: new_w -= 1
                            if new_h % 2 != 0: new_h -= 1
                            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                        pil_img.save(filepath, format="WEBP", quality=req.quality, method=4)
                    
                    import subprocess
                    subprocess.run([
                        "ffmpeg", "-y", "-framerate", str(req.fps),
                        "-i", os.path.join(req_dir_path, f"tmp_{cam_name}_%04d.webp"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        video_filepath
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    for i in range(num_frames):
                        os.remove(os.path.join(req_dir_path, f"tmp_{cam_name}_{i:04d}.webp"))
            else:
                if not is_cached:
                    camera = avdi.get_clip_feature(req.clip_id, cam_feature, maybe_stream=True)
                    safe_timestamps = np.clip(image_timestamps, camera.timestamps.min(), camera.timestamps.max())
                    frames_arr, _ = camera.decode_images_from_timestamps(safe_timestamps)

                for i in range(num_frames):
                    filename = f"{cam_name}_{i:04d}.webp"
                    filepath = os.path.join(req_dir_path, filename)

                    if not is_cached:
                        img_array = frames_arr[i]
                        pil_img = Image.fromarray(img_array)
                        # Resize proportionally if max_width specified
                        if req.max_width > 0 and pil_img.width > req.max_width:
                            ratio = req.max_width / pil_img.width
                            new_w = req.max_width
                            new_h = int(pil_img.height * ratio)
                            if new_w % 2 != 0: new_w -= 1
                            if new_h % 2 != 0: new_h -= 1
                            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                        # method=4 is a good balance between compression speed and file size
                        pil_img.save(filepath, format="WEBP", quality=req.quality, method=4)

                    frames_data[i]["cameras"][cam_name] = f"/static/frames/{req_dir_name}/{filename}"

        # Delete full-res cache dir if requested and a sized variant was used
        if req.delete_fullres and req_dir_name != base_dir_name:
            fullres_path = os.path.join(FRAMES_DIR, base_dir_name)
            if os.path.exists(fullres_path):
                shutil.rmtree(fullres_path)
                print(f"      [DEL] Deleted full-res cache: {base_dir_name}")

        print("[\u2705] Done.\n")
        return {
            "status": "success",
            "frames": frames_data,
            "num_frames": num_frames,
            "video_urls": video_urls
        }
    except Exception as e:
        print(f"[\u274C] Error: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
