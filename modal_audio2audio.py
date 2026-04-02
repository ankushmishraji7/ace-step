import modal
import os
import json

app = modal.App("acestep-audio2audio")

# Define the environment image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "wget", "ffmpeg")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    # STEP 1: Copy ONLY the requirements first to build the cache
    .add_local_file("requirements.txt", remote_path="/app/requirements.txt", copy=True)
    .run_commands("pip install --no-cache-dir -r /app/requirements.txt")
    # STEP 2: Copy the rest of the app (needed if your package requires 'setup.py' to run)
    .add_local_dir(".", remote_path="/app", copy=True)
    .run_commands("pip install --no-cache-dir /app")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/app"
    })
)

# Define a persistent volume to cache downloaded models (like the huggingface cache)
cache_volume = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",  # A100 is recommended for generation tasks with ACE-Step
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],  # Requires this secret to be created in your Modal dashboard
    volumes={"/cache": cache_volume}
)
def process_audio(
    env_data_json: str,
    checkpoint_path: str = "/cache/ace-step/checkpoints",
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    overlapped_decode: bool = False,
    device_id: int = 0
):
    """
    Executes the ACE-Step pipeline in the remote Modal container.
    """
    # Remote imports are required so that the local Modal client doesn't need to install these
    import sys
    from pathlib import Path
    
    # We must explicitly set the working directory to /app because that's where the codebase was copied
    os.chdir("/app")
    sys.path.insert(0, "/app")

    from utils.s3_utils import upload_to_s3, download_from_s3
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.data_sampler import DataSampler

    def get_input_data(json_data):
        return (
            json_data["audio_duration"],
            json_data["prompt"],
            json_data["lyrics"],
            json_data["infer_step"],
            json_data["guidance_scale"],
            json_data["scheduler_type"],
            json_data["cfg_type"],
            json_data["omega_scale"],
            ", ".join(map(str, json_data["actual_seeds"])),
            json_data["guidance_interval"],
            json_data["guidance_interval_decay"],
            json_data["min_guidance_scale"],
            json_data["use_erg_tag"],
            json_data["use_erg_lyric"],
            json_data["use_erg_diffusion"],
            ", ".join(map(str, json_data["oss_steps"])),
            json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
            json_data["guidance_scale_lyric"] if "guidance_scale_lyric" in json_data else 0.0,
        )

    try:
        print("=== Starting Audio Pipeline Execution on Modal ===")
        
        # 1. Parse JSON payload
        try:
            data = json.loads(env_data_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in env_data_json. {e}")
            raise e

        input_bucket = data.get("input_file", {}).get("bucket")
        input_key = data.get("input_file", {}).get("S3ObjectKey")
        output_bucket = data.get("output_file", {}).get("bucket")
        output_key = data.get("output_file", {}).get("S3ObjectKey")
        temperature = data.get("temperature")

        # Fallbacks
        input_audio_path = f"/app/data/{input_key}" if input_key else "/app/data/input.wav"
        output_path = f"/app/data/{output_key}" if output_key else "/app/data/output.wav"
        ref_audio_strength = float(temperature) if temperature is not None else 0.5

        if not input_bucket or not input_key:
            print("Warning: input_file bucket or S3ObjectKey is missing in DATA.")
        else:
            print(f"Parsed ENV DATA - Input S3: s3://{input_bucket}/{input_key}, Output S3: s3://{output_bucket}/{output_key}")

        # 2. S3 Setup / Data Dowload
        os.makedirs(os.path.dirname(input_audio_path), exist_ok=True)
        if input_bucket and input_key:
            print(f"Downloading source audio from S3: {input_bucket}/{input_key} ...")
            download_from_s3(input_bucket, input_key, input_audio_path)
            print("Download successful.")
        else:
            print("No S3 input configuration provided. Relying on existing local file if any.")

        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio file not found at {input_audio_path}")

        # 3. Model Setup
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print("Initializing ACEStepPipeline...")
        model_demo = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode
        )
        print("Pipeline initialized successfully.")

        # 4. Prepare data parameters
        print("Sampling generator parameters...")
        data_sampler = DataSampler()
        json_data_sampler = data_sampler.sample()
        
        (   audio_duration, prompt, lyrics, infer_step, guidance_scale,
            scheduler_type, cfg_type, omega_scale, manual_seeds,
            guidance_interval, guidance_interval_decay, min_guidance_scale,
            use_erg_tag, use_erg_lyric, use_erg_diffusion, oss_steps,
            guidance_scale_text, guidance_scale_lyric
        ) = get_input_data(json_data_sampler)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 5. Execute Model Inference
        print(f"Running inference with audio-to-audio enabled. Duration: {audio_duration}s")
        model_demo(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=oss_steps,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=output_path,
            audio2audio_enable=True,
            ref_audio_input=input_audio_path,
            ref_audio_strength=ref_audio_strength
        )
        
        print(f"Inference completed. Generated output saved to: {output_path}")

        # 6. S3 Upload generated output
        upload_bucket = output_bucket if output_bucket else "ai-generated-audio"
        upload_key = output_key if output_key else Path(output_path).name
        print(f"Uploading generated audio to S3 bucket: {upload_bucket}/{upload_key} ...")
        upload_to_s3(output_path, upload_bucket, upload_key)
        
        print("=== Audio Pipeline Execution Completed Successfully ===")
        # Modal handles the exit natively.
        
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise e

# 1. We use a standard @app.function() with NO gpu specified. 
# This tells Modal to run this endpoint on a cheap, fast CPU.
@app.function() 
@modal.fastapi_endpoint(method="POST")
def api_endpoint(data: dict):
    """
    Web entrypoint. Runs on a CPU, receives the HTTP POST, 
    and dispatches the job to the A100 GPU securely in the Cloud.
    """
    import json
    
    print(f"API hit! Received payload: {data}")
    
    if not data:
        return {"error": "You must provide a JSON payload in the POST body."}

    print("Dispatching workload to the A100 GPU...")
    
    try:
        # We convert the dict back to a string because your process_audio 
        # function currently expects `env_data_json: str`
        payload_str = json.dumps(data)
        
        # .remote() blocks this CPU container until the GPU container finishes
        process_audio.remote(payload_str)
        
        print("Modal GPU workload completed!")
        
        # Return a JSON response back to your MERN Fargate server
        return {
            "status": "success", 
            "message": "Audio generation completed and uploaded to S3."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
