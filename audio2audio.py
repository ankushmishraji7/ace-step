import modal
import os
import sys
import json
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = modal.App("acestep-audio2audio")

# Define the environment image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "wget", "ffmpeg")
    .pip_install(
        "fastapi[standard]", "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    .add_local_file("requirements.txt", remote_path="/app/requirements.txt", copy=True)
    .run_commands("pip install --no-cache-dir -r /app/requirements.txt")
    .add_local_dir(".", remote_path="/app", copy=True)
    .run_commands("pip install --no-cache-dir /app")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/app"
    })
)

cache_volume = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",  
    timeout=3600,
    secrets=[modal.Secret.from_name("aws-s3-credentials")],  
    volumes={"/cache": cache_volume}
)
def process_audio(
    payload_json: str,
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
    import sys
    from pathlib import Path
    
    os.chdir("/app")
    sys.path.insert(0, "/app")

    from utils.s3_utils import upload_to_s3, download_from_s3
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.data_sampler import DataSampler

    def extract_pipeline_hyperparameters(generator_data: dict, payload: dict) -> dict:
        return {
            "audio_duration": generator_data.get("audio_duration", -1),
            "prompt": payload.get("prompt", payload.get("tags", "funk, pop, soul, rock, melodic, guitar, drums, bass, keyboard, percussion, 105 BPM, energetic, upbeat, groovy, vibrant, dynamic")),
            "lyrics": payload.get("lyrics", generator_data.get("lyrics", """[verse]
Neon lights they flicker bright
City hums in dead of night
Rhythms pulse through concrete veins
Lost in echoes of refrains

[verse]
Bassline groovin' in my chest
Heartbeats match the city's zest
Electric whispers fill the air
Synthesized dreams everywhere

[chorus]
Turn it up and let it flow
Feel the fire let it grow
In this rhythm we belong
Hear the night sing out our song

[verse]
Guitar strings they start to weep
Wake the soul from silent sleep
Every note a story told
In this night we’re bold and gold

[bridge]
Voices blend in harmony
Lost in pure cacophony
Timeless echoes timeless cries
Soulful shouts beneath the skies

[verse]
Keyboard dances on the keys
Melodies on evening breeze
Catch the tune and hold it tight
In this moment we take flight""")),
            "infer_step": generator_data.get("infer_step", 60),
            "guidance_scale": generator_data.get("guidance_scale", 15.0),
            "scheduler_type": generator_data.get("scheduler_type", "euler"),
            "cfg_type": generator_data.get("cfg_type", "apg"),
            "omega_scale": generator_data.get("omega_scale", 10.0),
            "manual_seeds": ", ".join(map(str, generator_data.get("actual_seeds", []))),
            "guidance_interval": generator_data.get("guidance_interval", 0.5),
            "guidance_interval_decay": generator_data.get("guidance_interval_decay", 0.0),
            "min_guidance_scale": generator_data.get("min_guidance_scale", 3.0),
            "use_erg_tag": generator_data.get("use_erg_tag", True),
            "use_erg_lyric": generator_data.get("use_erg_lyric", False),
            "use_erg_diffusion": generator_data.get("use_erg_diffusion", True),
            "oss_steps": ", ".join(map(str, generator_data.get("oss_steps", []))),
            "guidance_scale_text": generator_data.get("guidance_scale_text", 0.0),
            "guidance_scale_lyric": generator_data.get("guidance_scale_lyric", 0.0),
        }

    try:
        print("=== Starting Audio Pipeline Execution on Modal ===")
        
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON payload. {e}")
            raise e

        # Extract S3 Config
        input_bucket = payload.get("input_file", {}).get("bucket")
        input_key = payload.get("input_file", {}).get("S3ObjectKey")
        output_bucket = payload.get("output_file", {}).get("bucket")
        output_key = payload.get("output_file", {}).get("S3ObjectKey")
        temperature = payload.get("temperature")

        input_audio_path = f"/app/data/{input_key}" if input_key else "/app/data/input.wav"
        output_audio_path = f"/app/data/{output_key}" if output_key else "/app/data/output.wav"
        ref_audio_strength = float(temperature) if temperature is not None else 0.5

        if input_bucket and input_key:
            print(f"S3 Input Target: s3://{input_bucket}/{input_key}")
        else:
            print("Warning: Missing input_file configuration in payload.")

        # S3 Input Download Phase
        os.makedirs(os.path.dirname(input_audio_path), exist_ok=True)
        if input_bucket and input_key:
            print("Downloading source audio...")
            download_from_s3(input_bucket, input_key, input_audio_path)

        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Missing input audio file at {input_audio_path}")

        # Model Instantiation Phase
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print("Initializing ACEStepPipeline...")
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode
        )

        data_sampler = DataSampler()
        sampled_data = data_sampler.sample()
        hyperparameters = extract_pipeline_hyperparameters(sampled_data, payload)

        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        # Execution Phase
        print(f"Executing generation pipeline (Duration: {hyperparameters['audio_duration']}s)...")
        pipeline(
            **hyperparameters,
            save_path=output_audio_path,
            audio2audio_enable=True,
            ref_audio_input=input_audio_path,
            ref_audio_strength=ref_audio_strength
        )
        print("Pipeline finished successfully.")

        # S3 Output Upload Phase
        target_bucket = output_bucket or "ai-generated-audio"
        target_key = output_key or Path(output_audio_path).name
        print(f"Uploading generated audio to S3: s3://{target_bucket}/{target_key}")
        upload_to_s3(output_audio_path, target_bucket, target_key)
        
        print("=== Execution Completed Successfully ===")
        
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise e


auth_scheme = HTTPBearer()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("api-auth-secret")] 
)
@modal.fastapi_endpoint(method="POST")
def api_endpoint(data: dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """
    Secured Web entrypoint. Runs on a CPU, authenticates the request, 
    and dispatches the job to the A100 GPU securely.
    """
    expected_api_key = os.environ.get("API_KEY")
    
    if token.credentials != expected_api_key:
        print("Unauthorized access attempt blocked.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    print("Authentication successful.")
    
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing JSON payload in POST body."
        )

    print("Dispatching task to GPU...")
    
    try:
        payload_str = json.dumps(data)
        process_audio.remote(payload_str)
        print("GPU task completed successfully.")
        return {
            "status": "success", 
            "message": "Audio generation completed and uploaded to S3."
        }
        
    except Exception as e:
        print(f"GPU task failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
