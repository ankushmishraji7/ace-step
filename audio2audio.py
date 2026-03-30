import click
import os
import json
import sys
from pathlib import Path

from utils.s3_utils import upload_to_s3, download_from_s3
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


def get_data_from_env():
    """Extracts job parameters from the DATA environment variable."""
    json_str = os.getenv("DATA")
    region = os.getenv("BUCKET_REGION")
    if not json_str:
        print("Error: DATA environment variable is not set or is empty.")
        raise ValueError("DATA environment variable missing")
    if not region:
        print("Error: BUCKET_REGION environment variable is not set or is empty.")
        raise ValueError("BUCKET_REGION environment variable missing")

    try:
        data = json.loads(json_str)
        input_bucket = data.get("input_file", {}).get("bucket")
        input_key = data.get("input_file", {}).get("S3ObjectKey")
        output_bucket = data.get("output_file", {}).get("bucket")
        output_key = data.get("output_file", {}).get("S3ObjectKey")
        temperature = data.get("temperature")

        if not input_bucket or not input_key:
            print("Warning: input_file bucket or S3ObjectKey is missing in DATA.")

        print(f"Parsed ENV DATA - Input S3: s3://{input_bucket}/{input_key}, Output S3: s3://{output_bucket}/{output_key}")
        return input_bucket, input_key, output_bucket, output_key, region, temperature

    except (TypeError, json.JSONDecodeError) as e:
        print(f"Error: Invalid JSON format in DATA env var. Exception: {e}")
        raise e


def get_input_data(json_data):
    """Formats sampler parameters for the pipeline."""
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
        (
            json_data["guidance_scale_lyric"]
            if "guidance_scale_lyric" in json_data
            else 0.0
        ),
    )


@click.command()
@click.option(
    "--checkpoint_path", type=str, default="", help="Path to the checkpoint directory"
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)
@click.option("--device_id", type=int, default=0, help="Device ID to use")
@click.option("--output_path", type=str, default=None, help="Path to save the output (usually in data folder)")
@click.option("--ref_audio_strength", type=float, default=0.5, help="Strength of the reference audio input (0.0 to 1.0)")
def main(checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode, device_id, output_path, ref_audio_strength):
    try:
        print("=== Starting Audio Pipeline Execution ===")
        
        # 1. Parse Environment variables
        input_bucket, input_key, output_bucket, output_key, region, temperature = get_data_from_env()
        
        # Override ref_audio_strength if temperature is provided securely from Env
        if temperature is not None:
            print(f"Using reference audio strength (temperature) from env: {temperature}")
            ref_audio_strength = float(temperature)

        # 2. S3 Setup / Data Dowload
        if input_bucket and input_key:
            print(f"Downloading source audio from S3: {input_bucket}/{input_key} ...")
            download_from_s3(input_bucket, input_key)
            print("Download successful.")
        else:
            print("No S3 input configuration provided in ENV. Relying on existing local file if any.")
            raise ValueError("No S3 input configuration provided in ENV")

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
        print(model_demo)

        # 4. Prepare data parameters
        print("Sampling generator parameters...")
        data_sampler = DataSampler()
        json_data = data_sampler.sample()
        json_data = get_input_data(json_data)
        print(json_data)

        (
            audio_duration,
            prompt,
            lyrics,
            infer_step,
            guidance_scale,
            scheduler_type,
            cfg_type,
            omega_scale,
            manual_seeds,
            guidance_interval,
            guidance_interval_decay,
            min_guidance_scale,
            use_erg_tag,
            use_erg_lyric,
            use_erg_diffusion,
            oss_steps,
            guidance_scale_text,
            guidance_scale_lyric,
        ) = json_data

        input_audio_path = f"/app/data/{input_key}"
        download_from_s3(input_bucket, input_key, input_audio_path)

        # 5. Output Path Formatting
        if output_path is None:
            input_p = Path(input_audio_path)
            output_path = str(input_p.parent / f"{input_p.stem}_output{input_p.suffix}")
        
        # Ensure target directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 6. Execute Model Inference
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

        # 7. S3 Upload generated output
        print(f"Uploading generated audio to S3 bucket: {output_bucket} ...")
        upload_to_s3(output_path, output_bucket, output_key)
        
        print("=== Audio Pipeline Execution Completed Successfully ===")
        sys.exit(0)
        
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
