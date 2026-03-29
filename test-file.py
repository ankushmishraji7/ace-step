import click
import os
from pathlib import Path

from utils.s3_utils import upload_to_s3, download_from_s3
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


def sample_data(json_data):
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
@click.option("--input_audio_path", type=str, required=True, help="Path to input audio file (from data folder)")
@click.option("--output_path", type=str, default=None, help="Path to save the output (usually in data folder)")
@click.option("--ref_audio_strength", type=float, default=0.5, help="Strength of the reference audio input (0.0 to 1.0)")
def main(checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode, device_id, input_audio_path, output_path, ref_audio_strength):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    print(model_demo)

    data_sampler = DataSampler()

    json_data = data_sampler.sample()
    json_data = sample_data(json_data)
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

    # Download input file from S3
    input_bucket_name = "ai-generated-audio"
    input_file_name = "test_track_001.mp3"
    download_from_s3(input_bucket_name, input_file_name, input_audio_path)

    # Set default output path automatically if not provided
    if output_path is None:
        input_p = Path(input_audio_path)
        output_path = str(input_p.parent / f"{input_p.stem}_output{input_p.suffix}")

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
        # Parameters for audio-to-audio
        audio2audio_enable=True,
        ref_audio_input=input_audio_path,
        ref_audio_strength=ref_audio_strength
    )

    print(f"Inference completed. Output saved to {output_path}")

    # Upload to S3
    generated_audio_bucket_name = "ai-generated-audio"
    upload_to_s3(output_path, generated_audio_bucket_name)

if __name__ == "__main__":
    main()
