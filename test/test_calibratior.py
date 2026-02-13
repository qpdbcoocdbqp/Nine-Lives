import torch
import os
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.config import INT8_DEFAULT_CFG, FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG
from diffusers import AutoPipelineForText2Image
from src.utils import Calibratior
from random import randint
import dotenv


dotenv.load_dotenv()
mto.enable_huggingface_checkpointing()

def main():
    # 1. Initialize Calibratior
    # We'll use a small sample size for demonstration
    calibratior = Calibratior(sample_size=128, seed=randint(1, 1000))
    calibratior_prompts = calibratior.get_calibration_prompts()
    
    model_id = os.getenv("MODEL_ID", "stabilityai/sd-turbo")
    configs = [
        ("int8", INT8_DEFAULT_CFG),
    ]

    # 3. Quantization Loop
    for fmt_name, config in configs:
        print(f"\n--- Starting {fmt_name.upper()} Quantization Process ---")
        
        # 1. Load a fresh Pipeline for each format 
        # (mtq.quantize modifies models in-place, so we need a clean model each time)
        print(f"Loading fresh model components for {fmt_name}...")

        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to("cuda")

        # Link the pipeline to the calibrator
        calibratior.set_pipeline(pipe)

        print(f"Quantizing UNet with {fmt_name} configuration...")
        with torch.no_grad():
            # mtq.quantize will call calibratior.calibrate_loop(quantized_unet)
            quantized_unet = mtq.quantize(
                pipe.unet, 
                config, 
                forward_loop=calibratior.calibrate_loop
            )

        # 2. Save the results
        output_dir = f"./quantized_sd_turbo_{fmt_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, "unet")

        quantized_unet.save_config(save_path)
        mto.save(quantized_unet, f"{save_path}/modelopt_model.pth")
        # quantized_unet.save_pretrained(save_path)
        mtq.compress(quantized_unet)
        mto.save(quantized_unet, f"{save_path}/modelopt_compressed_model.pth")
        
        print(f"✅ {fmt_name.upper()} Quantization complete! Model saved to: {save_path}")

        # 3. Cleanup GPU memory to avoid OOM for the next pass
        del pipe
        del quantized_unet
        torch.cuda.empty_cache()

    print("\nAll quantization tasks finished successfully.")

if __name__ == "__main__":
    main()
