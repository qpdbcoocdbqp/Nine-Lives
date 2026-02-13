import os
import torch
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import dotenv


dotenv.load_dotenv()
mto.enable_huggingface_checkpointing()

def get_model_mem_size(model):
    return sum(p.element_size() * p.nelement() for p in model.parameters())


model_id = os.getenv("MODEL_ID", "stabilityai/sd-turbo")
# load original and quantized UNet
original_unet = UNet2DConditionModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    variant="fp16", 
    subfolder="unet"
).to("cuda")

config = UNet2DConditionModel.load_config("./quantized_sd_turbo_int8/unet/config.json")
unet = UNet2DConditionModel.from_config(config)
quantized_unet = mto.restore(unet, "./quantized_sd_turbo_int8/unet/modelopt_model.pth")
quantized_unet = quantized_unet.to(torch.float16).to("cuda")
mtq.compress(quantized_unet)

print(f"Original UNet (Memory, FP16): {get_model_mem_size(original_unet) / 2**20:.2f} MB")
print(f"Quantized UNet (Memory, INT8): {get_model_mem_size(quantized_unet) / 2**20:.2f} MB")

pipe_fp16 = AutoPipelineForText2Image.from_pretrained(
    model_id,
    unet=original_unet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe_int8 = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe_int8.unet = quantized_unet

torch.cuda.empty_cache()

# Run inference
print("Pipeline already, using FP16 UNet. Running test inference...")
image_fp16 = pipe_fp16(
    "a photo of an astronaut riding a horse on mars",
    num_inference_steps=2, guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]
output_filename = f"test_image_fp16.png"
image_fp16.save(output_filename)
print(f"✅ Test image saved to {output_filename}")

print("Pipeline already, using INT8 UNet. Running test inference...")
image_int8 = pipe_int8(
    "a photo of an astronaut riding a horse on mars",
    num_inference_steps=2, guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]
output_filename = f"test_image_int8.png"
image_int8.save(output_filename)
print(f"✅ Test image saved to {output_filename}")
