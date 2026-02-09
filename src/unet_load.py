import torch
import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized
from diffusers import DiffusionPipeline, UNet2DConditionModel


mto.enable_huggingface_checkpointing()

compare_unet = {}

# load pipeline with original unet
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
compare_unet["original"] = pipe

# load pipeline with quantized unet
quantized_unet = UNet2DConditionModel.from_pretrained("./sd_turbo_nvfp4/unet")
quantized_unet.to(torch.float16)

pipe_qt_unet = DiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    unet=quantized_unet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
compare_unet["quantized"] = pipe_qt_unet

for name, module in compare_unet["quantized"].unet.named_modules():  
    if hasattr(module, 'weight_quantizer'):        
        print(f"{name}: quantized={is_quantized(module), type(module)}")

# gemerate images with both pipelines
prompt = "A cyberpunk city reflected in a puddle, neon lights, 8k resolution"

for key, pl in compare_unet.items():
    image = pl(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save(f"output_{key}.png")
