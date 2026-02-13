import torch
import modelopt.torch.quantization as mtq
from diffusers import UNet2DConditionModel
from modelopt.torch.quantization.utils import is_quantized  
import modelopt.torch.opt as mto


# use modelopt to save pretrained checkpoints
mto.enable_huggingface_checkpointing()

# load the unet model in fp16 precision
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/sd-turbo",
    subfolder="unet",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# define a forward loop that generates random inputs and runs the model for a few iterations
def random_forward_loop(model, batch_size=1):
    model.eval()
    with torch.no_grad():
        for _ in range(32):
            latents = torch.randn(batch_size, 4, 64, 64, device="cuda", dtype=model.dtype)
            timesteps = torch.randint(1, 1000, (batch_size,), device="cuda")
            prompt_embeds = torch.randn(batch_size, 77, 1024, device="cuda", dtype=model.dtype)
            model(latents, timesteps, encoder_hidden_states=prompt_embeds).sample

# quantize the unet model using the NVFP4 configuration and the random forward loop for calibration
quantized_unet = mtq.quantize(unet, mtq.NVFP4_DEFAULT_CFG, forward_loop=random_forward_loop)

# check which modules were quantized
for name, module in quantized_unet.named_modules():  
    if hasattr(module, 'weight_quantizer'):  
        print(f"{name}: quantized={is_quantized(module)}")

# save the quantized unet model
quantized_unet.save_pretrained("./sd_turbo_nvfp4/unet")
