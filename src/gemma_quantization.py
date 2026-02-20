import dotenv
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import os
import torch
from random import randint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.utils import Calibratior


dotenv.load_dotenv()

def generate(model, tokenizer, prompt, title="Model Output"):
    """Standardized generation function."""
    print(f"\n--- {title} ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_response = response[len(prompt):].strip() if response.startswith(prompt) else response
    print(f"Response: {clean_response}")
    return clean_response

def save_quantized(model, path_dir):
    model.config.save_pretrained(path_dir)
    tokenizer.save_pretrained(path_dir)    
    mto.save(model, os.path.join(path_dir, "modelopt_compressed_model.pth"))

def load_quantized(path_dir):
    """Demonstrates loading the saved quantized model."""

    print(f"\n{'='*15} RELOADING QUANTIZED MODEL {'='*15}")
    tokenizer = AutoTokenizer.from_pretrained(path_dir, trust_remote_code=True) 
    config = AutoConfig.from_pretrained(path_dir)
    model = AutoModelForCausalLM.from_config(config)
    ckpt = os.path.join(path_dir, "modelopt_compressed_model.pth")
    print(f"Restoring weights and state from: {ckpt}")
    mto.restore(model, ckpt)
    
    # Ensure all newly restored parameters (like scales) are moved to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-270m-it-qat-q4_0-unquantized")
OUTPUT_DIR = "./gemma-3-270m-it-int4-quantized"
TEST_PROMPT = "Explain the importance of model quantization for edge devices."


if __name__ == "__main__":
    # Base Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cuda")
    _ = model.eval()
    base_res = generate(model, tokenizer, TEST_PROMPT, "Original Model (BF16)")

    # Prepare calibratior
    calibratior = Calibratior(
        sample_size=128, seed=randint(1, 1000),
        tokenizer=tokenizer
        )
    calibratior_prompts = calibratior.get_calibration_prompts()

    # Quantization process
    with torch.no_grad():
        _ = mtq.quantize(
            model,
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            # mtq.NVFP4_DEFAULT_CFG,
            forward_loop=calibratior.calibration_with_tokenizer
        )
    mtq.compress(model)

    # save quantize model
    save_quantized(model, path_dir=OUTPUT_DIR)

    quant_res = generate(model, tokenizer, TEST_PROMPT, "Quantized Model (INT4)")

    # Final Comparison
    print(f"\n{'='*50}\nPRE vs POST QUANTIZATION\n{'='*50}")
    print(f"ORIGINAL:\n{base_res}\n{'-'*20}\nQUANTIZED:\n{quant_res}\n{'='*50}")

    # Reload Test
    reloaded_model, reloaded_tok = load_quantized(OUTPUT_DIR)
    generate(reloaded_model, reloaded_tok, TEST_PROMPT, "Reloaded Model Verification")
