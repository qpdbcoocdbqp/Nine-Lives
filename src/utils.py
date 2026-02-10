import torch
import os
import random
from datasets import load_dataset
from diffusers import AutoPipelineForText2Image
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG


class Calibratior:
    """
    A utility class for managing calibration datasets and performing the calibration loop 
    required for diffusion model quantization.
    """
    def __init__(self, sample_size=128, seed=42, pipe=None):
        """
        Initialize the Calibratior.

        Args:
            sample_size (int): The number of prompts to use for calibration.
            seed (int): The random seed for reproducibility.
            pipe: Optional diffusion pipeline to use for calibration.
        """
        self.sample_size = sample_size
        self.seed = seed
        self.pipe = pipe
        self.prompts = []

    def set_pipeline(self, pipe):
        """Set or update the pipeline used for calibration."""
        self.pipe = pipe

    def get_calibration_prompts(self, dataset_name="Gustavosta/Stable-Diffusion-Prompts"):
        """
        Load a dataset from Hugging Face and select a random subset of prompts.

        Args:
            dataset_name (str): The Hugging Face dataset identifier.
        
        Returns:
            list: The list of selected prompts.
        """
        print(f"Loading dataset: {dataset_name}...")
        
        try:
            dataset = load_dataset(dataset_name, split="train")
            # Extract 'Prompt' field from the dataset
            all_prompts = dataset["Prompt"]
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            return []
        
        # Select random samples
        random.seed(self.seed)
        if len(all_prompts) > self.sample_size:
            self.prompts = random.sample(all_prompts, self.sample_size)
        else:
            self.prompts = all_prompts
        
        print(f"Successfully selected {len(self.prompts)} calibration prompts.")
        return self.prompts

    def save_prompts(self, file_path="calib_prompts.txt"):
        """
        Save the current prompts list to a text file.

        Args:
            file_path (str): The path to the output text file.
        """
        if not self.prompts:
            print("No prompts available to save. Please load prompts first.")
            return

        with open(file_path, "w", encoding="utf-8") as f:
            for p in self.prompts:
                # Remove internal newlines and strip whitespace
                clean_p = p.replace("\n", " ").strip()
                if clean_p:
                    f.write(f"{clean_p}\n")
        
        print(f"Prompts saved to {file_path}")

    def load_prompts(self, file_path, count=None):
        """
        Load calibration prompts from a text file.

        Args:
            file_path (str): The path to the prompts file.
            count (int, optional): The maximum number of prompts to load. Defaults to self.sample_size.
        
        Returns:
            list: The list of loaded prompts.
        """
        if count is None:
            count = self.sample_size

        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Using default fallback prompts.")
            self.prompts = ["A high quality photo of a cat", "A beautiful landscape", "Astronaut in space"] * (count // 3 + 1)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
            print(f"Successfully loaded {len(self.prompts)} calibration prompts.")
        
        self.prompts = self.prompts[:count]
        return self.prompts

    def calibrate_loop(self, model=None):
        """
        Run the calibration inference loop. 
        Compatible with mtq.quantize(forward_loop=calibratior.calibrate_loop).

        Args:
            model: The model being quantized (passed automatically by mtq.quantize).
        """
        if not self.prompts:
            print("No prompts loaded for calibration. Please call get_calibration_prompts or load_prompts first.")
            return

        inference_executor = self.pipe if self.pipe is not None else model
        if inference_executor is None:
            print("Error: No pipeline or model available for calibration loop.")
            return

        print(f"Starting calibration inference on {len(self.prompts)} prompts...")
        
        for i, prompt in enumerate(self.prompts):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(self.prompts)}")
            
            # Truncate prompt using the executor's tokenizer
            if hasattr(inference_executor, "tokenizer"):
                tokens = inference_executor.tokenizer.encode(prompt, add_special_tokens=False)
                # CLIP limit 77 - 2 = 75
                if len(tokens) > 75:
                    prompt = inference_executor.tokenizer.decode(tokens[:75], skip_special_tokens=True)

            # Perform inference
            # If inference_executor is a pipe, this runs the full diffusion process
            # which ensures activation statistics are collected for the UNet.
            inference_executor(prompt, num_inference_steps=1, guidance_scale=0.0)
        
        print("Calibration inference completed successfully.")

if __name__ == "__main__":
    # Example: How to fetch and save calibration prompts
    calibratior = Calibratior(sample_size=128)
    calibratior.get_calibration_prompts()
    calibratior.save_prompts("calib_prompts.txt")
