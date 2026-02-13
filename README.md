# Nine-Lives
To do modelopt qunatization. Playing with [Nine Lives](https://www.youtube.com/watch?v=bFhYJEOI0L4).

* **About Nine Lives**

> Nine Lives·Aerosmith·Nine Lives

## Reference

* [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
* [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
* [Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)

## Model-Optimizer

* **Setup**

  * Recommend to run at Linux. On Windows, can try WSL.
  * Python setup

    ```sh
    uv venv --python 3.13
    uv pip install nvidia-modelopt[all]
    ```

  * Download demo diffusion model

    ```sh
    hf download stabilityai/sd-turbo --include='.*fp16.safetensor'
    ```

  * Download calibration dataset for quantization

    ```sh
    # only parquet file (10.3 MB)
    hf download Gustavosta/Stable-Diffusion-Prompts
    ```  

* **Example**

  * `stabilityai/sd-turbo`: `unet` quantization. Because ModelOpt only support `INT8` quantization for `CNN` network. So only `INT8_DEFAULT_CFG` configuration can be used.
  
* Simple quantize workflow

  * `test_calibratior` show

    * `quantize`
      ```mermaid
      ---
      title: quantize
      ---
      graph LR
      a[torch.Model]
      b{{modelopt.torch.quantization.quantize}}
      c[quantized model]
      a --> b --> c
      ```

    * `save`
      ```mermaid
      ---
      title: save model
      ---
      graph LR
      a[quantized model]
      b{{modelopt.torch.opt.enable_huggingface_checkpointing}}
      c{{modelopt.torch.opt.save}}
      d[pth file]
      a --> b --> c --> d
      ```

  * `test_meaturement` show
    * `load`
      ```mermaid
      ---
      title: load model
      ---
      graph LR
      a[pth file]
      b{{modelopt.torch.opt.enable_huggingface_checkpointing}}
      c{{modelopt.torch.opt.restore}}
      d{{modelopt.torch.quantization.compress}}
      a --> b --> c --> d
      ```
