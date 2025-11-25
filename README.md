<div align="center">
<h2>One-Step Diffusion Transformer for
Controllable Real-World Image Super-Resolution</h2>


<a href='https://arxiv.org/abs/2511.17138'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


Xiaohongshu Inc, Shanghai Jiao Tong University
</div>


## ⏰ Update
- **2025.11.25**: Arxiv link(including supplementary materials) is released.
- **2025.11.21**: code is released.


:star: If ODTSR is helpful to you, please help star this repo. Thanks! 

## 🌟 Overview Framework
![ODTSR_1](static/1.png)

1. Based on Qwen-Image, we train a **single-step** super-resolution model using LoRA, with the model size reaching 20B parameters.  
2. With our proposed NVS & FAA training strategies, the sr process can be jointly controlled by prompts, as well as a Fidelity Weight hyperparameter $f$.
3. Bilingual prompts are supported, and the model effectively inherits the capabilities of Qwen-Image, including strong performance in text rendering, textures, and other fine details.


## 😍 Visual Results
### Results with fixed prompts & high fidelity
<img src="static/4.jpeg" alt="" >

Under the high-fidelity setting with a fixed prompt, our model produces restorations that adhere more closely to the LQ input while remaining natural, significantly reducing the sense of ai processing.
### Text Real-ISR Results
<img src="static/2.jpeg" alt="">
In text scenarios, when the prompt specifies the text to be restored, the model automatically matches the LQ text and performs the restoration.

### Controllable Real-ISR Results

<img src="static/3.jpeg" alt="">


qualitative results of Controllable Real-ISR with prompt and adjustable Fidelity Weight (denoted as $f$) on Div2k-val
dataset. As f decreases from 1 to 0, detail generation and prompt adherence gradually strengthen.


## ⚙ Dependencies and Installation

1. prepare conda env:
```
conda create -n yourenv python=3.11
```
2. install pytorch (we recommend pytorch 2.6):
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  -f https://mirrors.aliyun.com/pytorch-wheels/cu124/
```
3. install this repo (based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main)):
```
cd xxxx/ODTSR
pip3 install -e . -v  -i https://mirrors.cloud.tencent.com/pypi/simple
```
note: It will automatically install the required packages based on the requirements file.

4. install basicsr(for training, dataloader):
```
pip install basicsr
```
note:
Because basicsr has not been updated for several years, there is an API bug that needs to be fixed.
You can apply the fix with the following command:
```
sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /opt/conda/lib/python3.11/site-packages/basicsr/data/degradations.py
```
Make sure to replace **/opt/conda** with the path to your own Conda environment.

5. download base model to your disk: [qwen-iamge](https://huggingface.co/Qwen/Qwen-Image/tree/main)

6. download base model to your disk: [wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main) (only training need)


## 🍭 Inference with script
> note: you need at least 40GB GPU memory to infer.Using CPU offload can reduce GPU memory usage, but it is not supported yet.

> Our code now supports tile-based processing (tile size: 512×512), enabling input of arbitrary resolutions and super-resolution at any scale factor.
```
sh examples/qwen_image/test_gan.sh
```
<img src="static/infer.png" alt="" >

## 🎦 Inference with gradio

```
sh examples/qwen_image/test_gradio.sh
```
<img src="static/gradio.jpeg" alt="" >

## 🔥 Training

to be updated


## License
This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
This project is based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main).
We also leveraged some of [PiSA-SR](https://github.com/csslc/PiSA-SR/tree/main)'s code in dataloader part.
Thanks for the awesome work. 
