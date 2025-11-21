<div align="center">
<h2>One-Step Diffusion Transformer for
Controllable Real-World Image Super-Resolution</h2>


<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


Xiaohongshu Inc, Shanghai Jiao Tong University
</div>


## ⏰ Update
- **2025.11.21**: code is released.


:star: If ODTSR is helpful to you, please help star this repo. Thanks! :hugs:

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

to be updated

## 🍭 Inference with script

to be updated

## 🎦 Inference with gradio

to be updated

## 🔥 Training

to be updated


## License
This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
This project is based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main).
We also leveraged some of [PiSA-SR](https://github.com/csslc/PiSA-SR/tree/main)'s code in dataloader part.
Thanks for the awesome work. 
