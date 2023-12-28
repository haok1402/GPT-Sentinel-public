# GPT-Sentinel: Distinguishing Human and ChatGPT Generated Content

Please note that this repository is no longer actively maintained. For the latest updates, please refer to our followup work:

- Paper: [Token Prediction as Implicit Classification to Identify LLM-Generated Text
](https://aclanthology.org/2023.emnlp-main.810/)
- Codebase: [T5-Sentinel-public](https://github.com/MarkChenYutian/T5-Sentinel-public)

## Overview

> :page_facing_up: [Link to Paper (arXiv)](https://arxiv.org/abs/2305.07969) | :floppy_disk: [Link to Dataset](https://drive.google.com/drive/folders/1Vnr-_nJWT4VXE-1wK38YSsCD4GcP6mk_?usp=share_link) | :package: [Link to Checkpoint](https://drive.google.com/drive/folders/17IPZUaJ3Dd2LzsS8ezkelCfs5dMDOluD?usp=share_link)

This repository is the codebase for paper *GPT-Sentinel: Distinguishing Human and ChatGPT Generating Content*.

1. We collect and publish **OpenGPTText** - a high quality dataset with approximately 30,000 text sample rephrased by `gpt-3.5-turbo` (ChatGPT).
2. We construct two detectors with different architectures - the **RoBERTa-Sentinel** and **T5-Sentinel**.
3. T5-Sentinel shows SOTA performance (**98% accuracy**) on OpenGPTText dataset

<p align="center">
<img width="718" alt="image" src="https://github.com/MarkChenYutian/GPT-Sentinel-public/assets/47029019/8a2125b6-3ba2-40ec-8310-52469f91aebd">
</p>
