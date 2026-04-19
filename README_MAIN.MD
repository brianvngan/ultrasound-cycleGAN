There are 3 pipelines available for synthetic data generation:
	1.	Stable Diffusion 2.1 (sd2.1_spleen/)
	2.	CycleGAN (SpleenCycleGan.ipynb)
	3.	DDPM (ddpm.ipynb)

Stable Diffusion 2.1 (sd2.1_spleen/):
This folder contains the Stable Diffusion 2.1 pipeline for generating synthetic spleen ultrasound images. For detailed setup and usage instructions, see the README inside the sd2.1_spleen/ directory.

CycleGAN:
Inside the SpleenCycleGan file is the pipeline for creating synthetic spleen images using GANs. There are directions on how the model works within the file.

DDPM (ddpm_spleen/ddpm.ipynb):
 Denoising Diffusion Probabilistic Model with a U-Net backbone and attention, trained from scratch on spleen ultrasound images using processed roboflow data from team5-data.  For detailed setup please see the README inside the ddpm_spleen directory.

 ## Acknowledgements
- [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion) — Ho et al., 2020
- [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — Zhu et al., 2017