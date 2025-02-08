LatentSync-main

LatentSync is a deep learning-based video and audio synchronization framework utilizing diffusion models.

Installation

Prerequisites

Python 3.10 or later

CUDA-compatible GPU (for acceleration)

ffmpeg installed and added to system path

Clone the Repository

git clone https://github.com/your-username/LatentSync-main.git
cd LatentSync-main

Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies

pip install -r requirements.txt

Running Inference

To run inference, execute the following command:

python -m scripts.inference \
  --unet_config_path configs/unet/second_stage.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --guidance_scale 1.0 \
  --video_path input-files/input_video.mp4 \
  --audio_path input-files/input_audio.mp3 \
  --video_out_path output-files/video_out.mp4 \
  --seed 42 \
  --superres CodeFormer

Arguments:

--unet_config_path: Path to UNet configuration file.

--inference_ckpt_path: Path to pre-trained checkpoint file.

--guidance_scale: Guidance scale for inference.

--video_path: Path to input video file.

--audio_path: Path to input audio file.

--video_out_path: Path to output synchronized video file.

--seed: Random seed for reproducibility.

--superres: Super-resolution model (default: CodeFormer).

Common Issues & Fixes

ImportError: cannot import name 'CrossAttention'

Ensure you have the correct version of diffusers installed:

pip install diffusers==0.19.3

ImportError: cannot import name 'cached_download'

Upgrade huggingface_hub:

pip install --upgrade huggingface_hub

License

This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments

Hugging Face Diffusers

PyTorch

ffmpeg
