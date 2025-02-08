# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth")
        os.system("ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")

        # Load super-resolution models if needed
        self.superres = None
        if self.superres == "GFPGAN":
            from gfpgan import GFPGANer
            self.gfpgan = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=2, device='cuda')
        elif self.superres == "CodeFormer":
            from basicsr.archs.codeformer_arch import CodeFormer
            self.codeformer = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9).to('cuda')
            self.codeformer.load_state_dict(torch.load('models/codeformer.pth'))
            self.codeformer.eval()


    def predict(
            
        
        self,
        video: Path = Input(
            description="Input video", default=None
        ),
        audio: Path = Input(
            description="Input audio to ", default=None
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=10, default=1.0
        ),
        seed: int = Input(
            description="Set to 0 for Random seed", default=0
        ),
        superres: str = Input(
        description="Super-resolution method (GFPGAN or CodeFormer)", default=None, choices=["GFPGAN", "CodeFormer"]
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/second_stage.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "output-files/video_out.mp4"
    
        # Run the following command:
        os.system(f"python -m scripts.inference --unet_config_path {config_path} --inference_ckpt_path {ckpt_path} --guidance_scale {str(guidance_scale)} --video_path {video_path} --audio_path {audio_path} --video_out_path {output_path} --seed {seed} --superres {superres}")           
        return Path(output_path)
    
