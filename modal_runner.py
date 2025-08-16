import modal
import subprocess
import os
import sys

# Force tqdm to show progress bars in Modal containers
import tqdm
tqdm.tqdm._instances.clear()
original_isatty = sys.stdout.isatty
sys.stdout.isatty = lambda: True
sys.stderr.isatty = lambda: True

image = (
    modal.Image.debian_slim()
    .uv_pip_install(["datasets<4.0.0", "tokenizers", "tqdm", "numpy", "torch", "accelerate"])
    .add_local_dir(os.path.dirname(__file__), "/root", copy=True)
)

volumes = {
    "/mnt/william": modal.Volume.from_name("william")
}

app = modal.App(image=image, volumes=volumes)

@app.function(
    cpu=2,
    memory=512,
    volumes=volumes,
    timeout=60*60*24,
)
def load_data():
    print("Starting load_data.py...")
    subprocess.run(["python", "load_data/load_data.py"])

@app.function(
    cpu=10,
    # memory=1024*336,
    memory=1024*84,
    volumes=volumes,
    timeout=60*60*24,
)
def train_tokenizer():
    print("Starting train_tokenizer.py...")
    # 7 min to load data, ~20 min to train tokenizer
    exec(open("tokenize/train_tokenizer.py").read(), globals())
