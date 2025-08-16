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
    .uv_pip_install([
        "datasets<4.0.0", "tokenizers", "tqdm", "numpy",
        "torch", "accelerate", "transformers", "wandb",
    ])
    .add_local_dir(os.path.dirname(__file__), "/root", copy=True)
)

volumes = {
    "/mnt/william": modal.Volume.from_name("william")
}

app = modal.App(image=image, volumes=volumes)

@app.function(
    cpu=64,
    # memory=512,
    memory=1024*84,
    volumes=volumes,
    timeout=60*60*24,
)
def load_data():
    print("Starting load_data.py...")
    exec(open("load_data/load_data.py").read(), globals())
    volumes["/mnt/william"].commit()

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
    volumes["/mnt/william"].commit()


@app.function(
    cpu=32,
    memory=1024*32,
    volumes=volumes,
    timeout=60*60*24,
)
def tokenize_data():
    print("Starting tokenize_data.py...")
    exec(open("tokenize/tokenize_data.py").read(), globals())
    volumes["/mnt/william"].commit()


n_gpus = 8
@app.function(
    cpu=12,
    gpu=f"h200:{n_gpus}",
    memory=1024*48,
    volumes=volumes,
    timeout=60*60*24,
    secrets=[modal.Secret.from_name("william-wandb-key")]
)
def train_model():
    print("Starting train_model.py...")
    subprocess.run([
        "accelerate", "launch",
        "--num_processes", str(n_gpus),
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        "train/train.py",
    ])
    volumes["/mnt/william"].commit()
