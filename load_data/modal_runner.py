import modal
import subprocess
import os

image = (
    modal.Image.debian_slim()
    .uv_pip_install(["datasets"])
    .add_local_file("load_data/load_data.py", "/root/load_data.py", copy=True)
    .add_local_file("load_data/test.py", "/root/test.py", copy=True)
)
app = modal.App(image=image)

volumes = {
    "/mnt/william": modal.Volume.from_name("william")
}

@app.function(
    cpu=2,
    memory=512,
    volumes=volumes,
    timeout=60*60*24,
)
def main():
    print("Starting load_data.py...")
    subprocess.run(["python", "load_data.py"])
