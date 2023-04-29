from modules import shared, script_callbacks
from os import path
from glob import glob
from pydantic import BaseModel, Field
from typing import List

class LoraResponse(BaseModel):
    names: List[str] = Field(title="Names", description="List of LoRA names available")

def list_loras():
    candidates = \
            glob(path.join(shared.cmd_opts.lora_dir, '**/*.pt'), recursive=True) + \
            glob(path.join(shared.cmd_opts.lora_dir, '**/*.safetensors'), recursive=True) + \
            glob(path.join(shared.cmd_opts.lora_dir, '**/*.ckpt'), recursive=True)

    return LoraResponse(names=[path.splitext(path.basename(c))[0] for c in candidates])
    

def app_started(demo, app):
    app.add_api_route("/sd_api_lora/lora", list_loras, methods=["GET"], response_model=LoraResponse)

script_callbacks.on_app_started(app_started)
