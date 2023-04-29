from modules import shared, script_callbacks
from os import path
from glob import glob
from pydantic import BaseModel, Field
from typing import List

class LoraItem(BaseModel):
    name: str = Field(title="Lora name")
    filename: str = Field(title="File name")

def list_loras():
    matches = glob(path.join(shared.cmd_opts.lora_dir, '**/*.pt'), recursive=True)
    matches += glob(path.join(shared.cmd_opts.lora_dir, '**/*.safetensors'), recursive=True)
    matches += glob(path.join(shared.cmd_opts.lora_dir, '**/*.ckpt'), recursive=True)

    resp = []
    for m in matches:
        if not path.isfile(m):
            continue
        
        name = path.splitext(path.basename(m))[0]
        resp.append(LoraItem(name=name, filename=path.abspath(m)))
    
    return resp

def app_started(demo, app):
    app.add_api_route("/sd_api_lora/lora", list_loras, methods=["GET"], response_model=List[LoraItem])

script_callbacks.on_app_started(app_started)
