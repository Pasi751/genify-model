from fastapi import FastAPI
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("Pasindu751/genify-llama2-q8_0", model_file="myllama-7b-v0.1.gguf", model_type="llama", gpu_layers=0,max_new_tokens=128)


app = FastAPI()

class Prompt(BaseModel):
    prompt: str

def generate(prompt):
    output = llm(prompt)
    return output

@app.post("/generate_text/")
async def generate_text(prompt: Prompt):
    # Generate text based on the prompt
    generated_text = generate(prompt)
    return {"generated_text": generated_text}

@app.get("/hello")
async def read_item():
    return {"message": "Welcome to our app"}

