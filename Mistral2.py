import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# start timer
start = time.time()

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
precision = "fp16"

if (precision == "fp16"):
    model = AutoModelForCausalLM.from_pretrained(f"N:\AI\mistral-7B-instruct", torch_dtype=torch.float16).to(0)
elif (precision == "fp32"):
    model = AutoModelForCausalLM.from_pretrained(f"N:\AI\mistral-7B-instruct")

def generate_response(text):
    # Append a prompt to the user's input
    text = "[INST] " + text + " [/INST]"
    if (precision == "fp16"):
        inputs = tokenizer(text, return_tensors="pt").to(0)
    elif (precision == "fp32"):
        inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, use_cache=True, top_k=50, top_p=0.95, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the start of the response
    response = response[len(text):]
    return response

print(f"Running {model_id.split('/')[1]} with 16-bit precision...")


#print(f"{model_id.split('/')[1]}: " + generate_response(f"Hello! My name is Connor, can you introduce yourself as an AI named {model_id.split('/')[1]}? Please keep the response short."))

# end timer
end = time.time()
print(f"Time taken: {end - start} seconds")

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("Freeing up memory...")
        break

    # start timer
    start = time.time()

    save = False
    if user_input.startswith("--save "):
        save = True
        user_input = user_input.replace("--save ", "")

    response = generate_response(user_input)
    print(f"{model_id.split('/')[1]}: {response}")

    # end timer
    end = time.time()
    print(f"Time taken: {((end - start)/60):.2f} minutes")

    if save:
        with open("response.txt", "a") as file:
            file.write(response + "\n")