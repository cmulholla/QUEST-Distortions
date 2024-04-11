import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, prepare_model_for_int8_training
import pandas as pd
import numpy as np

# Initialize the tokenizer and model
#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "teknium/OpenHermes-2-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
precision = "fp4-"
path=f"N:\\AI\\text-generation-webui-main\\models\\teknium_OpenHermes-2-Mistral-7B\\"
#path=f"N:\\AI\\mistral-7B-instruct\\"

# if the model variable exists, delete it to free up memory before loading the new model
if 'model' in locals():
    model = None

if (precision == "fp16"):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
elif (precision == "fp8"):
    model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True, device_map='cuda')
elif (precision == "fp4"):
    model = AutoModelForCausalLM.from_pretrained(path, load_in_4bit=True, device_map='cuda')


additional_context = "[INST] The following is a conversation with an AI assistant. The assistant is helpful and concise. The assistant does not respond to the question, and only does as the question says. [/INST]"
# Generate outputs
def generate_response(text):
    try:
        # Append a prompt to the user's input
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, use_cache=True, top_k=40, top_p=0.1, temperature=0.7, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
        responseIn = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        #text = responseIn + "\n[INST] Please generate another sentence. [/INST]"
        #inputs = tokenizer(text, return_tensors="pt").to("cuda")
        #outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, use_cache=True, top_k=40, top_p=0.1, temperature=0.7, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
        #responseIn = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the start of the response
        response = responseIn[len(text):]
    except Exception as e:
        response = "Sorry, I encountered an error. Please try again."
        print(e)
    return response

# function to generate a record based on 3 random records from the original data
def generate_record(original_data, input_text, distortion):
    distorion_data = original_data[original_data["Dominant Distortion"] == distortion]
    random_indices = np.random.choice(len(distorion_data), 3, replace=False)

    # Get the 3 random records
    rand_records = []
    for index in random_indices:
        rand_records.append(distorion_data.iloc[index]["Distorted part"])

    # Generate a response
    main_input = additional_context+"\n"+"[INST] " + input_text + " [/INST]\n\n" + rand_records[0] + "\n\n" + rand_records[1] + "\n\n" + rand_records[2] + "\n\n"

    output = generate_response(main_input)

    if len(output) == 0:
        output = "Sorry, I encountered an error. Please try again."

    if output[0] == '\n':
        output = output[1:]

    return generate_response(main_input)


to_generate = 200

"""generate_data = [
                    ("Mind Reading", 239),
                    ("Overgeneralization", 239),
                    ("Magnification", 195),
                    ("Labeling", 165),
                    ("Personalization", 153),
                    ("Fortune-telling", 143),
                    ("Emotional Reasoning", 134),
                    ("Mental filter", 122),
                    ("Should statements", 107),
                    ("All-or-nothing thinking", 100)
                ]"""

# perform the generation in blocks to avoid memory issues
generate_data = [("No Distortion", 100),
                 ("No Distortion", 100),
                 ("No Distortion", 100),
                 ("No Distortion", 100),
                 ("No Distortion", 100),
                 ("No Distortion", 100),
                 ("No Distortion", 200-64)]

# calculate total number of records to generate
total_records = 0
for i in range(len(generate_data)):
    total_records += to_generate - generate_data[i][1] if (to_generate - generate_data[i][1]) > 0 else 0
    print(f"Records to generate for {generate_data[i][0]}: {to_generate - generate_data[i][1] if (to_generate - generate_data[i][1]) > 0 else 0}")

print(f"Total records to generate: {total_records}")


data = pd.read_csv('Annotated_data.csv')

# print the unique distortions
print(data["Dominant Distortion"].unique())

# start the timer
start = time.time()

# Create a DataFrame from the history list
df = pd.DataFrame([], columns=["Patient Question","Distorted part","Dominant Distortion"])

# if the Distorted part column is empty, fill it with the Patient Question
data["Distorted part"] = data["Distorted part"].fillna(data["Patient Question"])

generated_records = 0

for c in range(len(generate_data)):

    generate = to_generate - generate_data[c][1] if (to_generate - generate_data[c][1]) > 0 else 0

    print(f"Generating {generate} records for {generate_data[c][0]}")

    # \"{generate_data[c][0]}\"
    inputText = f"Please generate four sentences with "+"no"+" cognitive distortions within it, from the perspective of the person. Each sentence should be unique but similar in style to the others."

    for i in range(generate):

        new_output = generate_record(data, inputText, generate_data[c][0])
        print(generate_data[c][0] + " " + str(i+1) + " of " + str(generate) + f"({int(generated_records*100/total_records)}%): " + new_output)
        # Concatenate the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame({"Distorted part": [new_output], "Dominant Distortion": [generate_data[c][0]]})], ignore_index=True)
        generated_records += 1
    
    # save the dataframe to a csv file
    df.to_csv("distorted_partsNoDistort.csv", index=False)

# end the timer
end = time.time()

# Print the time taken
print(f"Time taken: {end - start} seconds")

# time taken for 240 records: 2 hours