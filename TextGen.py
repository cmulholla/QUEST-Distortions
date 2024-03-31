import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, prepare_model_for_int8_training
import pandas as pd


# Initialize the tokenizer and model
#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "teknium/OpenHermes-2-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
precision = "fp4"
path=f"N:\\AI\\text-generation-webui-main\\models\\teknium_OpenHermes-2-Mistral-7B\\"
#path=f"N:\\AI\\mistral-7B-instruct\\"

to_generate = 150

generate_data = [
                    ("Mind reading", 239),
                    ("Overgeneralization", 239),
                    ("Magnification", 195),
                    ("Labeling", 165),
                    ("Personalization", 153),
                    ("Fortune-telling", 143),
                    ("Emotional Reasoning", 134),
                    ("Mental filter", 122),
                    ("Should statements", 107),
                    ("All-or-nothing thinking", 100)
                ]

# calculate total number of records to generate
total_records = 0
for i in range(len(generate_data)):
    total_records += 150 - generate_data[i][1] if (150 - generate_data[i][1]) > 0 else 0
    print(f"Records to generate for {generate_data[i][0]}: {150 - generate_data[i][1] if (150 - generate_data[i][1]) > 0 else 0}")

print(f"Total records to generate: {total_records}")

# if the model variable exists, delete it to free up memory before loading the new model
if 'model' in locals():
    model = None

if (precision == "fp16"):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
elif (precision == "fp8"):
    model = AutoModelForCausalLM.from_pretrained(path,  load_in_8bit=True, device_map='cuda')
elif (precision == "fp4"):
    model = AutoModelForCausalLM.from_pretrained(path,  load_in_4bit=True, device_map='cuda')


# start the timer
start = time.time()

# Example input text
additional_context = "[INST] The following is a conversation with an AI assistant. The assistant is helpful and concise. The assistant does not respond to the question, and only does as the question says. [/INST]"
input_text = "Please generate a single sentence with only the \"Mental Filters\" cognitive distortion, from the perspective of the person with the distortion."

# Generate outputs
def generate_response(text):
    try:
        # Append a prompt to the user's input
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, use_cache=True, top_k=40, top_p=0.1, temperature=0.7, repetition_penalty=1.25, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
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


# Create a DataFrame from the history list
df = pd.DataFrame([], columns=["Distorted part","Dominant Distortion"])

generated_records = 0

for c in range(len(generate_data)):

    # Generate a response
    main_input = additional_context+"\n"+"[INST] " + input_text + " [/INST]"
    output = generate_response(main_input)


    # Create a list to store inputs and outputs
    history = []

    # Add the initial input and output to the history
    history.append((main_input, output))

    generate = 150 - generate_data[c][1] if (150 - generate_data[c][1]) > 0 else 0

    print(f"Generating {generate} records for {generate_data[c][0]}")

    for i in range(generate):
        # Generate a new response
        input_string = ""
        for j in range(len(history)):
            input_string += history[j][0] + "\n" + history[j][1] + "\n"
        input_string = input_string + "[INST] " + "Please generate another completely new sentence with only the \"" + generate_data[c][0] + "\" cognitive distortion." + " [/INST]\n"

        if input_string.__len__() > 8192:
            # Remove the string from the start of the end of the first string
            input_string = input_string[0:main_input.__len__()] + input_string[input_string.find("[INST]", main_input.__len__()+(input_string.__len__() - 8192)):]


        new_output = generate_response(input_string)

        print(generate_data[c][0] + " " + str(i+1) + " of " + str(generate) + f"({int((generated_records*100.0)/total_records)/100.0}%): " + new_output)

        if new_output.__len__() > 1:
            if new_output[0] == '\n':
                new_output = new_output[1:]
        
        if "[INST]" in new_output:
            # Add 1 to the counter to repeat the same iteration
            i -= 1
        else:
            # Concatenate the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame({"Distorted part": [new_output], "Dominant Distortion": [generate_data[c][0]]})], ignore_index=True)
            generated_records += 1


        # Add the new input and output to the history
        history.append(("[INST] Please generate another sentence with only the \"" + generate_data[c][0] + "\" cognitive distortion. [/INST]", new_output))
    
    # save the dataframe to a csv file
    df.to_csv("distorted_parts.csv", index=False)

# end the timer
end = time.time()

# Print the time taken
print(f"Time taken: {end - start} seconds")

# DP3 took 8680 seconds (~2hrs) to generate 800 records, or 10.85 seconds per record