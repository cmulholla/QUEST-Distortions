# use openai's gpt-3 to generate text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, prepare_model_for_int8_training
import pandas as pd
import numpy as np
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Load the model
#model = oi.Completion.create(model="gpt-4", max_tokens=512, temperature=0.7, top_p=0.1, frequency_penalty=1.2, presence_penalty=0.5, stop=["\n", "[INST]"])

# Additional context to be added to the input
additional_context = "The following is a conversation with an AI assistant. The assistant is helpful and concise. The assistant does not respond to the question, and only does as the question says"

# Generate outputs
def generate_response(text):
    try:
        #outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, use_cache=True, top_k=40, top_p=0.1, temperature=0.7, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
        response = client.chat.completions.create(
            model = "gpt-4",
            messages = text
        )

        #text = responseIn + "\n[INST] Please generate another sentence. [/INST]"
        #inputs = tokenizer(text, return_tensors="pt").to("cuda")
        #outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, use_cache=True, top_k=40, top_p=0.1, temperature=0.7, repetition_penalty=1.2, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
        #responseIn = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # get the response from the model
        response = response.choices[0].message.content
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
    messages = [
        {"role": "system", "content": additional_context},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": rand_records[0]},
        {"role": "assistant", "content": rand_records[1]},
        {"role": "assistant", "content": rand_records[2]}
    ]

    output = generate_response(messages)

    if len(output) == 0:
        output = "Sorry, I encountered an error. Please try again."

    # if the output contains a newline character within, remove it
    if output.find("\n") != -1:
        output = output.replace("\n", " ")

    return output


to_generate = 240

generate_data = [
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
                ]

# calculate total number of records to generate
total_records = 0
for i in range(len(generate_data)):
    total_records += to_generate - generate_data[i][1] if (to_generate - generate_data[i][1]) > 0 else 0
    print(f"Records to generate for {generate_data[i][0]}: {to_generate - generate_data[i][1] if (to_generate - generate_data[i][1]) > 0 else 0}")

print(f"Total records to generate: {total_records}")


data = pd.read_csv('Annotated_data.csv')
data = data.dropna()

# print the unique distortions
print(data["Dominant Distortion"].unique())

# start the timer
start = time.time()

# Create a DataFrame from the history list
df = pd.DataFrame([], columns=["Distorted part","Dominant Distortion"])

generated_records = 0

for c in range(len(generate_data)):

    generate = to_generate - generate_data[c][1] if (to_generate - generate_data[c][1]) > 0 else 0

    inputText = f"Please generate four similar sentences with only the \"{generate_data[c][0]}\" cognitive distortion within it, from the perspective of the person with the distortion."

    print(f"Generating {generate} records for {generate_data[c][0]}")

    for i in range(generate):

        new_output = generate_record(data, inputText, generate_data[c][0])
        print(generate_data[c][0] + " " + str(i+1) + " of " + str(generate) + f"({int(generated_records*100/total_records)}%): " + new_output)
        # Concatenate the new row to the DataFrame
        df = pd.concat([df, pd.DataFrame({"Distorted part": [new_output], "Dominant Distortion": [generate_data[c][0]]})], ignore_index=True)
        generated_records += 1
    
    # save the dataframe to a csv file
    df.to_csv("distorted_partsGPT4.csv", index=False)

# end the timer
end = time.time()

# Print the time taken
print(f"Time taken: {end - start} seconds")

# I am able to generate 2100, 60 word records with $10