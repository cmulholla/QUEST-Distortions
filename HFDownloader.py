import requests
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a dictionary to store the average speeds
download_progress = {}

# Function to download a model file
def download_model(model_id, file_name, save_folder, progress_percentage):
    url = f"https://huggingface.co/{model_id}/resolve/main/{file_name}"
    save_path = os.path.join(save_folder, file_name)

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        # Send a GET request to the URL and stream the response
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        remaining_size = total_size
        block_size = 1024  # 1 KB
        downloaded_size = 0
        last_downloaded_size = 0
        counter = 0
        start_time = time.time()
        last_progress_time = start_time  # Track the time of the last progress ping

        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    downloaded_size += len(data)
                    file.write(data)
                    progress = int((downloaded_size / total_size) * 100)
                    if progress > counter and progress % progress_percentage == 0:
                        counter = progress
                        elapsed_time = time.time() - last_progress_time  # Calculate time since last progress ping
                        average_speed = ((downloaded_size - last_downloaded_size) / elapsed_time) / (1024 * block_size)  # Calculate average speed
                        remaining_size = (total_size - downloaded_size) / (1024 * 1024)  # Calculate remaining size in MB
                        remaining_time = remaining_size / average_speed
                        # Print download progress, average speed, and remaining time
                        print(f"Download progress: {counter}%, Speed: {average_speed:.2f} MB/s, Remaining Time: {int(remaining_time/60)}m {int(remaining_time%60)}s, Time Since Last Progress Ping: {int(elapsed_time)}s")
                        # Store the progress and speed in the dictionary
                        if file_name not in download_progress:
                            download_progress[file_name] = []
                        download_progress[file_name].append((counter, average_speed))
                        last_progress_time = time.time()  # Reset the timer for the next progress ping
                        last_downloaded_size = downloaded_size  # Reset the downloaded size for the next progress ping

            # Print success message and return
            print(f"File '{file_name}' downloaded successfully and saved to '{save_path}'.")
            return
        else:
            # Print failure message and retry
            print(f"Failed to download file '{file_name}'. Retrying...")
            retry_count += 1
            time.sleep(20)

    # if we reach here, it means we failed to download the file
    print(f"Failed to download file '{file_name}' after {max_retries} attempts. Skipping download.")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
save_folder = f"N:/AI/mistral-7B-instruct/"
max_files = 3
progress_percentage = 2

# Download each safetensors file
for i in range(1, max_files + 1):
    file_name = f"model-{str(i).zfill(5)}-of-{str(max_files).zfill(5)}.safetensors"
    print("Downloading: ", file_name)
    download_model(model_id, file_name, save_folder, progress_percentage)

# Convert the dictionary to a DataFrame
data = []
for file_name, progress in download_progress.items():
    for percentage, speed in progress:
        data.append({"File": file_name, "Percentage": percentage, "Speed": speed})
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Percentage", y="Speed", hue="File")
plt.show()