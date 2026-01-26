import asyncio
import json
import toml
import argparse
import random
import time
from datetime import datetime
from ollama import AsyncClient
from pathlib import Path
from tqdm.asyncio import tqdm

def log_message(message):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}, {message}", flush=True)

def save_response(prompt, response_text, output_dir, doi=None):
    """Saves the response to a JSON file with a unique filename."""
    # Generate a unique filename using the current epoch time and a random element
    if doi:
        filename = f"{doi.strip()}.json"  # '/', '(', ')' should all be converted already, so no need convert again
    else:
        epoch = int(time.time())
        random_suffix = random.randint(1000, 9999)
        filename = f"{epoch}-{random_suffix}.json"
    
    # Create the output dictionary
    output_data = {
        "prompt": prompt,
        "response": response_text
    }
    
    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the response to the JSON file
    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
        json_file.flush()

async def chat(system_msg, message, host, gpu_index, model, output_dir, doi, pbar):
    try:

        # Make the API request using the specified host and model
        response = await AsyncClient(host=f"http://{host}").chat(
            model=model,
            messages=[system_msg, message],
            options={"temperature": 0, "num_predict": 2_000}
        )

        # Extract the response content
        response_text = response['message']['content']
        prompt_text = message['content']

        # Save the response to a JSON file
        save_response(prompt_text, response_text, output_dir, doi)


    except Exception as e:
        log_message(f"Error on host {host}: {e}")
    finally:
        # Update progress bar
        pbar.update(1)

async def worker(host, gpu_index, model, task_queue, output_dir, pbar):
    """Worker function to process tasks using the specified host."""
    while not task_queue.empty():
        try:
            system_msg, message, doi = task_queue.get_nowait()
            await chat(system_msg, message, host, gpu_index, model, output_dir, doi, pbar)
            task_queue.task_done()
        except asyncio.QueueEmpty:
            break

async def main(config_path, prompts_path, dois_path, output_dir):
    # Get configuration  
    config = toml.load(config_path)
    model = config.get("model", "llama3.2")
    gpus = config["ollama_instances"]
    system_msg = json.loads(f'{{"role": "system", "content": {json.dumps(config.get("system_message"))}}}')

    # Load prompts from JSONL file
    prompts = []
    with open(prompts_path, "r") as file:
        for line in file:
            prompts.append(json.loads(line))

    # Load dois from text file
    dois = []
    with open(dois_path, "r") as file:
        for line in file:
            dois.append(line)

    # Create an async queue and populate it with prompts
    task_queue = asyncio.Queue()
    for message, doi in zip(prompts, dois):
        task_queue.put_nowait([system_msg, message, doi])

    # Create progress bar with total number of tasks
    total_tasks = len(prompts)
    pbar = tqdm(total=total_tasks, desc="Processing prompts", unit="prompt")
    
    # Create a list of worker tasks, one for each Ollama instance 
    tasks = []
    for host, gpu_index in gpus.items():
        tasks.append(worker(host, gpu_index, model, task_queue, output_dir, pbar))

    # Await the completion of all worker tasks
    await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Batch Processing Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to the configuration TOML file")
    parser.add_argument("--prompts", type=str, required=True, help="Path to the JSONL file with prompts")
    parser.add_argument("--dois", type=str, required=True, help="Path to the TXT file with dois")
    parser.add_argument("--output_dir", type=str, default="responses", help="Directory to save the response JSON files")

    args = parser.parse_args()

    try:
        asyncio.run(main(args.config, args.prompts, args.dois, args.output_dir))
        log_message("All prompts processed. Exiting...")
    except KeyboardInterrupt:
        log_message("Process interrupted by user. Exiting...")
