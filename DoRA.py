# ------------------INSTALLATION------------------

# Install Hugging Face Hub and log in to your account
!pip install -q -U huggingface_hub
from huggingface_hub import notebook_login
notebook_login()

# Install Weights and Biases (WandB) for experiment tracking and log in to your account
!pip install wandb -q -U
import wandb
wandb.login()

# Install stable versions of essential libraries
!python3 -m pip install --upgrade pip

# Install specific versions of the necessary libraries
!pip install -U transformers==4.38.1 -q         # Hugging Face Transformers
!pip install -U bitsandbytes==0.42.0 -q         # 8-bit optimizers for model efficiency
!pip install -U peft==0.8.2 -q                  # Parameter-Efficient Fine-Tuning (PEFT)
!pip install -U accelerate==0.27.2 -q           # Accelerate library for distributed training
!pip install datasets==2.17.1 -q                # Hugging Face Datasets library
!pip install -U scipy==1.12.0 -q                # SciPy for scientific computing
!pip install -U trl==0.7.11 -q                  # TRL library for training transformers with reinforcement learning
!pip install -U flash-attn==2.5.5 -q            # Flash Attention for efficient attention computation
!pip install hf_transfer==0.1.5                 # HF Transfer for transferring data/models between environments

# Install SentencePiece for tokenization (required for certain models like Yi)
# Uncomment the following line if needed
# !pip install sentencepiece -q -U

# Display the current environment configuration related to the Transformers library
!transformers-cli env

# Set environment variable to enable HF Transfer in Hugging Face Hub
%env HF_HUB_ENABLE_HF_TRANSFER=True

#If using DoRA
!pip uninstall peft -y
!pip install git+https://github.com/BenjaminBossan/peft.git@feat-dora -q
# ------------------MODEL LOADING------------------

# Define the model ID for loading the Mistral-7B model
model_id = "mistralai/Mistral-7B-v0.1"

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import tempfile

# Set up the cache directory
# Use a temporary directory for caching (replace with a specific path if needed)
cache_dir = tempfile.mkdtemp()
# Example of setting a custom cache directory
# cache_dir = "/path/to/your/cache/directory"  # Replace with your desired path

# Configuration for 4-bit quantization with BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.bfloat16
)

# Load the model with the specified configuration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,         # Uncomment if using 4-bit quantization
    # rope_scaling={"type": "linear", "factor": 2.0},  # Uncomment if using rope scaling
    device_map="auto",                        # Automatically map model layers to available devices (e.g., GPUs)
    # trust_remote_code=False,                # Uncomment if you do not want to trust remote code
    torch_dtype=torch.bfloat16,               # Use bfloat16 for reduced precision
    attn_implementation="flash_attention_2",  # Use Flash Attention for efficient attention computation
    cache_dir=cache_dir                       # Specify the cache directory
)

# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    use_fast=True,                            # Use the fast tokenizer implementation
    trust_remote_code=True                    # Allow custom code to be loaded from the model repository
)

# ------------------LOADING CHECKS------------------

# Check that no parameters are loaded onto CPU (meta device)
for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on the meta device, not properly loaded")

# Display some configuration details of the model
print(f"Max Position Embedding: {model.config.max_position_embeddings}")
print(f"EOS Token ID: {model.config.eos_token_id}")

# ------------------PREPARE FOR FINE-TUNING------------------
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0
    
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()

    print("Non-trainable parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name}")
    print(
        f"Summary:\n    Trainable params: {trainable_params}\n  Non Trainable params: {non_trainable_params}"
    )

#Standard LoRA or DoRA

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable() # Comment this in to save on VRAM
#model = prepare_model_for_kbit_training(model)
print(model)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"self_attn.rottary_emb.inv_freq",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lora_magnitude_vector",
        #"input_layernorm.weight",
        #"post_attention_layernorm.weight",
        #"model.norm.weight",
        #"lm_head.weight",
            #"dense_h_to_4h",
            #"dense_4h_to_h",
            #"query_key_value",
            #"dense"
    ],
    lora_dropout=0.1,
    bias=None,
    task_type="CAUSAL_LM",
    use_dora = True #only for DoRA 
)

model = get_peft_model(model, peft_config)
#print_trainable_parameters(model)

#Set up Tokenizer and Padding
print(tokenizer)
print(tokenizer.vocab_size)

print(tokenizer.bos_token)
print(tokenizer.eos_token)

## OPTIONALLY SET THE CHAT TEMPLATE MANUALLY
tokenizer.chat_template = """{% if not add_generation_prompt_is_defined %}
### Instruction:
{{content}}
{% endif %}"""

# Test the chat template
messages = [
    {'role': 'user', 'content': "write a quick sort algorithm in python."},
    {'role': 'assistant', 'content': "here you are."},
    {'role': 'user', 'content': "great."}
]

inputs = tokenizer.apply_chat_template(messages, tokenize=False)
print(inputs)

# Option A: Use an existing token as the pad token
if '<pad>' in tokenizer.get_vocab():
    print("<pad> token is in the tokenizer. Using <pad> for padding.")
    tokenizer.pad_token = "<pad>"
elif '' in tokenizer.get_vocab():
    print("Empty string token is in the tokenizer. Using an empty string for padding.")
    tokenizer.pad_token = ""
elif '<unk>' in tokenizer.get_vocab():
    print("<unk> token is in the tokenizer. Using <unk> for padding.")
    tokenizer.pad_token = "<unk>"
else:
    print(f"Using EOS token, {tokenizer.eos_token}, for padding.")
    tokenizer.pad_token = tokenizer.eos_token

# Option B: Create and add a pad token if it does not exist (Commented out)
# Check if the <pad> token is already in the tokenizer vocabulary
# if '<pad>' not in tokenizer.get_vocab():
#     print("<pad> token not in the tokenizer, adding a <pad> token.")

#     # Add the <pad> token to the tokenizer
#     tokenizer.add_tokens(['<pad>'])

#     # Set the newly added <pad> token as the pad token
#     tokenizer.pad_token = "<pad>"

#     # Resize the model's token embeddings to accommodate the new token
#     model.resize_token_embeddings(len(tokenizer))
# Update pad token id in model and its config

model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Check if they are equal
assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID."

# Print the pad token ids
print('Tokenizer pad token ID:', tokenizer.pad_token_id)
print('Model pad token ID:', model.pad_token_id)
print('Model config pad token ID:', model.config.pad_token_id)
print('Number of tokens now in tokenizer:', tokenizer.vocab_size)

print("Special Token map:", tokenizer.special_tokens_map)
#print("All special tokens:", tokenizer.all_special_tokens)

# Uncomment to swtich to left padding, not recommended for unsloth
# tokenizer.padding_size = "left"
#print(tokenizer)

# Set embed and norm layers to trainable (recommended for chat-fine tuning)
trainable_param_names = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "final_layer_norm"]

# Loop through all model parameters and set 'requires_grad' to True only for trainable parameters
for name, param in model.named_parameters():
    if any(trainable_name in name for trainable_name in trainable_param_names):
        param.requires_grad = True
        print(f"Parameter {name} is trainable.")
    else:
        param.requires_grad = False

#Make dictionary of trainable parameters
trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
#Convert trainable_params to state_dict
trainable_params_state_dict = {n:p.data for n,p in trainable_params.items()}

#Set up evaluation
from transformers import TextStreamer  # Corrected 'tansformers' to 'transformers'
from peft import PeftModel
import torch
import gc

def stream(user_prompt, model_type, tokenizer, model, checkpoint=''):
    """
    Function to generate and stream model outputs based on user input.

    Parameters:
    user_prompt (str): The prompt from the user.
    model_type (str): Type of the model to use ("base" or "fine_tuned").
    tokenizer: Tokenizer to be used for preparing inputs.
    model: The base model used for evaluation.
    checkpoint (str): Checkpoint path for the fine-tuned model (if applicable).
    """

    # Load and prepare the model based on its type
    if model_type == "base":
        eval_model = model
    elif model_type == "fine_tuned":
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")
        
        # Check if any parameters remain on the CPU
        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on CPU!")

    else:
        print("You must set the model_type to 'base' or 'fine_tuned'")
        exit()
    
    # Ensure the model uses cached computations
    eval_model.config.use_cache = True

    # Prepare the chat input messages
    messages = [
        {"role": "user", "content": f"{user_prompt.strip()}"}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Remove token_type_ids if present
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Initialize the streamer for output
    streamer = TextStreamer(tokenizer)
    
    # Generate text based on the inputs and stream the output
    _ = eval_model.generate(**inputs, streamer=streamer, max_new_tokens=100)
    
    # Clear CUDA memory cache and run garbage collection to free up memory
    torch.cuda.empty_cache()
    gc.collect()

def evaluation(model_type, tokenizer, model, checkpoint=''):
    """
    Function to evaluate the model on a set of questions.

    Parameters:
    model_type (str): Type of the model to use ("base" or "fine_tuned").
    tokenizer: Tokenizer to be used for preparing inputs.
    model: The base model used for evaluation.
    checkpoint (str): Checkpoint path for the fine-tuned model (if applicable).
    """

    # List of questions to ask the model
    questions = [
        "What planets are in our solar system?",
        "What are the first 5 numbers in the Fibonacci series?",
        "Generate a Python code snippet to add two numbers."
    ]
    
    # Placeholder for correct answers
    answers = [
        "",  # Add the correct answer here if desired
        "",  # Add the correct answer here if desired
        ""   # Add the correct answer here if desired
    ]

    # Stream model's answers to each question
    for question, answer in zip(questions, answers):
        stream(question, model_type, tokenizer, model, checkpoint)
        # Uncomment to print the correct answer for comparison
        # print("Correct answer:", answer)
        print("\n\n")


# Assuming model and tokenizer are predefined
print(model.generation_config)
evaluation("base", tokenizer, model)


#Load the Dataset
from datasets import load_dataset

dataset = "Trelis/openassistant-llama-style"
data = load_dataset(dataset)

print("First row of train:", data["train"][1])

text = data["train"][0]["text"]
tokens = tokenizer.encode(text, add_special_tokens = True)
decoded_text = tokenizer.decode(tokens)
print("Token IDS:", tokens)
print("Decoded Text:", decoded_text)

#TRAIN
#Using TLR trainer
#TRL trainer
model_name = model_id.split("/")[-1]
dataset_name = dataset.split("/")[-1]

epochs = 1
context_length = 512*8
grad_accum = 8
batch_size = 4
fine_tune_tag = "chat-fine-tuned-model"
save_dir = f"./results/{model_name}_{dataset_name}_{epochs}_epochs_{context_length}_context_length_fine_tuned"
print(save_dir)

import transformers
import os

import os
import transformers

class LoggingCallback(transformers.TrainerCallback):
    def __init__(self, log_file_path, save_dir):
        self.log_file_path = log_file_path
        self.save_dir = save_dir
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # Open the log file and append the current logs
        with open(self.log_file_path, "a") as f:
            if "loss" in logs:
                f.write(f"Step: {state.global_step}, Training Loss: {logs['loss']}\n")

            if "eval_loss" in logs:
                f.write(f"Step: {state.global_step}, Eval Loss: {logs['eval_loss']}\n")
            
            f.flush()  # Ensure data is written to the file immediately
    
        # Check if the current step is a checkpoint step
        if state.global_step % int(args.save_steps) == 0:
            # Check if the best model checkpoint path exists
            if state.best_model_checkpoint:
                checkpoint_dir = state.best_model_checkpoint
            else:
                # Construct the checkpoint directory path manually
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            # Ensure the checkpoint directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save trainable parameters in the checkpoint directory
            current_trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            current_trainable_params_state_dict = {n: p.data for n, p in current_trainable_params.items()}

            # Define the file path for saving the trainable parameters
            file_path = os.path.join(checkpoint_dir, "trainable_params.bin")

            # Save the trainable parameters state dict
            torch.save(current_trainable_params_state_dict, file_path)

log_file_path = os.path.join(cache_dir, "training_logs.txt")
logging_callback = LoggingCallback(log_file_path)

from transformers import Trainer
from trl import SFTTrainer

# Initialize the SFTTrainer
trainer = SFTTrainer(
    dataset_text_field="text",  # Corrected typo: `dataset_text_filed` to `dataset_text_field`
    max_seq_length=context_length,  # Set the maximum sequence length for model inputs
    tokenizer=tokenizer,  # Tokenizer to convert text into model inputs
    model=model,  # Model to be fine-tuned
    train_dataset=data["train"],  # Training dataset
    eval_dataset=data["test"],  # Evaluation dataset
    args=TrainingArguments(  # Training arguments for controlling the training process
        max_steps=20,  # Maximum number of steps to train
        save_steps=20,  # Save checkpoints every 20 steps
        logging_steps=1,  # Log training metrics every step
        num_train_epochs=epochs,  # Number of training epochs
        output_dir=save_dir,  # Directory to save model checkpoints and outputs
        evaluation_strategy="steps",  # Evaluate the model every few steps
        do_eval=True,  # Perform evaluation during training
        eval_steps=1,  # Corrected from `eval_step` to `eval_steps`, should be an integer (e.g., every 1 or 2 steps)
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        per_device_train_batch_size=batch_size,  # Batch size for training
        gradient_accumulation_steps=grad_accum,  # Corrected typo: `accumilation` to `accumulation`
        log_level="debug",  # Log level for detailed output during training
        # optim="paged_adamw_8bit",  # Uncomment for 8-bit optimizers, useful for large models
        # fp16=True,  # Uncomment if using FP16 precision for training on non-Ampere GPUs
        bf16=True,  # Use bfloat16 precision for Ampere GPUs
        max_grad_norm=0.3,  # Maximum gradient norm for gradient clipping
        lr_scheduler_type="constant",  # Corrected typo: `lr_schedulre_type` to `lr_scheduler_type`
        hub_private_repo=True,  # Push model to a private Hugging Face Hub repository
        # warmup_ratio=0.03,  # Uncomment to use a warmup ratio for learning rate scheduling
        optim="adamw_torch",  # Use AdamW optimizer; comment out if using LoRA+
        learning_rate=1e-4,  # Initial learning rate; comment out if using LoRA+
    ),
    callbacks=[logging_callback],  # Callback for logging during training
    # optimizers=(optimizer, scheduler)  # Uncomment if using custom optimizers and schedulers, like for LoRA+
)

# Disable caching of intermediate states in the model, which is useful when fine-tuning
model.config.use_cache = False

# Start the training process
trainer.train()
