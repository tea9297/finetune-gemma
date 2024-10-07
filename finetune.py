import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import transformers

from trl import SFTTrainer

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)

model_id = "model/gemma-2-27b"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


def get_completion(query: str, model, tokenizer) -> str:
    device = "auto"

    prompt_template = """
    <start_of_turn>user
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    {query}
    <end_of_turn>\n<start_of_turn>model


    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs,
                                   max_new_tokens=1000,
                                   do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    # decoded = tokenizer.batch_decode(generated_ids)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return (decoded)


def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    prefix_text = ""
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text


# Load the dataset from a JSON file
dataset = load_dataset('json', data_files='./ft/transformed_trainingset.json')

# Split the dataset into training and testing sets
train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=1234)

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

print(dataset_dict)

# Convert to pandas DataFrame if needed
df_train = dataset_dict['train'].to_pandas()
df_test = dataset_dict['test'].to_pandas()

# Add the "prompt" column in the dataset
text_column_train = [
    generate_prompt(data_point) for data_point in dataset_dict['train']
]
text_column_test = [
    generate_prompt(data_point) for data_point in dataset_dict['test']
]

dataset_dict['train'] = dataset_dict['train'].add_column(
    "prompt", text_column_train)
dataset_dict['test'] = dataset_dict['test'].add_column("prompt",
                                                       text_column_test)

# Shuffle the dataset
dataset_dict['train'] = dataset_dict['train'].shuffle(seed=1234)
dataset_dict['test'] = dataset_dict['test'].shuffle(seed=1234)

# Tokenize the dataset
dataset_dict = dataset_dict.map(lambda samples: tokenizer(samples["prompt"]),
                                batched=True)

train_data = dataset_dict['train']
test_data = dataset_dict['test']

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(model)
print(modules)

lora_config = LoraConfig(r=64,
                         lora_alpha=32,
                         target_modules=modules,
                         lora_dropout=0.05,
                         bias="none",
                         task_type="CAUSAL_LM")

model = get_peft_model(model, lora_config)

#### start finetuning ####
#
#
#

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        #warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                               mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

new_model = "model/gemma2-27b-panaSQL-Instruct-Finetune"  #Name of the model you will be pushing to huggingface model hub

trainer.model.save_pretrained(new_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

merged_model = PeftModel.from_pretrained(base_model, new_model)
merged_model = merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(
    "model/gemma2-27b-panaSQL-Instruct-Finetune-merged_model",
    safe_serialization=True)
tokenizer.save_pretrained(
    "model/gemma2-27b-panaSQL-Instruct-Finetune-merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
