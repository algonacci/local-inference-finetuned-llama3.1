from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

prompt = input("Masukkan prompt:\n")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

inputs = tokenizer(
[
    alpaca_prompt.format(
        prompt,
        "",
        "",
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)