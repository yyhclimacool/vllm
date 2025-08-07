import torch
from vllm import LLM, SamplingParams


compute_capability = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
print(
    f"GPU: {gpu_name}, Compute Capability: {compute_capability[0]}.{compute_capability[1]}"
)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# sampling_params = SamplingParams(temperature=0, top_p=1, use_beam_search=True, n=2)

# Create an LLM.
# # OPT 系列（您之前用过的）
# llm = LLM(model="facebook/opt-125m")       # 125M 参数
# llm = LLM(model="facebook/opt-350m")       # 350M 参数
# llm = LLM(model="facebook/opt-1.3b")       # 1.3B 参数
# llm = LLM(model="facebook/opt-2.7b")       # 2.7B 参数
# openlm-research/open_llama_3b不支持
llm = LLM(model="models/lmsys/vicuna-7b-v1.3", worker_use_ray=False)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# print("===========================")
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
