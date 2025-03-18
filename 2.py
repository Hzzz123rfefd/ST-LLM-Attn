from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PeftModel
import torch

# 加载大模型基座
base_model =  AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16
)

# 加载lora配置
peft_config =  LoraConfig.from_pretrained("lora")

# 加载已经训练过的lora的大模型
backbone = PeftModel.from_pretrained(
    model = base_model, 
    model_id = "lora",
    is_trainable = True
)

# 打印可训练参数
backbone.print_trainable_parameters()

# 如果需要继续训练，则继续保存LORA配置即可
backbone.save_pretrained("lora")