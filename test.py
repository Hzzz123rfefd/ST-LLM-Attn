import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoConfig
# 加载 GPT-2 模型和分词器
model_name = "gpt2"
model  =  AutoModelForCausalLM.from_pretrained(
            model_name
        )

config = AutoConfig.from_pretrained(model_name)


model.save_pretrained("test")

print(config.hidden_size)

input_ids = torch.rand(2, 6, 4096)
# 运行前向过程
outputs = model(inputs_embeds=input_ids)

# 获取 logits（下一步预测的概率分布）
logits = outputs.logits  # 形状: [batch_size, seq_length, vocab_size]

# 获取隐藏层表示
hidden_states = outputs.hidden_states  # 每一层 Transformer 的输出

# 获取注意力权重
attentions = outputs.attentions  # 每一层注意力机制的输出

print("Logits shape:", logits.shape)  # [1, seq_length, vocab_size]
print("Hidden states shape:", len(hidden_states), hidden_states[0].shape)  # (num_layers+1, batch_size, seq_length, hidden_dim)
print("Attention shape:", len(attentions), attentions[0].shape)  # (num_layers, batch_size, num_heads, seq_length, seq_length)
