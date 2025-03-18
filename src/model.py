import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from src.modules import TimeProcess
from src.base import ModelRegression,ModelLLMWithLoraBase
        
class TimeSeriesBaseLLM(ModelRegression):
    def __init__(self, input_dim, hidden_dim, llm_model="gpt2", device = "cuda"):
        super().__init__(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_feature_extractor = TimeProcess(self.input_dim, self.hidden_dim)

        self.config = AutoConfig.from_pretrained(llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm_hidden_dim = self.config.hidden_size
        
        for param in self.llm.parameters():
            param.requires_grad = False 
        
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim)
        self.predict_head = nn.Linear(self.llm_hidden_dim, input_dim)
    
    def forward(self, input):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        x = input["seq"].to(self.device)
        feature = self.time_feature_extractor(x)  
        feature = self.dim_transform(feature)
        llm_out = self.llm(inputs_embeds = feature, output_hidden_states=True)   # (b, t, hidden_dim)
        hidden_states = llm_out.hidden_states[-1]
        
        last_hidden = hidden_states[:, -1, :] 
        next_step_pred = self.predict_head(last_hidden)
        
        output = {
            "predict": next_step_pred,
            "label": input["predict"].to(self.device)
        }
        return output  
    
    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")    


class STLLMAttn(ModelRegression):
    def __init__(self, input_dim, hidden_dim, model_name_or_path="gpt2", device="cuda"):
        # config = AutoConfig.from_pretrained(model_name_or_path)
        ModelRegression.__init__(self, device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_feature_extractor = TimeProcess(self.input_dim, self.hidden_dim)

        self.llm = ModelLLMWithLoraBase(
            model_name_or_path =  model_name_or_path,
            lora_path = None, 
            device = "cuda"
        )
        
        self.llm_hidden_dim = self.llm.hidden_dim
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim)
        self.predict_head = nn.Linear(self.llm_hidden_dim, input_dim)

    def forward(self, input):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        x = input["seq"].to(self.device)
        feature = self.time_feature_extractor(x)  
        feature = self.dim_transform(feature)
        llm_out = self.llm.backbone(inputs_embeds = feature, output_hidden_states=True)   # (b, t, hidden_dim)
        hidden_states = llm_out.hidden_states[-1]
        
        last_hidden = hidden_states[:, -1, :] 
        next_step_pred = self.predict_head(last_hidden)
        
        output = {
            "predict": next_step_pred,
            "label": input["predict"].to(self.device)
        }
        return output  

    def load_pretrained(self, save_model_dir):
        self.llm.backbone = self.llm.load_model_with_lora(save_model_dir)
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        self.llm.backbone.save_pretrained(save_model_dir)
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
# # 测试模型
# batch_size, time_steps, input_dim, hidden_dim = 4, 10, 16, 64
# model = TimeSeriesGPT2(input_dim, hidden_dim)
# x = torch.randn(batch_size, time_steps, input_dim)  # 随机输入
# output = model(x)
# print("Output shape:", output.shape)  # (b, 1, d)
