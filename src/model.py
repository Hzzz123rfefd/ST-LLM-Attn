import os
import torch
import torch.nn as nn
import numpy as np
from src.modules import STAtt, ResidualBlock
from src.base import ModelRegression,ModelLLMWithLoraBase, ModelTimeSeriesBase
import torch.nn.functional as F
from src.utils import *

class STLLMAttn(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, model_name_or_path="gpt2", device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.model_name_or_path = model_name_or_path
        self.station_num = station_num
        self.hidden_dim = hidden_dim
        
        self.inflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.outflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)

        self.llm = ModelLLMWithLoraBase(
            model_name_or_path =  model_name_or_path,
            lora_path = None, 
            device = "cuda"
        ).to(self.device)
        self.llm_hidden_dim = self.llm.hidden_dim
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)

    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature = self.inflow_feature_extractor(inflow)  
        outflow_feature = self.outflow_feature_extractor(outflow)  
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        llm_out = self.llm.backbone(inputs_embeds = feature, output_hidden_states=True)   # (b, t, hidden_dim)
        hidden_states = llm_out.hidden_states[-1]
        
        last_hidden = hidden_states[:, -1, :] 
        next_step_pred = self.predict_head(last_hidden)
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        inflow_predict = torch.clamp(inflow_predict, min = 0)
        outflow_predict = torch.clamp(outflow_predict, min = 0)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output  

    def load_pretrained(self, save_model_dir):
        # 加载大模型参数
        self.llm.load_pretrained(save_model_dir)
        # 加载其他参数
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"), strict=False)
        
    def save_pretrained(self,  save_model_dir):
        # 保存大模型参数
        self.llm.save_pretrained(save_model_dir)
        # 保存其他参数
        full_state_dict = self.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() if not k.startswith('llm.')
        }
        torch.save(filtered_state_dict, save_model_dir + "/model.pth")
        

class STLLMAttnLSTM(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, model_name_or_path="gpt2", device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.model_name_or_path = model_name_or_path
        self.inflow_feature_extractor = nn.LSTM(self.station_num, (int)(self.hidden_dim / 2), batch_first=True, bidirectional= False).to(self.device)
        self.outflow_feature_extractor = nn.LSTM(self.station_num, (int)(self.hidden_dim / 2), batch_first=True, bidirectional= False).to(self.device)
        self.llm = ModelLLMWithLoraBase(
            model_name_or_path =  model_name_or_path,
            lora_path = None, 
            device = "cuda"
        ).to(self.device)
        self.llm_hidden_dim = self.llm.hidden_dim
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)
        
    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature, _  = self.inflow_feature_extractor(inflow)  
        outflow_feature, _  = self.outflow_feature_extractor(outflow)  
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        llm_out = self.llm.backbone(inputs_embeds = feature, output_hidden_states=True)   # (b, t, hidden_dim)
        hidden_states = llm_out.hidden_states[-1]
        
        last_hidden = hidden_states[:, -1, :] 
        next_step_pred = self.predict_head(last_hidden)
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    

    def load_pretrained(self, save_model_dir):
        # 加载大模型参数
        self.llm.load_pretrained(save_model_dir)
        # 加载其他参数
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"), strict=False)
        
    def save_pretrained(self,  save_model_dir):
        # 保存大模型参数
        self.llm.save_pretrained(save_model_dir)
        # 保存其他参数
        full_state_dict = self.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() if not k.startswith('llm.')
        }
        torch.save(filtered_state_dict, save_model_dir + "/model.pth")     

class STLLMAttnNoAttn(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, model_name_or_path="gpt2", device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.model_name_or_path = model_name_or_path
        self.llm = ModelLLMWithLoraBase(
            model_name_or_path =  model_name_or_path,
            lora_path = None, 
            device = "cuda"
        ).to(self.device)
        self.llm_hidden_dim = self.llm.hidden_dim
        self.dim_transform = nn.Linear(self.station_num * 2, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)

    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)

        feature = torch.concat([inflow, outflow], dim = -1)
        feature = self.dim_transform(feature)
        llm_out = self.llm.backbone(inputs_embeds = feature, output_hidden_states=True)   # (b, t, hidden_dim)
        hidden_states = llm_out.hidden_states[-1]
        
        last_hidden = hidden_states[:, -1, :] 
        next_step_pred = self.predict_head(last_hidden)
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    

    def load_pretrained(self, save_model_dir):
        # 加载大模型参数
        self.llm.load_pretrained(save_model_dir)
        # 加载其他参数
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"), strict=False)
        
    def save_pretrained(self,  save_model_dir):
        # 保存大模型参数
        self.llm.save_pretrained(save_model_dir)
        # 保存其他参数
        full_state_dict = self.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() if not k.startswith('llm.')
        }
        torch.save(filtered_state_dict, save_model_dir + "/model.pth")     

class STLLMAttnNoLLM(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.inflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.outflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.llm_hidden_dim = 768
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)

    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature  = self.inflow_feature_extractor(inflow)  
        outflow_feature  = self.outflow_feature_extractor(outflow)  
        
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    

class STLLMAttnDecoder(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.inflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.outflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.decoder = nn.TransformerDecoderLayer(d_model = hidden_dim, nhead = 1, batch_first=True).to(self.device)
        self.predict_head = nn.Linear(self.hidden_dim, station_num * 2).to(self.device)

    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature  = self.inflow_feature_extractor(inflow)  
        outflow_feature  = self.outflow_feature_extractor(outflow)  
        
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.decoder(feature, feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    
    
class BPNN(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.llm_hidden_dim = 768
        self.inflow_feature_extractor = nn.Linear(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.outflow_feature_extractor = nn.Linear(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)
        
        self.relu = nn.ReLU()
    
    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature  = self.relu(self.inflow_feature_extractor(inflow)) 
        outflow_feature  = self.relu(self.outflow_feature_extractor(outflow)) 
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    
        
class CNN(ModelTimeSeriesBase):    
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.llm_hidden_dim = 768
        self.inflow_feature_extractor = nn.Conv1d(self.station_num, (int)(self.hidden_dim / 2), kernel_size=3, stride=1, padding=1).to(self.device)
        self.outflow_feature_extractor = nn.Conv1d(self.station_num, (int)(self.hidden_dim / 2), kernel_size=3, stride=1, padding=1).to(self.device)
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)
        
        self.relu = nn.ReLU()
    
    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature  = self.relu(self.inflow_feature_extractor(inflow.permute(0, 2, 1))).permute(0, 2, 1)
        outflow_feature  = self.relu(self.outflow_feature_extractor(outflow.permute(0, 2, 1))).permute(0, 2, 1)
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    

class GRU(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.llm_hidden_dim = 768
        self.inflow_feature_extractor = nn.GRU(self.station_num, (int)(self.hidden_dim / 2), 3, batch_first=True).to(self.device)
        self.outflow_feature_extractor = nn.GRU(self.station_num, (int)(self.hidden_dim / 2), 3, batch_first=True).to(self.device)
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)
        
        self.relu = nn.ReLU()
    
    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature, _  = self.inflow_feature_extractor(inflow)
        inflow_feature = self.relu(inflow_feature)
        outflow_feature, _  = self.outflow_feature_extractor(outflow)
        outflow_feature = self.relu(outflow_feature)
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    
    
class LSTM(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.llm_hidden_dim = 768
        self.inflow_feature_extractor = nn.LSTM(self.station_num, (int)(self.hidden_dim / 2), 3, batch_first=True).to(self.device)
        self.outflow_feature_extractor = nn.LSTM(self.station_num, (int)(self.hidden_dim / 2), 3, batch_first=True).to(self.device)
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)
        
        self.relu = nn.ReLU()
    
    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature, _  = self.inflow_feature_extractor(inflow)
        inflow_feature = self.relu(inflow_feature)
        outflow_feature, _  = self.outflow_feature_extractor(outflow)
        outflow_feature = self.relu(outflow_feature)
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output    
    
class STResNet(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.station_num = station_num
        self.hidden_dim = hidden_dim
        
        self.inflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)
        self.outflow_feature_extractor = STAtt(self.station_num, (int)(self.hidden_dim / 2)).to(self.device)

        self.llm_hidden_dim = 768
        self.dim_transform = nn.Linear(self.hidden_dim, self.llm_hidden_dim).to(self.device)
        self.cnn = nn.Conv1d(self.llm_hidden_dim, self.llm_hidden_dim, kernel_size=3, stride=1, padding=1).to(self.device)
        self.predict_head = nn.Linear(self.llm_hidden_dim, station_num * 2).to(self.device)

    def forward(self, input, is_train = True):
        """
        x: (b, t, d) -> (batch_size, time_steps, input_dim)
        """
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow_feature = self.inflow_feature_extractor(inflow)  
        outflow_feature = self.outflow_feature_extractor(outflow)  
        feature = torch.concat([inflow_feature, outflow_feature], dim = -1)
        feature = self.dim_transform(feature)
        feature2 = self.cnn(feature.permute(0, 2, 1)).permute(0, 2, 1)  
        feature = feature + feature2
        next_step_pred = self.predict_head(feature)
        next_step_pred = torch.mean(next_step_pred, dim=1) 
        inflow_predict, outflow_predict = next_step_pred.split(self.station_num, dim = -1)
        if is_train == True: 
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_predict,
                "outflow_predict":outflow_predict,
            }
        return output  
    
class Transformer(ModelTimeSeriesBase):
    def __init__(self, station_num, num_heads = 8, hidden_dim = 128, num_layers=2, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        
        self.embedding = nn.Linear(self.station_num * 2, self.hidden_dim).to(self.device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads, batch_first=True).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers).to(self.device)
        
        self.fc_inflow = nn.Linear(hidden_dim, self.station_num).to(self.device)
        self.fc_outflow = nn.Linear(hidden_dim, self.station_num).to(self.device)
        
    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        x = torch.cat([inflow, outflow], dim= -1) 
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        
        inflow_output = self.fc_inflow(x.mean(dim=1)) 
        outflow_output = self.fc_outflow(x.mean(dim=1))  
        
        if is_train == True: 
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
            }
        return output    

class WaveNet(ModelTimeSeriesBase):
    def __init__(self, station_num,  res_channels=32, hidden_dim = 128, num_blocks=2, num_layers=4, kernel_size=2, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.input_conv = nn.Conv1d(2, res_channels, 1)

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                self.residual_blocks.append(ResidualBlock(res_channels, kernel_size, dilation))

        self.output_conv1 = nn.Conv1d(res_channels, res_channels, 1)
        self.output_conv2 = nn.Conv1d(res_channels, 2, 1)

    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow = inflow.permute(0, 2, 1)
        outflow = outflow.permute(0, 2, 1)
        x = torch.stack([inflow, outflow], dim=2)
        batch_size, n_stations, _, time_steps = x.shape
        x = x.reshape(batch_size * n_stations, 2, time_steps)
        x = self.input_conv(x)
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        out = sum(skip_connections)
        out = F.relu(out)
        out = self.output_conv1(out)
        out = F.relu(out)
        out = self.output_conv2(out)
        out = out.reshape(batch_size, n_stations, 2, time_steps)
        inflow_output = out[:, :, 0, -1]
        outflow_output = out[:, :, 1, -1]
        if is_train == True: 
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
            }
        return output    
    
class TCN(ModelTimeSeriesBase):
    def __init__(self, station_num, hidden_dim, num_layers = 2 , kernel_size=3, dropout=0.2, device="cuda"):
        super().__init__(station_num, hidden_dim, device)
        self.encoder_in = nn.Sequential(
            nn.Conv1d(station_num, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.encoder_out = nn.Sequential(
            nn.Conv1d(station_num, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.projection_in = nn.Linear(hidden_dim, station_num)
        self.projection_out = nn.Linear(hidden_dim, station_num)

    def forward(self, input, is_train = True):
        inflow = input["inflow"].to(self.device)
        outflow = input["outflow"].to(self.device)
        inflow = inflow.permute(0, 2, 1)
        outflow = outflow.permute(0, 2, 1)

        in_encoded = self.encoder_in(inflow)  # [batch_size, hidden_dim, seq_len]
        

        out_encoded = self.encoder_out(outflow)  # [batch_size, hidden_dim, seq_len]

        final_in_state = in_encoded[:, :, -1]  # [batch_size, hidden_dim]
        final_out_state = out_encoded[:, :, -1]  # [batch_size, hidden_dim]

        inflow_output = self.projection_in(final_in_state)  # [batch_size, output_dim]
        outflow_output = self.projection_out(final_out_state)  # [batch_size, output_dim]
        
        if is_train == True: 
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
                "inflow_label": input["inflow_label"].to(self.device),
                "outflow_label": input["outflow_label"].to(self.device)
            }
        else:
            output = {
                "inflow_predict": inflow_output,
                "outflow_predict":outflow_output,
            }
        return output    