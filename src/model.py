import torch
import torch.nn as nn
import numpy as np
from src.modules import STAtt
from src.base import ModelRegression,ModelLLMWithLoraBase
import torch.nn.functional as F
from src.utils import *

class STLLMAttn(ModelRegression):
    def __init__(self, station_num, hidden_dim, model_name_or_path="gpt2", device="cuda"):
        ModelRegression.__init__(self, device)
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
        
    def inference(self, inflow, outflow):
        """
            t: shepe (t, dim)
        """
        inflow = np.array(inflow, dtype=np.float32)  
        inflow = torch.from_numpy(inflow).unsqueeze(0).to(self.device)
        outflow = np.array(outflow, dtype=np.float32)  
        outflow = torch.from_numpy(outflow).unsqueeze(0).to(self.device)
        input = {
            "inflow" :  inflow,
            "outflow" : outflow
        }
        with torch.no_grad():
            output = self.forward(input, is_train = False)
            inflow_predict = output["inflow_predict"]
            outflow_predict = output["outflow_predict"]
        inflow_predict = inflow_predict.squeeze(0).cpu().numpy()
        outflow_predict = outflow_predict.squeeze(0).cpu().numpy()
        return inflow_predict, outflow_predict

    def compute_loss(self, input):
        inflow_loss = F.mse_loss(input["inflow_predict"], input["inflow_label"])
        outflow_loss = F.mse_loss(input["outflow_predict"], input["outflow_label"])
        total_loss = inflow_loss + outflow_loss
        output = {
            "inflow_loss":  inflow_loss,
            "outflow_loss": outflow_loss,
            "total_loss":total_loss
        }
        return output

    def eval_epoch(self, epoch, val_dataloader, log_path = None):
        inflow_max_value = 5770
        inflow_min_value = 0
        outflow_max_value = 5770
        outflow_min_value = 0
        rmse_in = AverageMeter()
        mae_in = AverageMeter()
        wmape_in =  AverageMeter()
        rmse_out = AverageMeter()
        mae_out = AverageMeter()
        wmape_out =  AverageMeter()
        with torch.no_grad():
            for batch_id,inputs in enumerate(val_dataloader):
                output = self.forward(inputs, is_train = True)
                inflow_predict = output["inflow_predict"] * (inflow_max_value - inflow_min_value) + inflow_min_value
                inflow_label = output["inflow_label"] * (inflow_max_value - inflow_min_value) + inflow_min_value
                outflow_predict = output["outflow_predict"] * (outflow_max_value - outflow_min_value) + outflow_min_value
                outflow_label = output["outflow_label"] * (outflow_max_value - outflow_min_value) + outflow_min_value
                rmse_in.update(rmse(inflow_predict, inflow_label).cpu().item())
                mae_in.update(mae(inflow_predict, inflow_label).cpu().item())
                wmape_in.update(wmape(inflow_predict, inflow_label).cpu().item())
                rmse_out.update(rmse(outflow_predict, outflow_label).cpu().item())
                mae_out.update(mae(outflow_predict, outflow_label).cpu().item())
                wmape_out.update(wmape(outflow_predict, outflow_label).cpu().item())
        str =  f"inflow RMSE = {rmse_in.avg}, MAE = {mae_in.avg}, WMAPE = {wmape_in.avg}, outflow RMSE = {rmse_out.avg}, MAE = {mae_out.avg}, WMAPE = {wmape_out.avg}"
        print(str)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(str + "\n")
                
        
        
        

