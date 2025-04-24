import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from torch import optim
from abc import abstractmethod
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch

from src.utils import *

class ModelBase(nn.Module):
    def __init__(
        self,
        device = "cuda"
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
    
    def trainning(
            self,
            train_dataloader: DataLoader = None,
            test_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 0,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            eval_interval:int = 10,
            save_model_dir:str = None
        ):
            ## 1 trainning log path 
            first_trainning = True
            check_point_path = save_model_dir  + "/checkpoint.pth"
            log_path = save_model_dir + "/train.log"

            ## 2 get net pretrain parameters if need 
            if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
                self.load_pretrained(save_model_dir)  
                first_trainning = False

            else:
                if not os.path.isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                with open(log_path, "w") as file:
                    pass

            ##  3 get optimizer
            if optimizer_name == "Adam":
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            elif optimizer_name == "AdamW":
                optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, 
                mode = "min", 
                factor = factor, 
                patience = patience
            )

            ## 4 init trainng log
            if first_trainning:
                best_loss = float("inf")
                last_epoch = 0
            else:
                checkpoint = torch.load(check_point_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_loss = checkpoint["loss"]
                last_epoch = checkpoint["epoch"] + 1

            try:
                for epoch in range(last_epoch, total_epoch):
                    log_str = f"Learning rate: {optimizer.param_groups[0]['lr']}"
                    with open(log_path, "a") as file:
                        file.write(log_str+"\n")
                    train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = train_loss + test_loss
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": None,
                            "lr_scheduler": None
                        },
                        check_point_path
                    )
                    if epoch % eval_interval == 0:
                        self.eval_epoch(epoch, val_dataloader, log_path)
                    
                    if is_best:
                        self.save_pretrained(save_model_dir)
                    
                    # if(optimizer.param_groups[0]['lr'] == 9e-06):
                    #     break

            # interrupt trianning
            except KeyboardInterrupt:
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        check_point_path
                    )
    
    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train().to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "Train Epoch: {:d}, total_loss: {:.4f}, use_memory: {:.1f}G".format(
                epoch,
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
            
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")

        return total_loss.avg

    def test_epoch(self, epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())

            str = "Test Epoch: {:d}, total_loss: {:.4f}".format(
                epoch,
                total_loss.avg, 
            )
        print(str)
        with open(log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg
    
    @abstractmethod
    def eval_epoch(self, epoch, val_dataloader, log_path = None):
        pass
        
    @abstractmethod    
    def compute_loss(self, input):
        pass
    
    @abstractmethod
    def load_pretrained(self, save_model_dir):
        pass

    @abstractmethod
    def save_pretrained(self, save_model_dir):
        pass

class ModelRegression(ModelBase):
    def compute_loss(self, input):
        output = {
            "total_loss": F.mse_loss(input["predict"], input["label"])
        }
        return output
    
class ModelLLMWithLoraBase(ModelBase):
    def __init__(
        self,
        model_name_or_path,
        lora_path = None, 
        device = "cuda"
    ):
        super().__init__(device)
        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path
        # load llm model
        self.base_model =  AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32
        )
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.hidden_dim = self.config.hidden_size
        self.backbone = self.load_model_with_lora(self.lora_path)
        self.backbone.print_trainable_parameters()
        
    def load_model_with_lora(self, lora_path):
        if lora_path == None:
            self.lora_config = LoraConfig(
                task_type = TaskType.CAUSAL_LM, 
                inference_mode = False,
                r = 8,
                lora_alpha = 16,                                                                                  # LoRA 的缩放因子
                lora_dropout = 0.05 
            )
            return get_peft_model(self.base_model, self.lora_config)

        return PeftModel.from_pretrained(
            model = self.base_model, 
            model_id = lora_path,
            is_trainable = True
        )
    
    def load_pretrained(self, save_model_dir):
        if save_model_dir != None:
            self.load_model_with_lora(save_model_dir)

    def save_pretrained(self, save_model_dir):
        self.backbone.save_pretrained(save_model_dir)

class ModelTimeSeriesBase(ModelRegression):
    def __init__(self, station_num, hidden_dim, device="cuda"):
        super().__init__(device)
        self.station_num = station_num
        self.hidden_dim = hidden_dim

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
        inflow_max_value = 5770      # 5770  3246.0
        inflow_min_value = 0
        outflow_max_value = 5945    # 5945  4365
        outflow_min_value = 0
        rmse_in = AverageMeter()
        mae_in = AverageMeter()
        wmape_in =  AverageMeter()
        rmse_out = AverageMeter()
        mae_out = AverageMeter()
        wmape_out =  AverageMeter()
        inflow_predicts = []
        inflow_labels = []
        outflow_predicts = []
        outflow_labels = []
        with torch.no_grad():
            for batch_id,inputs in enumerate(val_dataloader):
                output = self.forward(inputs, is_train = True)
                inflow_predict = output["inflow_predict"] * (inflow_max_value - inflow_min_value) + inflow_min_value
                inflow_label = output["inflow_label"] * (inflow_max_value - inflow_min_value) + inflow_min_value
                outflow_predict = output["outflow_predict"] * (outflow_max_value - outflow_min_value) + outflow_min_value
                outflow_label = output["outflow_label"] * (outflow_max_value - outflow_min_value) + outflow_min_value
                inflow_predicts.append(inflow_predict.cpu())
                inflow_labels.append(inflow_label.cpu())
                outflow_predicts.append(outflow_predict.cpu())
                outflow_labels.append(outflow_label.cpu())
                # rmse_in.update(rmse(inflow_predict, inflow_label).cpu().item())
                # mae_in.update(mae(inflow_predict, inflow_label).cpu().item())
                # wmape_in.update(wmape(inflow_predict, inflow_label).cpu().item())
                # rmse_out.update(rmse(outflow_predict, outflow_label).cpu().item())
                # mae_out.update(mae(outflow_predict, outflow_label).cpu().item())
                # wmape_out.update(wmape(outflow_predict, outflow_label).cpu().item())
        inflow_predicts = torch.cat(inflow_predicts, dim=0)
        inflow_labels = torch.cat(inflow_labels, dim=0)
        outflow_predicts = torch.cat(outflow_predicts, dim=0)
        outflow_labels = torch.cat(outflow_labels, dim=0)
        str =  f"inflow RMSE = {rmse(inflow_predicts, inflow_labels).cpu().item()}, MAE = {mae(inflow_predicts, inflow_labels).cpu().item()}, WMAPE = {wmape(inflow_predicts, inflow_labels).cpu().item()}, outflow RMSE = {rmse(outflow_predicts, outflow_labels).cpu().item()}, MAE = {mae(outflow_predicts, outflow_labels).cpu().item()}, WMAPE = {wmape(outflow_predicts, outflow_labels).cpu().item()}"
        print(str)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(str + "\n")
        inflow_predicts = inflow_predicts.numpy()
        inflow_labels = inflow_labels.numpy()
        outflow_predicts = outflow_predicts.numpy()
        outflow_labels = outflow_labels.numpy()
        return inflow_predicts, inflow_labels, outflow_predicts, outflow_labels
                
    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"), strict=False)
        
    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")     

        
 