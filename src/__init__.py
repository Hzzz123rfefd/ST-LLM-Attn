from src.model import STLLMAttn
from src.dataset import DatasetForTimeSeries


datasets = {
    "time_series": DatasetForTimeSeries,
}

models = {
    "st_llm_attn":STLLMAttn,
}
