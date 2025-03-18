from src.model import TimeSeriesBaseLLM,STLLMAttn
from src.dataset import DatasetForTimeSeries


datasets = {
    "time_series": DatasetForTimeSeries,
}

models = {
    "time_series": TimeSeriesBaseLLM,
    "st_llm_attn":STLLMAttn
}
