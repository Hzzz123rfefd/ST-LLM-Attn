from src.model import TimeSeriesBaseLLM
from src.dataset import DatasetForTimeSeries


datasets = {
    "time_series": DatasetForTimeSeries,
}

models = {
    "time_series": TimeSeriesBaseLLM,
}
