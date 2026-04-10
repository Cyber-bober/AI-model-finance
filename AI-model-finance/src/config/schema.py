# Pydantic-валидация всех параметров
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import date

class DataFilters(BaseModel):
    start_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    instruments: List[str] = Field(min_length=1)
    min_avg_volume: int = 0
    frequency: str = Field(default="1D", pattern="^(1H|4H|1D|1W)$")

class ModelParams(BaseModel):
    type: str = Field(default="lightgbm", pattern="^(lightgbm|xgboost|catboost)$")
    max_depth: int = Field(ge=1, le=16)
    num_leaves: int = Field(ge=2, le=127)
    learning_rate: float = Field(gt=0.0, le=1.0)
    n_estimators: int = Field(ge=50, le=2000)
    min_child_samples: int = Field(ge=1)

class TrainingConfig(BaseModel):
    lookback: int = Field(ge=10, le=365)
    horizon: int = Field(ge=1, le=30)
    target_type: str = Field(default="classification", pattern="^(classification|regression)$")
    class_weights: Optional[str] = "balanced"

class Optimization(BaseModel):
    enabled: bool = False
    n_trials: int = 20
    timeout: int = 3600  # секунд

class ExperimentConfig(BaseModel):
    data: DataFilters
    model: ModelParams
    training: TrainingConfig
    optimization: Optimization = Optimization()
    output_dir: str = "./models"
    log_dir: str = "./logs"
