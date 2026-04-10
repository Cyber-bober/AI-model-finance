from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from src.config.schema import ExperimentConfig
from src.models.trainer import run_training
import os, json, uuid, yaml

app = FastAPI(title="QuantML Local API")

class TrainRequest(BaseModel):
    config: dict  # UI отправляет JSON, который валидируется Pydantic

@app.get("/api/filters")
async def get_available_filters():
    # Возвращает список инструментов, диапазоны дат, доступные индикаторы
    # В реальности читается из кэша или лёгкой БД
    return {
        "instruments": ["AAPL", "MSFT", "BTC-USD", "SBER.ME"],
        "frequencies": ["1H", "4H", "1D", "1W"],
        "indicators": ["RSI_14", "MACD", "ATR_14", "BBANDS", "OBV", "VWAP"],
        "date_range": {"min": "2015-01-01", "max": "2025-12-31"}
    }

@app.post("/api/train")
async def start_training(req: TrainRequest, bg: BackgroundTasks):
    try:
        cfg = ExperimentConfig(**req.config)  # строгая валидация
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {e}")

    run_id = uuid.uuid4().hex[:8]
    cfg.output_dir = f"./models/{run_id}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Сохраняем конфиг для истории
    with open(f"{cfg.output_dir}/config.yaml", "w") as f:
        yaml.dump(req.config, f)

    bg.add_task(run_training, cfg, run_id)
    return {"run_id": run_id, "status": "queued"}

@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    # Простой статус-файл, обновляемый в процессе обучения
    status_path = f"./models/{run_id}/status.json"
    if not os.path.exists(status_path):
        return {"status": "not_found"}
    with open(status_path) as f:
        return json.load(f)
