import os
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from sqlalchemy import create_engine, text
from prefect.client import get_client

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
engine = create_engine(DATABASE_URL)

# Configure MLflow
mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_url)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API para predição em tempo real de Churn de clientes, alimentada por modelos gerenciados no MLflow.",
    version="1.0.0"
)

# Cache do modelo em memória
model_cache = {
    "model": None,
    "model_name": None,
    "version": None
}

class CustomerInput(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float
    paperless_billing: int
    contract_type: str  # "Month-to-month", "One year", "Two year"

def load_latest_model():
    """
    Verifica a tabela active_model no PostgreSQL para carregar
    ou atualizar o modelo em produção no cache.
    """
    try:
        with engine.connect() as conn:
            # Verifica se a tabela existe
            table_check = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'active_model');"
            )).scalar()
            
            if not table_check:
                return False

            result = conn.execute(text(
                "SELECT model_name, version FROM active_model ORDER BY id DESC LIMIT 1;"
            )).fetchone()
            
            if result:
                model_name, version = result[0], result[1]
                
                # Se o modelo no cache é diferente ou nulo, atualiza
                if model_cache["version"] != version or model_cache["model_name"] != model_name:
                    print(f"Carregando modelo do MLflow: {model_name} (versão: {version})...")
                    # Carrega direto do Model Registry usando a URI de Produção
                    model_uri = f"models:/{model_name}/Production"
                    if "rf" in model_name:
                        import mlflow.sklearn
                        model_cache["model"] = mlflow.sklearn.load_model(model_uri)
                    elif "xgb" in model_name:
                        import mlflow.xgboost
                        model_cache["model"] = mlflow.xgboost.load_model(model_uri)
                    else:
                        model_cache["model"] = mlflow.pyfunc.load_model(model_uri)
                    model_cache["model_name"] = model_name
                    model_cache["version"] = version
                    print("Modelo carregado no cache com sucesso!")
                return True
    except Exception as e:
        print(f"Erro ao carregar o modelo de produção: {e}")
    return False

@app.on_event("startup")
def startup_event():
    # Tenta carregar o modelo inicial
    load_latest_model()

@app.get("/health")
def health():
    """
    Endpoint de verificação de saúde da API e status do modelo.
    """
    model_loaded = load_latest_model()
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "active_model": model_cache["model_name"],
        "active_version": model_cache["version"]
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    """
    Realiza a predição de Churn para o cliente enviado.
    """
    # Garante que o modelo atualizado está no cache
    if not load_latest_model() and model_cache["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo de predição ativo no momento. Por favor execute o pipeline primeiro."
        )
    
    # Pré-processamento das features de entrada
    contract_month_to_month = 1 if customer.contract_type == "Month-to-month" else 0
    contract_one_year = 1 if customer.contract_type == "One year" else 0
    contract_two_year = 1 if customer.contract_type == "Two year" else 0

    features = [
        customer.age,
        customer.tenure,
        customer.monthly_charges,
        customer.total_charges,
        customer.paperless_billing,
        contract_month_to_month,
        contract_one_year,
        contract_two_year
    ]
    
    # Converter para dataframe/array
    X = pd.DataFrame([features], columns=[
        "age", "tenure", "monthly_charges", "total_charges", 
        "paperless_billing", "contract_month_to_month", 
        "contract_one_year", "contract_two_year"
    ])
    
    model = model_cache["model"]
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    churn_probability = float(probabilities[1])

    return {
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "decision": "Alto Risco de Cancelamento (Churn)" if prediction == 1 else "Baixo Risco",
        "model_used": model_cache["model_name"],
        "model_version": model_cache["version"]
    }

@app.post("/trigger")
async def trigger_pipeline():
    """
    Dispara a execução do pipeline de ML no Prefect Server sob demanda.
    """
    try:
        async with get_client() as client:
            # Buscar o deployment
            deployment = await client.read_deployment_by_name("churn_ml_pipeline/churn-ml-deployment")
            flow_run = await client.create_flow_run_from_deployment(deployment.id)
            return {
                "status": "pipeline_triggered",
                "flow_run_id": str(flow_run.id),
                "info": "A execução do pipeline foi iniciada com sucesso no Prefect Server."
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao disparar o pipeline no Prefect: {str(e)}"
        )
