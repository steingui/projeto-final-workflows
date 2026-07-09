import time
import httpx
import os
from flows.churn_pipeline import churn_ml_pipeline

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")

def wait_for_prefect():
    print(f"Aguardando o Prefect Server iniciar em: {PREFECT_API_URL}...")
    while True:
        try:
            response = httpx.get(f"{PREFECT_API_URL}/health")
            if response.status_code == 200:
                print("Prefect Server está ativo e saudável!")
                break
        except Exception:
            pass
        time.sleep(3)

if __name__ == "__main__":
    wait_for_prefect()
    
    # Registra o deploy no Prefect e inicia o Worker loop
    print("Iniciando o Prefect Flow Server...")
    churn_ml_pipeline.serve(
        name="churn-ml-deployment",
        cron="0 0 * * *",  # Agendamento diário
        tags=["production", "churn"]
    )
