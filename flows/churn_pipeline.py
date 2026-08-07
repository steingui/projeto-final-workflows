import os
import time
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from prefect import flow, task, get_run_logger
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.tracking import MlflowClient

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")

def get_db_engine():
    return create_engine(DATABASE_URL)

@task(retries=3, retry_delay_seconds=5)
def ingest_data_to_bronze(num_records: int = 1000):
    """
    Ingere dados brutos na camada Bronze do Postgres.
    Garante idempotência usando INSERT ON CONFLICT DO UPDATE (Upsert).
    """
    logger = get_run_logger()
    logger.info(f"Iniciando a ingestão de {num_records} registros na camada Bronze...")

    # Gerar dados sintéticos de churn
    np.random.seed(42)
    customer_ids = [f"CUST-{i:05d}" for i in range(1, num_records + 1)]
    ages = np.random.randint(18, 80, size=num_records)
    tenures = np.random.randint(1, 72, size=num_records)
    monthly_charges = np.random.uniform(20.0, 120.0, size=num_records)
    # total_charges = tenure * monthly_charges + some noise
    total_charges = tenures * monthly_charges + np.random.normal(0, 50, size=num_records)
    total_charges = np.clip(total_charges, 20.0, None)
    
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_records, p=[0.5, 0.25, 0.25])
    paperless_billing = np.random.choice([0, 1], size=num_records, p=[0.3, 0.7])
    
    # Churn probability based on features
    churn_prob = 1 / (1 + np.exp(-(
        -0.02 * ages + 
        -0.05 * tenures + 
        0.02 * monthly_charges + 
        (contract_types == "Month-to-month") * 1.5 - 
        1.0
    )))
    churns = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "tenure": tenures,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "paperless_billing": paperless_billing,
        "churn": churns,
        "updated_at": [datetime.datetime.utcnow()] * num_records
    })

    # Criar a tabela bronze se não existir
    with get_db_engine().begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze_customers (
                customer_id VARCHAR(50) PRIMARY KEY,
                age INT,
                tenure INT,
                monthly_charges NUMERIC,
                total_charges NUMERIC,
                contract_type VARCHAR(50),
                paperless_billing INT,
                churn INT,
                updated_at TIMESTAMP
            );
        """))

        # Executar Upsert em lote para garantir idempotência
        for idx, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO bronze_customers (customer_id, age, tenure, monthly_charges, total_charges, contract_type, paperless_billing, churn, updated_at)
                VALUES (:customer_id, :age, :tenure, :monthly_charges, :total_charges, :contract_type, :paperless_billing, :churn, :updated_at)
                ON CONFLICT (customer_id) DO UPDATE SET
                    age = EXCLUDED.age,
                    tenure = EXCLUDED.tenure,
                    monthly_charges = EXCLUDED.monthly_charges,
                    total_charges = EXCLUDED.total_charges,
                    contract_type = EXCLUDED.contract_type,
                    paperless_billing = EXCLUDED.paperless_billing,
                    churn = EXCLUDED.churn,
                    updated_at = EXCLUDED.updated_at;
            """), row.to_dict())

    logger.info("Ingestão concluída com sucesso na tabela bronze_customers!")
    return len(df)

@task(retries=2, retry_delay_seconds=10)
def clean_and_transform_bronze_to_silver():
    """
    Consome da camada Bronze, realiza limpeza, tipagem, feature engineering
    e grava na camada Silver. Sobrescreve para manter a idempotência.
    """
    logger = get_run_logger()
    logger.info("Lendo dados da camada Bronze...")
    
    df = pd.read_sql("SELECT * FROM bronze_customers", get_db_engine())
    
    logger.info("Executando transformações e limpeza na camada Silver...")
    # Limpeza/Tratamento de nulos
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(df["tenure"] * df["monthly_charges"])
    
    # Feature Engineering (One-Hot Encoding para contract_type)
    df["contract_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
    df["contract_one_year"] = (df["contract_type"] == "One year").astype(int)
    df["contract_two_year"] = (df["contract_type"] == "Two year").astype(int)
    
    # Salvar na camada Silver (com exclusão prévia ou sobrescrita do pandas to_sql)
    with get_db_engine().begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS silver_customers;"))
        
    df.to_sql("silver_customers", get_db_engine(), index=False, if_exists="fail")
    logger.info("Dados limpos e gravados com sucesso na tabela silver_customers!")
    return len(df)

@task(retries=2, retry_delay_seconds=30)
def train_and_track_models():
    """
    Treina modelos de Machine Learning (Random Forest e XGBoost)
    e rastreia hiperparâmetros/métricas com o MLflow.
    """
    logger = get_run_logger()
    logger.info("Carregando dados da camada Silver para treinamento...")
    
    df = pd.read_sql("SELECT * FROM silver_customers", get_db_engine())
    
    # Features e label
    feature_cols = ["age", "tenure", "monthly_charges", "total_charges", 
                    "paperless_billing", "contract_month_to_month", 
                    "contract_one_year", "contract_two_year"]
    X = df[feature_cols]
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar MLflow Tracking
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("Churn_Prediction")
    
    # Modelo 1: Random Forest
    with mlflow.start_run(run_name="RandomForestClassifier"):
        logger.info("Treinando RandomForestClassifier...")
        n_estimators = 100
        max_depth = 8
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(rf, "model", registered_model_name="churn_rf_model")
        logger.info(f"RF Treinado. F1-Score: {f1:.4f}")
        
    # Modelo 2: XGBoost
    with mlflow.start_run(run_name="XGBClassifier"):
        logger.info("Treinando XGBClassifier...")
        learning_rate = 0.1
        max_depth = 5
        xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        
        preds = xgb.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.xgboost.log_model(xgb, "model", registered_model_name="churn_xgb_model")
        logger.info(f"XGB Treinado. F1-Score: {f1:.4f}")

@task(retries=2, retry_delay_seconds=15)
def register_and_promote_best_model():
    """
    Busca o melhor modelo treinado com base na métrica F1-Score no MLflow
    e promove para o estágio de 'Production' no Model Registry.
    """
    logger = get_run_logger()
    logger.info("Identificando o melhor modelo...")
    
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_url)
    
    # Buscar todos os runs do experimento
    experiment = client.get_experiment_by_name("Churn_Prediction")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    best_f1 = -1
    best_model_name = None
    best_run_id = None
    
    for run in runs:
        f1 = run.data.metrics.get("f1_score", 0)
        run_name = run.data.tags.get("mlflow.runName", "")
        if f1 > best_f1:
            best_f1 = f1
            best_run_id = run.info.run_id
            best_model_name = "churn_rf_model" if "RandomForest" in run_name else "churn_xgb_model"
            
    logger.info(f"Melhor modelo encontrado: {best_model_name} com F1-Score: {best_f1:.4f}")
    
    # Registrar/Promover a versão mais recente deste modelo para Production
    model_versions = client.get_latest_versions(best_model_name, stages=["None"])
    if model_versions:
        latest_version = model_versions[0].version
        logger.info(f"Promovendo {best_model_name} versão {latest_version} para Production...")
        
        # Transicionar estágios anteriores em Production para Archived
        try:
            prod_versions = client.get_latest_versions(best_model_name, stages=["Production"])
            for pv in prod_versions:
                client.transition_model_version_stage(
                    name=best_model_name,
                    version=pv.version,
                    stage="Archived"
                )
        except Exception as e:
            logger.warning(f"Nenhum modelo de produção anterior arquivado: {e}")
            
        client.transition_model_version_stage(
            name=best_model_name,
            version=latest_version,
            stage="Production"
        )
        
        # Salvar metadados do modelo de produção selecionado em uma tabela/arquivo local para a API ler
        with get_db_engine().begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS active_model (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    version VARCHAR(50),
                    f1_score NUMERIC,
                    promoted_at TIMESTAMP
                );
            """))
            conn.execute(text("""
                INSERT INTO active_model (model_name, version, f1_score, promoted_at)
                VALUES (:model_name, :version, :f1_score, :promoted_at);
            """), {
                "model_name": best_model_name,
                "version": latest_version,
                "f1_score": best_f1,
                "promoted_at": datetime.datetime.utcnow()
            })
        logger.info(f"Modelo {best_model_name} promovido e metadados persistidos!")
    else:
        logger.error("Nenhuma versão de modelo encontrada para promoção!")

@flow(name="churn_ml_pipeline")
def churn_ml_pipeline():
    """
    Fluxo principal orquestrado do Prefect.
    """
    ingest_data_to_bronze(1000)
    clean_and_transform_bronze_to_silver()
    train_and_track_models()
    register_and_promote_best_model()

if __name__ == "__main__":
    churn_ml_pipeline()
