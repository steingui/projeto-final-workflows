# Pipeline de ML End-to-End para Previsão de Churn

Este projeto implementa um pipeline de dados e Machine Learning completo, idempotente e resiliente para prever o cancelamento de clientes (Churn), estruturado em camadas de dados (Bronze e Silver) e orquestrado localmente de forma automatizada.

---

## 🏗️ Arquitetura do Sistema

O projeto é modular e distribuído em microserviços containerizados usando **Docker Compose**:

1. **PostgreSQL (`db`)**: Armazena os dados brutos (camada Bronze), processados (camada Silver), metadados do modelo campeão em produção (`active_model`) e serve como backend-store do Prefect e do MLflow.
2. **MLflow Server (`mlflow`)**: Gerencia o rastreamento de experimentos, logs de parâmetros/métricas dos modelos e o Model Registry para controle de versão dos modelos campeões.
3. **Prefect Server (`prefect-server`)**: Interface gráfica e motor de agendamento de fluxos.
4. **Prefect Worker (`prefect-worker`)**: Agente responsável por executar o pipeline de treinamento em background (polling) de forma assíncrona ou agendada.
5. **FastAPI Application (`fastapi-app`)**: API REST que expõe os endpoints para inferência em tempo real e permite acionar o pipeline sob demanda.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.11**
- **FastAPI / Uvicorn**
- **Prefect 3.x**
- **MLflow**
- **PostgreSQL**
- **SQLAlchemy 2.x / asyncpg**
- **Pandas / Scikit-Learn / XGBoost**

---

## 🚀 Como Executar

### 1. Iniciar a Infraestrutura
Na raiz do projeto, execute o comando para construir e subir todos os serviços em background:
```bash
docker compose up -d --build
```

### 2. Verificar os Status dos Serviços
Você pode monitorar a inicialização através dos logs:
```bash
docker compose logs -f
```

- **Prefect UI**: Disponível em [http://localhost:4200](http://localhost:4200)
- **MLflow UI**: Disponível em [http://localhost:5000](http://localhost:5000)
- **FastAPI Docs**: Disponível em [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚡ Utilização dos Endpoints da API

### A. Acionar o Pipeline Manualmente (Trigger)
Para forçar a execução imediata do pipeline do Prefect de treino e seleção de modelos:
```bash
curl -X POST http://localhost:8000/trigger
```

### B. Predição em Tempo Real (Inferência)
Envie um payload JSON com as características do cliente para classificar o risco de Churn. A API carregará dinamicamente o modelo campeão definido em estágio `Production` no MLflow Model Registry:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "tenure": 12,
    "monthly_charges": 70.5,
    "total_charges": 846.0,
    "paperless_billing": 1,
    "contract_type": "Month-to-month"
  }' \
  http://localhost:8000/predict
```

#### Resposta Esperada:
```json
{
  "prediction": 1,
  "churn_probability": 0.6626,
  "decision": "Alto Risco de Cancelamento (Churn)",
  "model_used": "churn_rf_model",
  "model_version": "2"
}
```

---

## 🔄 Funcionamento do Pipeline (Prefect Flow)

1. **Ingestão Bronze (`ingest_data_to_bronze`)**: Gera registros simulados e insere na tabela `bronze_customers` com política de upsert (`ON CONFLICT DO UPDATE`) para garantir idempotência.
2. **Transformação Silver (`clean_and_transform_bronze_to_silver`)**: Realiza limpeza, tratamento de valores ausentes e feature engineering (One-Hot Encoding para tipos de contrato).
3. **Treinamento e Tracking (`train_and_track_models`)**:
   - Treina múltiplos modelos em paralelo (**Random Forest** e **XGBoost**).
   - Loga parâmetros, métricas de performance (Acurácia, Precisão, Recall e F1-Score) e os artefatos de cada run no MLflow.
4. **Registro e Promoção (`register_and_promote_best_model`)**: 
   - Compara as métricas do histórico.
   - Seleciona o modelo com o maior **F1-Score**.
   - Atualiza a tag no MLflow Model Registry para `Production` e grava os metadados do campeão na tabela `active_model` no Postgres para consumo imediato da API.
