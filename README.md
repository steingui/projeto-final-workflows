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

---

## 🎯 Contexto e Problema

Churn (cancelamento de clientes) é uma das métricas mais críticas em negócios de assinatura. O custo de adquirir um novo cliente é 5–7x maior que o de retê-lo. Este pipeline resolve o problema de **identificar proativamente clientes com alto risco de cancelamento**, permitindo que equipes de CRM atuem antes do cancelamento acontecer.

O desafio central não é apenas treinar um modelo, mas garantir que o ciclo completo — da ingestão ao serving — seja **reprodutível, idempotente e observável**, funcionando de forma autônoma em produção sem intervenção manual.

---

## 🧠 Decisões Técnicas

### Por que Prefect e não Airflow?

O Prefect 3.x foi escolhido por três razões concretas:

1. **Deploy sem infraestrutura extra**: O `flow.serve()` sobe um worker polling com uma única linha de Python, sem precisar de um scheduler DAG separado. O Airflow exigiria configurar Celery/Redis ou Kubernetes Executor para distribuição equivalente.
2. **Código-primeiro**: Os flows são funções Python puras decoradas com `@flow`/`@task`, sem XML, YAML ou DSL proprietário. Isso reduz o tempo de onboarding e facilita testes unitários.
3. **Orquestração event-driven nativa**: O endpoint `/trigger` da FastAPI aciona o flow via Prefect Client (`run_deployment`) sem precisar de sensores ou ExternalTaskSensor do Airflow.

### Por que MLflow para rastreamento e não W&B ou outro?

- **Self-hosted**: O MLflow roda completamente on-premise via Docker, sem dependência de serviços externos ou custos variáveis.
- **Model Registry integrado**: Suporta staging (`None → Production → Archived`) nativamente, eliminando a necessidade de controle manual de versões.
- **Backend SQL**: O `backend-store-uri` apontando para PostgreSQL centraliza metadados de experimentos no mesmo banco do pipeline, simplificando backups e consistência transacional.

### Estratégia de Idempotência

- **Camada Bronze**: `INSERT ... ON CONFLICT (customer_id) DO UPDATE` garante upsert por chave natural — re-executar o pipeline nunca duplica registros.
- **Camada Silver**: `DROP TABLE IF EXISTS` + `to_sql` recria a tabela derivada a partir do Bronze, que é a fonte de verdade. Operação determinística e segura para re-execução.

### Por que PostgreSQL como backend unificado?

Um único PostgreSQL serve como: (1) store de dados do pipeline (Bronze/Silver/active_model), (2) backend do Prefect Server, e (3) backend do MLflow. Isso reduz a superfície de infraestrutura, facilita backups atômicos e elimina problemas de sincronização entre datastores heterogêneos.

### Resiliência por camada

Cada `@task` tem política de retry proporcional ao seu custo computacional:
- **Ingestão** (`retries=3, delay=5s`): falhas de rede/DB são transitórias e rápidas de resolver.
- **Transformação** (`retries=2, delay=10s`): operações de leitura/escrita no Postgres, delay maior para evitar lock contention.
- **Treinamento** (`retries=2, delay=30s`): operação cara; delay longo para o MLflow Server se estabilizar após restart.
- **Promoção** (`retries=2, delay=15s`): chamadas à API do MLflow Client.

---

## 🎬 Vídeo de Demonstração

> 🔗 Link do vídeo: *(a ser preenchido antes da entrega final)*

O vídeo cobre: contexto do problema, arquitetura da solução, demonstração do pipeline rodando end-to-end e principais decisões técnicas.
