# Movie Recommendation Pipeline – MLOps Mini Project
## 1. Project Overview
본 프로젝트는 TMDB 영화 데이터를 기반하여 영화 추천 모델을 대상으로,  
MLOps 아키텍처 중 **모델 학습 자동화**와 **Airflow 기반 파이프라인 오케스트레이션** 구현에 집중한 미니 프로젝트입니다.
- 데이터 수집부터 학습, 평가, 추론까지의 머신러닝 파이프라인을 모듈화
- 컨테이너 기반 Airflow를 사용하여 학습 파이프라인을 자동화
* 본 프로젝트에서는 CI/CD 파이프라인 및 고급 모니터링 시스템은 범위에서 제외하였습니다. 

## 2. Architecture Scope
### Focus Area (Implemented)
#### Automated Pipeline
- Model Training (current)
- 확장 가능 구조: Data Preparation / Evaluation / Inference
#### Container-based Orchestration
- Docker 기반 Airflow 운영
- DockerOperator를 통한 ML 작업 실행
#### Model Serving
- (Design only) FastAPI 기반 모델 서빙 구조 설계
### Out of Scope
- Online Model Serving (FastAPI runtime)
- CI/CD Pipeline
- Feature Store
- Model Registry
- Performance / Drift Monitoring
### Model Serving (Design Only)
본 프로젝트에서는 모델 서빙을 FastAPI 기반으로 확장할 수 있도록 inference 및 api 모듈을 설계하였으나,  
이번 범위에서는 컨테이너 기반 학습 파이프라인 구현에 집중하여 실제 FastAPI 서버 실행 및 배포는 포함하지 않았습니다.

## 3. System Architecture
```text
[TMDB API]
     ↓
[data_prepare]
     ↓
[Watch Log Dataset]
     ↓
[Train (Movie Recommendation Model)]
     ↓
[Airflow (DockerOperator)]
     ↓
[Trainer Docker Container]
     ↓
[Model Artifact (.pkl + hash)]
     ↓
[S3 (External Model Storage)]
```

## 4. Project Structure
```text
├── airflow/
│   ├── dags/
│   │   └── mlops_automated_dag.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── trainer/
│   ├── dataset/
│   │   └── raw/
│   │       └── popular.json
│   └── src/
│       ├── data_prepare/
│       │   ├── crawler.py
│       │   ├── preprocessing.py
│       │   └── main.py
│       ├── dataset/
│       │   ├── data_loader.py
│       │   └── watch_log.py
│       ├── train/
│       │   └── train.py
│       ├── evaluate/
│       │   └── evaluate.py
│       ├── inference/
│       │   └── inference.py
│       ├── model/
│       │   └── movie_predictor.py
│       ├── postprocess/
│       │   └── postprocess.py
│       ├── utils/
│       │   ├── utils.py
│       │   └── s3.py
│       ├── api.py
│       └── main.py
│
├── Dockerfile
├── requirements.txt
├── .env.template
└── README.md
```

## 5. Core Components
### 5.1 Data Preparation (data_prepare)
- 원천 데이터 수집 및 정제
- 학습에 필요한 데이터셋 생성
- 파이프라인의 선행 단계 (현재 DAG에는 미포함, 확장 가능)
### 5.2 Training (train)
- Scikit-learn 기반 영화 평점 예측 모델 학습
- 학습 결과를 .pkl 파일로 저장
- 모델 무결성 검증을 위한 hash 파일 생성
### 5.3 Evaluation & Inference (evaluate, inference)
- 학습된 모델 로드 및 성능 평가
- 단건 / 배치 추론 로직 제공
- 학습 시 사용한 전처리 객체 재사용
### 5.4 Postprocess (postprocess)
- 추론 결과 후처리
- 결과 저장 또는 외부 시스템 연동을 위한 확장 포인트
### 5.5 Utilities (utils)
- 공통 유틸리티 함수 관리

## 6. Automated Pipeline (Container-based Airflow)
본 프로젝트에서는 Apache Airflow를 컨테이너 기반으로 운영하며,  
각 ML 작업을 DockerOperator를 통해 독립된 컨테이너에서 실행합니다.
### DAG Overview
DAG ID: mlops_automated_pipeline  
Schedule: 0 0 * * * (매일 09:00 KST 실행)  
Catchup: Disabled
### Implemented Task
#### model_training
- Docker Image: {DOCKERHUB_USERNAME}/mlops-trainer:latest
- Command:
```text
python -m src.main train
```
- 역할:
  - 학습 파이프라인 실행
  - 모델 아티팩트 생성 및 저장
### Environment Variables
Airflow는 다음 환경 변수를 Trainer 컨테이너에 주입합니다:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION (default: ap-northeast-2)
- S3_BUCKET_NAME

## 7. Tech Stack
- Language: Python
- ML: scikit-learn
- Pipeline Orchestration: Apache Airflow
- Execution Runtime: Docker, DockerOperator
- Model Serving(Design only): FastAPI
- Data Handling: pandas, numpy

## 8. Design Decisions
### Automated Pipeline 중심 설계
MLOps Lv2 가이드에서 강조하는 핵심 컴포넌트 구현
### 컨테이너 기반 Airflow 운영
실행 환경 분리 및 재현성 확보
### DockerOperator 활용
오케스트레이션과 실행 환경의 명확한 분리
### 과도한 MLOps 구성 요소 배제
과제 및 포트폴리오 목적에 맞는 범위 설정

## 9. Future Work
- Data Preparation / Evaluation / Inference 태스크를 DAG로 확장
- CI 기반 테스트 자동화
- Model Registry 연동
- 데이터 드리프트 감지
