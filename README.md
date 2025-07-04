# 게임 커뮤니티 특화 감성 분석 모델 구축 및 자동화 파이프라인
**부제: LLM과 Human-in-the-Loop를 통한 MLOps 기반 모델 지속 성장 시스템**

## 1. 📖 프로젝트 개요

본 프로젝트는 일반적인 데이터로 학습된 감성 분석 모델이 특정 도메인(게임 커뮤니티)의 은어와 뉘앙스를 정확히 파악하지 못하는 문제를 해결하기 위해 시작되었습니다. **범용(General-Domain) 모델을 특정 목적(Domain-Specific)에 맞게 지속적으로 똑똑하게 만드는 시스템**을 구축하는 것을 목표로 합니다.

이를 위해 메이플스토리 커뮤니티 데이터를 기반으로 **특화된 감성 분석 모델(Domain-Specific Sentiment Analysis Model)을** 구축하고, 나아가 **MLOps(Machine Learning Operations)** 파이프라인을 설계하여 데이터 수집, 자동 라벨링, 재학습, 배포, 그리고 사용자 피드백 반영까지 이어지는 전체 생명주기를 자동화했습니다.

이 아키텍처는 메이플스토리뿐만 아니라 다른 게임 커뮤니티로도 쉽게 확장 가능하며, 지속적으로 살아 움직이는 머신러닝 시스템을 구축하는 경험을 쌓는 데 중점을 두었습니다.

---

## 2. 🌟 시스템 주요 기능

* **병렬 처리 파이프라인**: 데이터 수집 프로세스와 데이터 처리/학습 프로세스를 분리하여 24시간 멈추지 않고 동작합니다.
* **LLM 기반 자동 라벨링**: 로컬에서 실행되는 Gemma 모델(LM Studio)을 활용하여 수집된 데이터에 '긍정/중립/부정' 라벨을 자동으로 부여합니다.
* **점진적 학습 (Incremental Learning)**: 신규 데이터와 사용자 피드백, 그리고 방대한 기존 데이터를 적절히 조합하여 모델이 최신 트렌드를 반영하면서도 일반화 성능을 잃지 않도록 학습합니다.
* **모델 버전 관리**: 학습된 모델을 덮어쓰지 않고 타임스탬프 기반의 버전으로 관리하여, 언제든 이전 버전으로 롤백하거나 성능 변화를 추적할 수 있습니다.
* **인간 참여형 루프 (Human-in-the-Loop)**: Streamlit 앱을 통해 사용자가 모델의 예측 결과를 수정하고 피드백을 제출할 수 있습니다. 이 고품질 데이터는 다음 모델 학습에 반영되어 시스템의 정확도를 지속적으로 향상시킵니다.
* **인터랙티브 대시보드**: 최신 데이터를 기반으로 직업군/직업별 감성 분포와 주요 키워드를 시각화하고, 실시간으로 텍스트 감성을 분석할 수 있는 웹 애플리케이션을 제공합니다.

---

## 3. 🛠️ 기술 스택 및 핵심 역량 (Tech Stack & Skills)

| 구분 | 기술 스택 및 역량 |
| :--- | :--- |
| **Backend & ML** | `Python`, `TensorFlow`, `Keras`, `Scikit-learn`, `Konlpy` |
| **LLM & Prompting** | `Langchain`, `LM Studio`, `Gemma`, `Prompt Engineering` |
| **MLOps & Automation**| `Multiprocessing`, `APScheduler`, `Git`, `Git LFS` |
| **Frontend & VIz** | `Streamlit`, `Pandas`, `Matplotlib`, `WordCloud` |
| **핵심 경험 역량** | **MLOps 파이프라인 설계 및 구축**, **병렬 처리 아키텍처 구현**, **Human-in-the-Loop(HITL) 기반 모델 개선 사이클** 설계, **점진적 학습(Incremental Learning)** 및 **전이 학습(Transfer Learning)** 개념 적용, **모델 버전 관리** 및 배포 전략 수립 |

---

## 4. ⚙️ MLOps 아키텍처
**데이터 수집기(Crawler)** 와 **파이프라인 Orchestrator**이 독립적으로 동작하는 **비동기 병렬 처리 구조**로 설계되었습니다.

```mermaid
graph TD
    subgraph "프로세스 1: 크롤러 (crawler_main.py)"
        A[무한 루프 시작] --> B{주기적 사이트 확인};
        B --> C[신규 게시글 수집];
        C --> D[all_posts.csv에 누적 저장];
        D --> A;
    end

    subgraph "프로세스 2: 오케스트레이터 (main.py)"
        E[무한 루프 시작] --> F{주기적 조건 확인};
        F -- "신규 데이터 1000건 이상?" --> G[raw_chunk 생성];
        G --> H[라벨링 실행];
        F -- "사용자 피드백 100건 이상?" --> I[최신 청크 데이터 로드];
        
        subgraph "학습 파이프라인 (training_pipeline.py)"
            J[데이터 결합]
            J --> K[모델 학습 및 평가]
            K --> L[새 버전으로 모델 저장]
        end

        H --> J;
        I --> J;
    end

    subgraph "사용자 인터페이스 (app.py)"
        M[Streamlit 앱 실행] --> N{최신 모델/데이터 로드};
        N --> O[대시보드 시각화];
        N --> P[실시간 감성 분석];
        P --> Q[user_feedback.csv에 저장];
    end

    D -.-> F;
    Q -.-> F;
    L -.-> N;
```

* **프로세스 1 (크롤러)**: 멈추지 않고 계속해서 메이플 인벤의 새로운 게시글을 수집하여 `all_posts.csv`에 쌓습니다.
* **프로세스 2 (오케스트레이터)**: 주기적으로 깨어나 **두 가지 조건**을 확인합니다.
    1.  새로 수집된 데이터가 1,000건 이상 쌓였는가?
    2.  사용자 피드백이 100건 이상 쌓였는가?
    * 위 조건 중 하나라도 충족되면, 데이터 라벨링과 모델 재학습 파이프라인을 실행합니다. 이 모든 과정 동안 크롤러는 영향을 받지 않고 계속 동작합니다.

---

## 5. 🗃️ 데이터 구조
**디렉토리구조**
```bash
project
├── data
│   ├── raw_chunks
│   │   ├── raw_data_01.csv
│   │   └── raw_data_02.csv
│   ├── labeled_chunks
│   │   ├── labeled_data_01.csv
│   │   └── labeled_data_02.csv
│   ├── archived_feedback
│   ├── all_posts.csv
│   ├── korean_stopwords.txt
│   ├── merged_label_final.csv
│   ├── pipeline_state.json
│   └── user_feedback.csv
├── lib
│   └── crawler_noCmt.py
├── new_model
│   ├── best_maple_model.h5
│   └── tokenizer.pkl
├── production_models
│   └── v_initial_model
│   │   ├── model.h5
│   │   └── tokenizer.pkl
├── app.py
├── crawer_main.py
├── inference_app.py
├── labeler_main.py
├── myLangchainService.py
├── training_pipeline.py
└── main.py

```

| 파일/폴더명 | 설명 | 생성 주체 | 사용 주체 |
| :--- | :--- | :--- | :--- |
| `all_posts.csv` | 모든 크롤링 데이터가 누적되는 원본 파일 | `crawler_main.py` | `main.py` |
| `/raw_chunks/` | `all_posts.csv`에서 1000건 단위로 잘라낸 라벨링 대상 파일 (`raw_data_XX.csv`) | `main.py` | `labeler_main.py` |
| `/labeled_chunks/`| 라벨링이 완료된 데이터 파일 (`labeled_data_XX.csv`) | `labeler_main.py` | `training_pipeline.py`, `app.py` |
| `user_feedback.csv`| Streamlit 앱에서 사용자가 직접 입력/수정한 고품질 데이터 | `app.py` | `training_pipeline.py` |
| `/archived_feedback/`| 학습에 사용된 피드백 데이터가 백업되는 폴더 | `main.py` | - |
| `merged_label_final.csv`| (초기 데이터) 일반적인 게임 도메인 데이터. 모델의 기반 지식을 위해 사용 | 수동 | `training_pipeline.py` |
| `korean_stopwords.txt`| (초기 데이터) 한국어 불용어 사전 | 수동 | `training_pipeline.py`, `app.py` |

---

## 6. 🧠 모델 학습 방식

모델의 성능을 지속적으로 향상시키기 위해 다음과 같은 학습 전략을 사용합니다.

1.  **데이터 구성**: 재학습 시, 아래 3종류의 데이터를 모두 결합하여 사용합니다.
    * **신규 메이플 데이터**: 가장 최신 트렌드를 반영하는 데이터 (LLM 라벨링)
    * **사용자 피드백 데이터**: 사람이 직접 검증한 고품질 정답 데이터
    * **일반 게임 데이터 (샘플링)**: 모델이 메이플스토리 데이터에만 과적합되는 것을 방지하고, 게임 도메인 전반의 문맥을 이해하도록 돕는 기반 데이터. 신규 데이터와 1:1 비율로 샘플링하여 사용합니다.

2.  **모델 아키텍처**:
    * **Stacked Bi-LSTM with Dropout**: 단순한 Bi-LSTM을 넘어, 여러 층을 쌓고 각 층 사이에 Dropout을 추가하여 더 복잡한 문맥을 학습하면서도 과적합을 효과적으로 방지하는 구조를 사용합니다.

      ```
      Embedding -> Dropout -> Bi-LSTM(return_seq) -> Dropout -> Bi-LSTM -> Dense -> Dropout -> Output
      ```

2.  **평가지표**:
<table>
  <thead>
    <tr>
      <th>평가항목</th>
      <th>결과</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Model Summary</strong></td>
      <td><img src="./assets/model_summary.png" alt="Model Summary" width="600" height="300"></td>
    </tr>
    <tr>
      <td><strong>Training_History</strong></td>
      <td><img src="./assets/training_history.png" alt="Training_History" width="600" height="300"></td>
    </tr>
    <tr>
      <td><strong>Classification Report</strong></td>
      <td><img src="./assets/Classification Report.png" alt="Classification Report" width="600" height="300"></td>
    </tr>
    <tr>
      <td><strong>Confusion_Matrix</strong></td>
      <td><img src="./assets/confusion_matrix.png" alt="Confusion_Matrix" width="600" height="300"></td>
    </tr>
  </tbody>
</table>

---


## 7. 🚀 실행 방법

### 7.1. 환경 설정

1.  Anaconda 가상환경 2개를 생성합니다.
    ```bash
    conda create -n nlp-tfgpu python=3.9
    conda create -n LangchainEnv python=3.9
    ```
2.  각 환경을 활성화하고, 필요한 라이브러리를 설치합니다.
    ```bash
    # nlp-tfgpu 환경
    conda activate nlp-tfgpu
    pip install -r requirements_nlp.txt

    # LangchainEnv 환경
    conda activate LangchainEnv
    pip install -r requirements_langchain.txt
    ```
3.  Konlpy(Okt) 사용을 위해 JDK 설치가 필요할 수 있습니다.

### 7.2. 초기 모델 학습 (최초 1회)

1.  `data` 폴더에 `merged_label_final.csv` 와 `korean_stopwords.txt` 파일을 위치시킵니다.
2.  `training_pipeline.py`를 직접 실행하여 초기 베이스 모델을 생성합니다.
    ```bash
    conda activate nlp-tfgpu
    # 인자로 초기 데이터를 임시로 전달하여 실행
    python training_pipeline.py ./data/merged_label_final.csv
    ```
    * 실행이 완료되면 `production_models/` 폴더에 첫 버전의 모델이 생성됩니다.

### 7.3. 파이프라인 실행

1.  LM Studio를 실행하고, 감성 분석에 사용할 모델(예: `google/gemma-2-9b-it`)을 로컬 서버로 실행합니다.
2.  **터미널 1**에서 MLOps 파이프라인을 시작합니다.
    ```bash
    conda activate nlp-tfgpu # 또는 main.py가 있는 환경
    python main.py
    ```
    * 이제 크롤러와 오케스트레이터가 동시에 실행되며 자동화 파이프라인이 시작됩니다.

3.  **터미널 2**에서 사용자용 대시보드 앱을 실행합니다.
    ```bash
    conda activate nlp-tfgpu
    streamlit run app.py
    ```
    * 웹 브라우저에서 `http://localhost:8501` 주소로 접속하여 대시보드를 확인합니다.
      
| 기능 | 실행 화면 |
| :--- | :--- |
| **크롤링된 데이터 직업별 감성분포** | ![크롤링된 데이터 직업별 감성분포](./assets/tab1_1.png) |
| **크롤링된 데이터 직업별 워드클라우드** | ![크롤링된 데이터 직업별 워드클라우드](./assets/tab1_2.png) |

| 기능 | 실행 화면 |
| :--- | :--- |
| **사용자 의견 입력** | ![사용자 의견 입력](./assets/tab2_1.png) |
| **사용자 의견 감성 예측** | ![사용자 의견 감성예측](./assets/tab2_2.png) |
| **사용자 의견 피드백 반영** | ![사용자 의견 피드백 반영](./assets/tab2_3.png) |


## 📈 주요 결과 및 성과 (Key Results & Achievements)

본 프로젝트를 통해 다음과 같은 구체적인 성과를 달성했습니다.

* **자동화된 MLOps 파이프라인 구축**: 데이터 수집부터 모델 학습, 버전 관리, 피드백 반영까지 이어지는 전체 머신러닝 파이프라인을 성공적으로 자동화하여, 사람의 개입 없이도 시스템이 스스로 발전할 수 있는 기반을 마련했습니다.

* **게임 도메인 특화 감성 분석 모델 확보**: 약 30만 건의 일반 게임 데이터와 LLM으로 자동 라벨링된 메이플스토리 최신 데이터를 결합하여, **F1-Score 0.6972**인 게임 커뮤니티 특화 감성 분석 모델을 확보했습니다.

* **사용자 피드백 기반 모델 개선 사이클 구현**: Streamlit 앱을 통해 사용자가 직접 모델의 예측을 교정하고, 이 고품질 데이터가 다음 학습에 자동으로 반영되는 **Human-in-the-Loop** 시스템을 구현하여 모델의 점진적인 성능 향상을 가능하게 했습니다.

* **데이터 기반 여론 분석 대시보드 제공**: 수집된 데이터를 바탕으로 직업별 여론 동향과 주요 키워드를 시각적으로 분석할 수 있는 인터랙티브 대시보드를 구축하여, 데이터에 기반한 의사결정을 지원할 수 있는 가능성을 제시했습니다.

---

## 🤔 한계점 및 도전 과제 (Limitations & Challenges)

프로젝트를 진행하며 다음과 같은 한계점과 기술적 도전 과제를 마주했습니다.

* **뉘앙스 및 은어 파악의 한계**: 커뮤니티에서 사용되는 비꼬는 말투, 반어법, 특정 유저들만 아는 은어 등은 현재의 Bi-LSTM 모델뿐만 아니라 LLM조차 정확히 의도를 파악하고 라벨링하는 데 어려움이 있었습니다.

* **데이터 불균형 문제**: 상대적으로 '중립'적인 감성의 데이터가 '긍정'이나 '부정'에 비해 많게 측정이 되었는데 실제로 보면 위에 언급한 이유로 긍정 혹은 부정적인 내용이 중립으로 표시되는 경우가 많았습니다.
 그 결과 모델이 중립적인 문장을 판단하는 데 어려움을 겪는 경향이 나타났습니다.

* **자동라벨링된 데이터의 한계**: 현재 파이프라인은 LLM이 생성한 라벨을 별도의 신뢰도 검증 없이 모두 학습 데이터로 사용합니다. 이로 인해, 분류 확률이 특정 임계값 미만인 모호한 데이터
  (예: 긍정 35%, 중립 32%, 부정 33%)까지 학습에 포함되어 데이터 노이즈(noise)가 증가했고, 이는 모델의 초기 성능을 저해하는 주요 원인으로 작용했습니다. 학습 데이터의 품질을 확보하기 위한
  데이터 정제(Data Curation) 및 검증 단계의 부재가 한계점으로 남습니다. 

* **로컬 환경의 리소스 한계**: 모든 파이프라인을 로컬 PC에서 실행하여, 특히 LLM 라벨링과 모델 학습 단계에서 상당한 시간과 컴퓨팅 자원이 소요되었습니다.

---

## 💡 향후 개선 방향 (Future Work)

본 프로젝트는 다음과 같은 방향으로 더욱 발전될 수 있습니다.

1.  **데이터 중심 AI(Data-Centric AI) 접근법 도입**:
    * **신뢰도 기반 데이터 정제 (Confidence-based Filtering)**: LLM이 라벨링한 데이터 중, 분류 확률이 특정 임계값 미만인 모호한 데이터(예: 긍정 35%, 중립 33%, 부정 32%)는 학습에서 제외하거나 별도의 검수 대상으로 분리하여
      데이터셋의 전체적인 품질(Quality)을 향상시킵니다.
    * **능동 학습(Active Learning) 루프 구현**: 모델이 가장 헷갈려 하는 데이터를 선별하여 우선적으로 사용자 피드백(Human-in-the-Loop)을 요청함으로써, 최소한의 비용으로 모델 성능을 극대화하는 학습 전략을 도입합니다.

2.  **고성능 사전 학습 모델(PLM) 기반 전이 학습(Transfer Learning)**:
    * 현재의 Bi-LSTM 모델을 `Ko-BERT`, `Ko-ELECTRA` 등 한국어에 사전 학습된 대규모 언어 모델로 교체하고, 본 프로젝트에서 수집한 도메인 특화 데이터로 미세 조정(Fine-tuning)하여 모델의 근본적인 자연어 이해(NLU) 성능을 향상시킵니다.

3.  **클라우드 기반 파이프라인 확장 및 자동화 고도화**:
    * `AWS`, `GCP` 등 클라우드 플랫폼으로 파이프라인을 이전하여 로컬 리소스 한계를 극복하고, `Airflow`, `Kubeflow Pipelines` 같은 전문 워크플로우 오케스트레이션 도구를 도입하여 전체 MLOps 파이프라인의 안정성, 확장성, 모니터링
      기능을 강화합니다.

4.  **심층 분석 기능 추가 및 서비스 확장**:
    * 단순 긍/부정 분류를 넘어, **주제 모델링(Topic Modeling)을** 통해 주요 불만/칭찬 요인을 도출하거나, 특정 키워드(예: "신규 보스", "이벤트")에 대한 감성 추이를 시계열로 분석하는 등 더 심층적인 분석 기능을 추가합니다.
