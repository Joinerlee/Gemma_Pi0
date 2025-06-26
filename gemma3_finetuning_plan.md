# Gemma 3n 기반 Reasoning-Action VLM 로봇 에이전트 개발 (Pi0 스타일)

## 0. 프로젝트 개요 (Project Overview)

- **목표:** LeRobot 및 Pi0 논문에서 영감을 받아, `<think>` (사고) 토큰과 `<action>` (행동) 토큰을 사용하는 Reasoning VLM 로봇 에이전트를 개발합니다.
- **핵심 모델:** Google의 최신 모델인 **Gemma 3n**을 기반으로 합니다.
- **학습 환경:** 로컬 GPU 및 실제 로봇의 부재로, **Google Colab**을 활용하여 학습을 진행합니다.
- **학습 전략:** **PEFT (Parameter-Efficient Fine-Tuning)**, 특히 **QLoRA**와 같은 기법을 사용하여 최소한의 비용으로 최대 효율을 추구합니다. **양자화(Quantization)** 기술 또한 학습하고 적용하는 것을 목표로 합니다.
- **데이터셋:** 실제 로봇이 없기 때문에, **시뮬레이션(PyBullet)**을 통해 (Vision, Instruction, Think, Action) 데이터셋을 생성하고 활용합니다.
- **추론 강화:** DeepSeek-R과 같이, 단순 모방 학습을 넘어 **강화학습(RL)**을 통해 모델의 `<think>` 과정을 최적화하여 더 나은 추론 능력을 갖춘 에이전트를 만듭니다.
- **결과 관리:** 모든 실험 과정, 특히 **하이퍼파라미터 튜닝**에 대한 근거와 결과를 **보고서**로 상세히 기록하여 경험과 지식을 체계적으로 축적합니다.

---

## 1. 프로젝트 비전 및 최종 목표

### 1.1. 비전
LLM의 추론(Reasoning) 능력을 강화학습(RL)을 통해 직접적으로 최적화하여, 복잡한 시각 및 언어적 지시를 이해하고 효율적인 행동 계획을 수립하여 실행하는 로봇 제어 에이전트를 개발한다. 이는 단순한 모방 학습(Imitation Learning)을 넘어, 모델이 스스로 더 나은 해결책을 탐색하고 학습하는 것을 목표로 한다.

### 1.2. 핵심 접근법: SFT + RL
1.  **1단계: Supervised Fine-Tuning (SFT)**
    - 전문가 데이터(Expert Demonstrations)를 사용하여 모델이 `<think>`와 `<action>`의 기본 구조와 역할을 학습하도록 한다. (기본기 훈련)
2.  **2단계: Reinforcement Learning (RL)**
    - SFT로 학습된 모델을 기반으로, 시뮬레이션 환경과의 상호작용을 통해 얻는 **보상(Reward)**을 사용하여 모델의 `<think>` (추론/계획) 과정을 직접적으로 강화한다. (실전 훈련)

### 1.3. 이론적 기반 및 관련 연구 (Theoretical Foundations & Related Work)

본 프로젝트는 다양한 분야의 선행 연구 아이디어를 융합하여 구축됩니다.

#### 1.3.1. 대규모 언어 모델 (LLM) 및 트랜스포머 (Large Language Models & Transformers)
-   **Attention Is All You Need (Google, 2017)**
    -   **영향:** Transformer 아키텍처를 제안하여 시퀀스 모델링의 패러다임을 바꾸고, 이후 모든 LLM의 기반이 되었습니다.
    -   **논문:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
-   **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Google, 2018)**
    -   **영향:** 양방향 Transformer 인코더를 사용한 사전 학습 모델로, 다양한 NLP 태스크에서 SOTA 성능을 달성하며 사전 학습-파인튜닝 패러다임을 확립했습니다.
    -   **논문:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
-   **Language Models are Few-Shot Learners (OpenAI, 2020) - GPT-3**
    -   **영향:** 대규모 모델이 방대한 데이터로 사전 학습될 경우, 소수의 예시만으로도 새로운 태스크를 수행하는 Few-Shot Learning 능력을 보여주었습니다.
    -   **논문:** [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
-   **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Google, 2019) - T5**
    -   **영향:** 모든 NLP 태스크를 텍스트-투-텍스트 형식으로 통일하여 Transformer 모델로 학습하는 접근법을 제시했습니다.

#### 1.3.2. 비전-언어 모델 (VLM) 및 범용 에이전트 (Vision-Language Models & Generalist Agents)
-   **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Google, 2020) - ViT**
    -   **영향:** 이미지 처리에도 Transformer 아키텍처를 성공적으로 적용하여, Vision Transformer의 시대를 열었습니다.
    -   **논문:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
-   **Learning Transferable Visual Models From Natural Language Supervision (OpenAI, 2021) - CLIP**
    -   **영향:** 이미지와 텍스트 간의 임베딩 공간을 학습하여, 제로샷(zero-shot) 이미지 분류 및 검색 등 다양한 비전-언어 태스크에서 뛰어난 성능을 보였습니다.
    -   **논문:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
-   **Flamingo: a Visual Language Model for Few-Shot Learning (DeepMind, 2022)**
    -   **영향:** 사전 학습된 LLM과 비전 모델을 결합하여 Few-Shot VLM 능력을 보여주었습니다.
    -   **논문:** [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
-   **A Generalist Agent (Gato) (Google DeepMind, 2022)**
    -   **영향:** 단일 Transformer 모델로 이미지, 텍스트, 행동 등 다양한 종류의 데이터를 처리하는 "범용 에이전트"의 개념을 제시했습니다.
    -   **논문:** [https://arxiv.org/abs/2205.06175](https://arxiv.org/abs/2205.06175)
-   **RT-2: Vision-Language-Action Models (Google DeepMind, 2023)**
    -   **영향:** 웹 데이터로 사전 학습된 VLM이 "think and act" 능력을 로봇 제어에 전이할 수 있음을 보여준 선구적인 연구입니다. 로봇의 행동을 텍스트 토큰으로 처리하는 아이디어를 제공합니다.
    -   **논문:** [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
-   **Octo: An Open-Source Generalist Robot Transformer (Google DeepMind & UC Berkeley, 2024)**
    -   **영향:** 다양한 로봇과 Task에 대한 데이터를 표준화된 Transformer 아키텍처로 학습시키는 범용 로봇 에이전트의 최신 사례입니다. 본 프로젝트의 모델 아키텍처 및 데이터 처리 방식에 참고가 됩니다.
    -   **프로젝트:** [https://octo-models.github.io/](https://octo-models.github.io/)

#### 1.3.3. 추론 및 행동 생성 (`<think>` & `<action>`)
-   **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Google, 2022)**
    -   **영향:** 복잡한 문제에 대해 LLM이 중간 추론 단계를 생성하도록 유도하면 최종 결과의 정확성이 향상된다는 것을 보여주었습니다. `<think>` 토큰은 이 **"생각의 사슬(Chain-of-Thought)"**을 모델이 스스로 생성하고 학습하도록 구조화하는 역할을 합니다.
    -   **논문:** [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
-   **SayCan: Learning Language-Conditioned Affordances in a Robotic Manipulator (Google, 2022)**
    -   **영향:** 언어 명령을 기반으로 로봇이 수행할 수 있는 행동(affordances)을 학습하고, 이를 통해 복잡한 지시를 따르는 방법을 제시했습니다.
    -   **논문:** [https://arxiv.org/abs/2204.01691](https://arxiv.org/abs/2204.01691)
-   **Pi0: A Generalist World Model for Autonomous Driving (Wayve, 2024)**
    -   **영향:** `<think>`와 `<action>` 토큰을 명시적으로 사용하여 모델의 **내부 추론 과정(internal reasoning)**을 감독하고, **설명 가능한(interpretable)** 행동을 생성하는 핵심 아이디어를 제공합니다. 본 프로젝트의 가장 직접적인 영감입니다.
    -   **논문:** [https://arxiv.org/abs/2406.13544](https://arxiv.org/abs/2406.13544)

#### 1.3.4. 강화학습 (RL) 및 정책 최적화 (Reinforcement Learning & Policy Optimization)
-   **Playing Atari with Deep Reinforcement Learning (DeepMind, 2013) - DQN**
    -   **영향:** 딥러닝과 강화학습을 결합하여 Atari 게임에서 인간 수준의 성능을 달성하며 딥 강화학습의 시대를 열었습니다.
    -   **논문:** [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)
-   **Asynchronous Methods for Deep Reinforcement Learning (DeepMind, 2016) - A3C**
    -   **영향:** Actor-Critic 방법론을 비동기적으로 구현하여 학습 효율성을 크게 높였습니다.
    -   **논문:** [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)
-   **Proximal Policy Optimization Algorithms (OpenAI, 2017) - PPO**
    -   **영향:** 구현이 비교적 간단하면서도 안정적이고 좋은 성능을 보이는 정책 경사(Policy Gradient) 방법론으로, 현재 가장 널리 사용되는 RL 알고리즘 중 하나입니다. 본 프로젝트의 핵심 RL 알고리즘입니다.
    -   **논문:** [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
-   **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (UC Berkeley, 2018) - SAC**
    -   **영향:** 엔트로피 정규화를 통해 탐험(exploration)을 장려하고, 샘플 효율성이 높은 Off-Policy RL 알고리즘입니다.
    -   **논문:** [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)
-   **DeepSeek-R: Reasoning via Planning (DeepSeek-AI, 2024)**
    -   **영향:** RL을 사용하여 LLM의 **추론 경로 자체를 보상(reward)으로 최적화**하는 아이디어를 제시합니다. 본 프로젝트의 2단계(RL)에서 `<think>` 과정을 강화하는 핵심적인 접근법입니다.
    -   **논문:** [https://arxiv.org/abs/2405.06328](https://arxiv.org/abs/2405.06328)
-   **Training language models to follow instructions with human feedback (InstructGPT, OpenAI, 2022)**
    -   **영향:** RLHF(Reinforcement Learning from Human Feedback)의 기본 방법론을 제공합니다. 본 프로젝트에서는 "Human Feedback"을 시뮬레이션의 성공/실패 여부인 **"Environment Feedback"**으로 대체하여 적용합니다 (RLAIF).
    -   **논문:** [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

#### 1.3.5. 모방 학습 및 행동 복제 (Imitation Learning & Behavioral Cloning)
-   **A Survey of Imitation Learning (CMU, 2018)**
    -   **영향:** 모방 학습의 다양한 방법론과 응용 분야를 포괄적으로 다룬 서베이 논문입니다.
    -   **논문:** [https://arxiv.org/abs/1808.06701](https://arxiv.org/abs/1808.06701)
-   **Decision Transformer: Reinforcement Learning via Sequence Modeling (Berkeley, 2021)**
    -   **영향:** 강화학습 문제를 시퀀스 모델링 문제로 재해석하여, Transformer가 과거의 상태, 행동, 보상 시퀀스를 기반으로 미래 행동을 예측하도록 학습합니다.
    -   **논문:** [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)

#### 1.3.6. 효율적 파인튜닝 및 프레임워크 (Efficient Fine-Tuning & Frameworks)
-   **LoRA: Low-Rank Adaptation of Large Language Models (Microsoft, 2021)**
    -   **영향:** 대규모 모델의 파인튜닝 시 학습 가능한 파라미터 수를 크게 줄여 메모리 효율성을 높이는 기법입니다.
    -   **논문:** [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
-   **QLoRA: Efficient Finetuning of Quantized LLMs (University of Washington, 2023)**
    -   **영향:** 4비트 양자화된 모델에 LoRA를 적용하여 메모리 사용량을 극적으로 줄이면서도 성능을 유지하는 기법입니다. Colab과 같은 제한된 환경에서 거대 모델을 파인튜닝하기 위한 필수적인 기술입니다.
    -   **논문:** [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
-   **LeRobot: A Framework for Real-World Transfer of Imitation and Reinforcement Learning (Hugging Face, 2024)**
    -   **영향:** 시뮬레이션 및 실제 로봇 환경에서 모방 학습과 RL을 쉽게 적용할 수 있도록 데이터셋, 사전 학습된 모델, 유틸리티를 제공하는 프레임워크입니다. 데이터 처리 및 환경 연동에 참고할 수 있습니다.
    -   **논문:** [https://arxiv.org/abs/2406.01844](https://arxiv.org/abs/2406.01844)


---

## 2. 1단계: 지도 파인튜닝 (SFT)

*이 단계의 목표는 모델이 기본적인 지시-추론-행동의 흐름을 배우도록 하는 것입니다. (이전 계획과 대부분 동일)*

### 2.1. 데이터 생성 (PyBullet)
-   **데이터 형식:** 이전 계획과 동일한 (Vision, Instruction, Think, Action) 튜플을 사용한다.
    ```json
    {
      "image_path": "...",
      "instruction": "...",
      "output": "<think>...</think><action>...</action>"
    }
    ```

### 2.2. 모델 학습 (SFT)
-   **기법:** QLoRA
-   **목표:** 주어진 `image`와 `instruction`에 대해 전문가의 `output`을 최대한 유사하게 모방하도록 학습한다.

---

## 3. 2단계: 강화학습 (RL)

*이 단계의 목표는 SFT 모델을 개선하여, 더 높은 Task 성공률을 보이는 추론 및 행동을 생성하도록 정책(Policy)을 직접 최적화하는 것입니다.*

### 3.1. RL 방법론: PPO on Reasoning
-   **핵심 아이디어:** 모델이 생성한 `<think>` 내용(추론 과정)이 최종적인 Task 성공에 얼마나 기여했는지를 평가하고, 성공 확률을 높이는 방향으로 `<think>` 생성을 강화한다.
-   **알고리즘:** **PPO (Proximal Policy Optimization)**

### 3.2. RL 파이프라인
1.  **Rollout (경험 수집):**
    -   현재 정책 모델(SFT로 초기화된 Gemma 3n)에 `image`와 `instruction`을 입력한다.
    -   모델은 `<think>... </think>`와 `<action>... </action>`을 **생성(Generate)**한다.
    -   생성된 `<action>`을 PyBullet 시뮬레이션에서 **실행**한다.
2.  **Reward 계산:**
    -   **Task 성공 시:** +1.0의 높은 보상을 부여한다.
    -   **Task 실패 시:** -1.0의 패널티를 부여한다.
    -   **(선택적) 추가 보상:**
        -   **효율성 보상:** 더 짧은 `<think>` 과정이나 더 적은 `action` 단계로 성공 시 추가 보상.
        -   **안전성 패널티:** 로봇팔이 충돌하거나 불안정한 움직임을 보일 때 추가 패널티.
3.  **PPO 업데이트:**
    -   수집된 (State, Action, Reward) 데이터를 사용하여 PPO 알고리즘으로 정책 모델(LoRA 가중치)을 업데이트한다.
    -   **State:** `<vision>[Image Tokens] User: {instruction}`
    -   **Action:** `<think>... </think><action>... </action>` (모델이 생성한 전체 텍스트)
    -   이 과정을 통해 "좋은 보상을 받는 (State, Action) 쌍의 확률은 높이고, 나쁜 보상을 받는 쌍의 확률은 낮추도록" 모델을 업데이트한다.

### 3.3. 구현을 위한 라이브러리
-   **RL 라이브러리:** `trl` (Transformers Reinforcement Learning) by Hugging Face
    -   `trl`은 SFT 모델을 PPO로 쉽게 튜닝할 수 있도록 `PPOTrainer`와 같은 유용한 도구를 제공한다.
-   **시뮬레이션 연동:** PyBullet 환경과 `trl`의 PPO 루프를 연동하는 커스텀 스크립트 작성이 필요하다.

---

## 4. 통합 개발 및 평가 계획

### 4.1. 모델 및 환경
-   **기반 모델:** Gemma 3n (Tokenizer 및 Embedding 확장)
-   **SFT 모델:** 1단계에서 QLoRA로 학습된 모델
-   **RL 정책 모델:** SFT 모델로 가중치가 초기화된 LoRA 모델
-   **시뮬레이션:** PyBullet

### 4.2. 평가
-   **핵심 지표:** **Task 성공률** (SFT 모델 vs RL 모델)
-   **분석:** RL 학습 후, 실패했던 Task를 성공하는 사례 분석. `<think>` 내용이 어떻게 더 정교하고 효율적으로 변했는지 정성적으로 비교 분석.

### 4.3. 예상 일정 (12주 계획)
-   **1-3주차:** SFT 단계 (데이터 생성 및 모델 학습)
-   **4-5주차:** RL 환경 구축 (`trl`과 PyBullet 연동, Reward 함수 설계)
-   **6-9주차:** RL 학습 및 하이퍼파라미터 튜닝 (PPO 학습은 SFT보다 불안정하고 많은 실험이 필요함)
-   **10-11주차:** SFT 모델과 최종 RL 모델의 성능 비교 평가
-   **12주차:** 프로젝트 결과 정리, 보고서 및 데모 영상 제작