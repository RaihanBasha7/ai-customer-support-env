# 🚀 AI Customer Support OpenEnv Environment

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

An OpenEnv-compliant simulation environment where an LLM acts as a customer support agent, resolving issues step-by-step using actions, rewards, and task-based evaluation.

---

## 🎯 Objective

Build a realistic environment for training and evaluating AI agents in customer support scenarios such as:

- Refund requests
- Angry customers
- Complex multi-step issues
- Fraud and edge cases

Agents are evaluated on **correctness**, **efficiency**, and **decision-making quality**.

---

## 🏗️ Project Structure
AI-CUSTOMER-SUPPORT-ENV/
│
├── api/                  # FastAPI endpoints (/reset, /step)
├── env/                  # Core environment logic
│   ├── environment.py
│   ├── tasks.py
│   ├── reward_config.py
│   ├── grader.py
│   └── models.py
│
├── tests/                # Test cases for the environment
├── docker/               # Docker setup
│   └── Dockerfile
│
├── inference.py          # LLM interaction loop
├── openenv.yaml          # OpenEnv spec (defines obs/action space & metadata)
├── requirements.txt
└── README.md
---

## ⚙️ Core Features

### 🔹 Environment Design
Simulates real-world customer support workflows and maintains full state across turns — including conversation history, ticket status, and issue classification. Supports four task difficulty levels:

| Level | Example Scenario |
|-------|-----------------|
| Easy | Refund request |
| Medium | Angry customer |
| Hard | Complex multi-step issue |
| Edge | Fraud / ambiguous case |

---

### 🔹 Action Space
At each step, the agent selects one of the following actions:

| Action | Description |
|--------|-------------|
| `classify` | Label the issue type |
| `ask` | Request more info from the customer |
| `reply` | Respond to the customer |
| `escalate` | Hand off to a human agent |
| `close` | Mark the ticket as resolved |

---

### 🔹 Reward System
Defined in `reward_config.py`. The agent is rewarded for correct, efficient resolutions and penalized for poor decisions.

**Rewards:**
- Correct classification
- Proper resolution
- Efficient decision-making

**Penalties:**
- Invalid actions
- Premature closing
- Incorrect responses
- Excess steps

---

### 🔹 Step Engine
Handles the core loop per turn:
- Action validation
- Reward assignment
- State transitions
- Tracks turn count, conversation history, and last action

---

### 🔹 Explainability
Every step logs a structured trace:
- Action taken
- Reward received
- Reason for the reward

This makes agent behavior transparent and easy to debug during evaluation.

---

### 🔹 Grader System
Produces a deterministic final score between **0.0 and 1.0** at episode end, evaluating:
- Task completion
- Correct action usage
- Overall efficiency

---

## 🔁 How It Works

**Observation** (input to agent):
```json
{
  "customer_message": "I was charged twice for my order!",
  "conversation": [],
  "status": "open"
}
```

**Action** (agent output):
```json
{
  "action_type": "classify",
  "content": "billing_issue"
}
```

The environment processes the action, updates state, returns the next observation, reward, and explanation.

---

## 🧪 Running Locally

**Run tests:**
```bash
python -m tests.test_env
```

**Run inference loop:**
```bash
python inference.py
```

---

## 🐳 Docker
```bash
cd docker
docker build -t customer-support-env .
docker run customer-support-env
```

---

## 🎯 Hackathon Evaluation Alignment

| Criterion | Status |
|-----------|--------|
| Real-world utility | ✅ Customer support simulation |
| Task & grader quality | ✅ Deterministic 0–1 scoring |
| Clean environment design | ✅ OpenEnv-compliant |
| Code structure & modularity | ✅ Separated concerns across modules |
| Creativity | ✅ Reward shaping + step explainability |

---

## 👥 Team

| Role | Name | Responsibilities |
|------|------|-----------------|
| Lead & Integration | Shaik Raihan Basha | Environment design, reward system, grader, inference loop |
| UI/API & Deployment | Shaik Suhail | FastAPI endpoints, Hugging Face deployment, Docker setup |
| Environment Logic | Shaik Inzamam | Step engine, state transitions, action handling |

---

## 🚀 Goal

Build a realistic, testable, and explainable environment for evaluating AI agents — targeting **Top 10% selection**.