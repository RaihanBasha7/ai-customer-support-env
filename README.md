# OpenEnv — AI Customer Support Training Environment

> Scaler × Meta AI Hackathon — Submission

An **OpenAI Gym-style reinforcement learning environment** where an AI agent acts as a customer support representative, resolving tickets step-by-step through structured actions, observations, and a reward system.

---

## Project Structure

```
openenv/
├── api/
│   └── main.py               # FastAPI server (/reset, /step, /state, /tasks)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── env/
│   ├── __init__.py
│   ├── environment.py        # Core CustomerSupportEnv class
│   ├── grader.py             # Deterministic 0.0–1.0 scorer
│   ├── models.py             # Pydantic models
│   ├── reward_config.py      # All reward values
│   └── tasks.py              # 4 tasks (easy → edge case)
├── tests/
│   └── test_env.py
├── inference.py              # Required hackathon runner
├── openenv.yaml              # Environment config
├── requirements.txt
└── README.md
```

---

## Quickstart

### Install
```bash
pip install -r requirements.txt
```

### Run inference (all 4 tasks)
```bash
python inference.py
```

### Run a specific task
```bash
python inference.py --task easy_refund
```

### Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### Run tests
```bash
pytest tests/ -v
```

### Docker
```bash
cd docker && docker-compose up --build
```

---

## Environment Design

### Actions
| Action | Description |
|---|---|
| `classify` | Label the ticket type (refund / angry / complex / edge_case) |
| `ask` | Ask a clarifying question (must contain `?`) |
| `reply` | Respond to the customer with a resolution |
| `escalate` | Escalate to a senior team (valid for complex / edge_case only) |
| `close` | Close the ticket (valid only when status is `resolved`) |

### Tasks
| ID | Difficulty | Max Turns | Description |
|---|---|---|---|
| `easy_refund` | Easy | 4 | Simple refund request |
| `medium_angry` | Medium | 5 | Angry customer complaint |
| `hard_complex` | Hard | 8 | Delayed order + refund + compensation |
| `edge_case` | Edge Case | 6 | Unauthorized charge / potential fraud |

### Reward System
| Event | Reward |
|---|---|
| Correct classify | +0.30 |
| Wrong classify | −0.20 |
| Valid ask | +0.10 |
| Correct reply | +0.50 |
| Wrong reply | −0.30 |
| Correct escalate | +0.30 |
| Wrong escalate | −0.40 |
| Correct close | +0.20 |
| Wrong close | −0.50 |
| Step penalty | −0.02 |
| Efficiency bonus | +0.20 |

### Grader (0.0 – 1.0)
The deterministic grader produces a normalised score per episode:

| Grade | Score |
|---|---|
| Excellent | ≥ 0.85 |
| Good | ≥ 0.60 |
| Partial | ≥ 0.30 |
| Fail | < 0.30 |

---

## API Reference

### `POST /reset`
Start a new episode.
```json
{ "task_id": "easy_refund", "random_task": false }
```

### `POST /step`
Take one action.
```json
{ "action_type": "classify", "content": "refund" }
```

### `GET /state`
Get the current environment state.

### `GET /tasks`
List all available tasks.

### `GET /health`
Liveness probe — returns `{ "status": "ok" }`.

---

## Team

| Name | Role |
|---|---|
| Shaik Raihan Basha | Team Lead — Tasks, Grader, Inference, Integration |
| Shaik Inzamam | Environment Logic — env.py, step(), reset(), state() |
| Shaik Suhail | UI & API — FastAPI, Hugging Face Spaces, Docker |