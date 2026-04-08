"""
tests/test_env.py — Day 2 test suite for OpenEnv step engine.

Covers every action handler, all status transitions,
reward correctness, and the grader.

HOW TO RUN
----------
Correct way (shows all 43 results):
    python -m pytest tests/test_env.py -v

Also works directly on Windows:
    python tests/test_env.py
    python tests\test_env.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env import CustomerSupportEnv, Action
from env.grader import calculate_reward, grade_episode


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def easy():
    e = CustomerSupportEnv(task_id="easy_refund")
    e.reset()
    return e

@pytest.fixture
def medium():
    e = CustomerSupportEnv(task_id="medium_angry")
    e.reset()
    return e

@pytest.fixture
def hard():
    e = CustomerSupportEnv(task_id="hard_complex")
    e.reset()
    return e

@pytest.fixture
def edge():
    e = CustomerSupportEnv(task_id="edge_case")
    e.reset()
    return e


# ══════════════════════════════════════════════════════════════════════
#  reset()
# ══════════════════════════════════════════════════════════════════════
class TestReset:
    def test_returns_observation(self):
        env = CustomerSupportEnv(task_id="easy_refund")
        obs = env.reset()
        assert obs.customer_message == "I want a refund for my product"
        assert obs.turn == 0
        assert obs.status == "open"
        assert obs.max_turns == 4

    def test_history_seeded_with_customer_message(self):
        env = CustomerSupportEnv(task_id="easy_refund")
        obs = env.reset()
        assert len(obs.history) == 1
        assert "CUSTOMER" in obs.history[0]

    def test_state_zeroed_on_reset(self):
        env = CustomerSupportEnv(task_id="easy_refund")
        env.reset()
        env.step(Action(action_type="classify", content="refund"))
        env.reset()                          # second reset
        s = env.state
        assert s["turn"] == 0
        assert s["total_reward"] == 0.0
        assert s["classified"] is False
        assert s["status"] == "open"
        assert len(s["actions_log"]) == 0

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            CustomerSupportEnv(task_id="nonexistent").reset()


# ══════════════════════════════════════════════════════════════════════
#  state property
# ══════════════════════════════════════════════════════════════════════
class TestState:
    def test_required_keys_present(self, easy):
        s = easy.state
        for key in ["task_id", "task_type", "difficulty", "customer_msg",
                    "turn", "max_turns", "status", "classified",
                    "history", "actions_log", "total_reward"]:
            assert key in s, f"Missing state key: '{key}'"

    def test_initial_status_open(self, easy):
        assert easy.state["status"] == "open"

    def test_task_id_correct(self, easy):
        assert easy.state["task_id"] == "easy_refund"

    def test_state_returns_empty_dict_before_reset(self):
        env = CustomerSupportEnv(task_id="easy_refund")
        assert env.state == {}


# ══════════════════════════════════════════════════════════════════════
#  Action: classify
# ══════════════════════════════════════════════════════════════════════
class TestClassify:
    def test_correct_classify_positive_reward(self, easy):
        _, reward, _, info = easy.step(
            Action(action_type="classify", content="refund"))
        assert reward > 0
        assert "correct" in info["classify"]

    def test_correct_classify_sets_classified_and_pending(self, easy):
        easy.step(Action(action_type="classify", content="refund"))
        assert easy.state["classified"] is True
        assert easy.state["status"] == "pending"

    def test_wrong_classify_negative_reward(self, easy):
        _, reward, _, info = easy.step(
            Action(action_type="classify", content="angry"))
        assert reward < 0
        assert "wrong" in info["classify"]
        assert easy.state["classified"] is False
        assert easy.state["status"] == "open"   # status unchanged

    def test_repeated_classify_penalised(self, easy):
        easy.step(Action(action_type="classify", content="refund"))
        _, reward, _, info = easy.step(
            Action(action_type="classify", content="refund"))
        assert reward < 0
        assert "repeated" in info["classify"]

    def test_classify_case_insensitive(self, easy):
        _, reward, _, _ = easy.step(
            Action(action_type="classify", content="REFUND"))
        assert reward > 0


# ══════════════════════════════════════════════════════════════════════
#  Action: ask
# ══════════════════════════════════════════════════════════════════════
class TestAsk:
    def test_valid_question_positive_reward(self, hard):
        _, reward, _, info = hard.step(
            Action(action_type="ask",
                   content="Could you provide your order number please?"))
        assert reward > 0
        assert info["ask"] == "valid"

    def test_no_question_mark_invalid(self, hard):
        _, reward, _, info = hard.step(
            Action(action_type="ask", content="Tell me your order number"))
        assert reward < 0

    def test_too_short_invalid(self, hard):
        _, reward, _, _ = hard.step(
            Action(action_type="ask", content="Why?"))
        assert reward < 0

    def test_ask_sets_status_pending(self, hard):
        hard.step(Action(action_type="ask",
                         content="Could you provide your order number please?"))
        assert hard.state["status"] == "pending"


# ══════════════════════════════════════════════════════════════════════
#  Action: reply
# ══════════════════════════════════════════════════════════════════════
class TestReply:
    def test_correct_reply_positive_reward(self, easy):
        _, reward, _, info = easy.step(
            Action(action_type="reply",
                   content="We will process your refund immediately."))
        assert reward > 0
        assert "correct" in info["reply"]

    def test_correct_reply_sets_resolved(self, easy):
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        assert easy.state["status"] == "resolved"

    def test_wrong_keywords_negative_reward(self, easy):
        _, reward, _, info = easy.step(
            Action(action_type="reply",
                   content="Thank you for contacting us today, we appreciate it."))
        assert reward < 0
        assert "no keyword match" in info["reply"]

    def test_too_short_reply_penalised(self, easy):
        _, reward, _, _ = easy.step(
            Action(action_type="reply", content="ok done"))
        assert reward < 0

    def test_angry_correct_keywords(self, medium):
        _, reward, _, _ = medium.step(
            Action(action_type="reply",
                   content="We sincerely apologize for the frustration you experienced."))
        assert reward > 0


# ══════════════════════════════════════════════════════════════════════
#  Action: escalate
# ══════════════════════════════════════════════════════════════════════
class TestEscalate:
    def test_escalate_correct_on_complex(self, hard):
        _, reward, _, info = hard.step(
            Action(action_type="escalate",
                   content="Escalating to senior team."))
        assert reward > 0
        assert "correct" in info["escalate"]
        assert hard.state["status"] == "resolved"

    def test_escalate_correct_on_edge_case(self, edge):
        _, reward, _, info = edge.step(
            Action(action_type="escalate",
                   content="Escalating to senior team."))
        assert reward > 0

    def test_escalate_wrong_on_easy(self, easy):
        _, reward, _, info = easy.step(
            Action(action_type="escalate",
                   content="Escalating unnecessarily."))
        assert reward < 0
        assert "wrong" in info["escalate"]

    def test_escalate_wrong_on_medium(self, medium):
        _, reward, _, _ = medium.step(
            Action(action_type="escalate",
                   content="Escalating unnecessarily."))
        assert reward < 0


# ══════════════════════════════════════════════════════════════════════
#  Action: close
# ══════════════════════════════════════════════════════════════════════
class TestClose:
    def test_close_after_resolve_correct(self, easy):
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        _, reward, done, info = easy.step(
            Action(action_type="close", content="Closing ticket."))
        assert reward > 0
        assert done is True
        assert easy.state["status"] == "closed"
        assert "correct" in info["close"]

    def test_close_without_resolve_penalised(self, easy):
        _, reward, done, info = easy.step(
            Action(action_type="close", content="Closing ticket."))
        assert reward < 0
        assert done is True
        assert "wrong" in info["close"]

    def test_close_ends_episode(self, easy):
        _, _, done, _ = easy.step(
            Action(action_type="close", content="Closing ticket."))
        assert done is True


# ══════════════════════════════════════════════════════════════════════
#  Status transitions
# ══════════════════════════════════════════════════════════════════════
class TestStatusTransitions:
    def test_open_to_pending_via_classify(self, easy):
        assert easy.state["status"] == "open"
        easy.step(Action(action_type="classify", content="refund"))
        assert easy.state["status"] == "pending"

    def test_pending_to_resolved_via_reply(self, easy):
        easy.step(Action(action_type="classify", content="refund"))
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        assert easy.state["status"] == "resolved"

    def test_resolved_to_closed_via_close(self, easy):
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        easy.step(Action(action_type="close", content="Done."))
        assert easy.state["status"] == "closed"

    def test_full_lifecycle_easy(self, easy):
        """open → pending → resolved → closed in order."""
        assert easy.state["status"] == "open"
        easy.step(Action(action_type="classify", content="refund"))
        assert easy.state["status"] == "pending"
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        assert easy.state["status"] == "resolved"
        easy.step(Action(action_type="close", content="Closing."))
        assert easy.state["status"] == "closed"


# ══════════════════════════════════════════════════════════════════════
#  Reward accumulation & step penalty
# ══════════════════════════════════════════════════════════════════════
class TestRewards:
    def test_step_penalty_applied_every_turn(self, easy):
        # Two no-op actions — reward should be negative (only penalty)
        easy.step(Action(action_type="ask", content="ok"))  # invalid ask
        s_before = easy.state["total_reward"]
        easy.step(Action(action_type="ask", content="ok"))  # invalid ask again
        s_after  = easy.state["total_reward"]
        # Both turns had step penalty, so total should be more negative
        assert s_after < s_before

    def test_efficiency_bonus_on_fast_resolve(self):
        """Resolve within 60 % of max_turns → efficiency bonus."""
        env = CustomerSupportEnv(task_id="easy_refund")  # max_turns=4
        env.reset()
        env.step(Action(action_type="classify", content="refund"))
        # Turn 2 — 2/4 = 50 % < 60 % → should earn bonus
        _, _, _, info = env.step(
            Action(action_type="reply",
                   content="We will process your refund immediately."))
        assert info.get("efficiency_bonus") is True

    def test_max_turns_timeout_applies_penalty(self):
        """Exhaust turns without resolving → timeout penalty."""
        env = CustomerSupportEnv(task_id="easy_refund")  # max_turns=4
        env.reset()
        for _ in range(3):
            env.step(Action(action_type="ask",
                            content="Can you clarify your order please?"))
        _, _, done, info = env.step(
            Action(action_type="ask",
                   content="Can you clarify your order please?"))
        assert done is True
        assert info.get("timeout_penalty") is True


# ══════════════════════════════════════════════════════════════════════
#  Grader
# ══════════════════════════════════════════════════════════════════════
class TestGrader:
    def test_calculate_reward_bounds(self, easy):
        easy.step(Action(action_type="classify", content="refund"))
        score = calculate_reward(easy.state, "classify")
        assert 0.0 <= score <= 1.0

    def test_close_resolved_scores_high(self, easy):
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        easy.step(Action(action_type="close", content="Done."))
        score = calculate_reward(easy.state, "close")
        assert score >= 0.7

    def test_close_unresolved_scores_zero(self, easy):
        score = calculate_reward(easy.state, "close")
        assert score == 0.0

    def test_grade_episode_structure(self, easy):
        easy.step(Action(action_type="classify", content="refund"))
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        easy.step(Action(action_type="close", content="Done."))
        result = grade_episode({
            "total_reward": easy.state["total_reward"],
            "turns_used":   easy.state["turn"],
            "max_turns":    easy.state["max_turns"],
            "final_status": easy.state["status"],
            "task_type":    easy.state["task_type"],
        })
        assert "grade" in result
        assert "normalised_score" in result
        assert "passed" in result
        assert "success" in result
        assert "efficiency" in result

    def test_grade_labels(self):
        """Grade boundaries map correctly."""
        perfect = grade_episode({
            "total_reward": 1.2,
            "turns_used": 2,
            "max_turns": 4,
            "final_status": "closed",
            "task_type": "refund",
        })
        assert perfect["grade"] == "excellent"

        fail = grade_episode({
            "total_reward": -1.0,
            "turns_used": 4,
            "max_turns": 4,
            "final_status": "open",
            "task_type": "refund",
        })
        assert fail["grade"] == "fail"


# ══════════════════════════════════════════════════════════════════════
#  Info dict completeness
# ══════════════════════════════════════════════════════════════════════
class TestInfoDict:
    def test_grader_score_always_present(self, easy):
        _, _, _, info = easy.step(
            Action(action_type="classify", content="refund"))
        assert "grader_score" in info

    def test_episode_summary_on_close(self, easy):
        easy.step(Action(action_type="reply",
                         content="We will process your refund immediately."))
        _, _, _, info = easy.step(
            Action(action_type="close", content="Done."))
        ep = info.get("episode", {})
        assert "task_id" in ep
        assert "total_reward" in ep
        assert "success" in ep
        assert "grade" in ep


# ══════════════════════════════════════════════════════════════════════
#  Entry point — works with BOTH commands on Windows:
#    python -m pytest tests/test_env.py -v    <- recommended
#    python tests/test_env.py                 <- also works
#    python tests\test_env.py                <- also works
#
#  NOTE: uses subprocess so pytest does NOT collect this file twice,
#  which caused the INTERNALERROR / SystemExit crash on Windows.
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import subprocess
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "pytest",
             os.path.abspath(__file__), "-v", "--tb=short"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )