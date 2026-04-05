import argparse
from env import CustomerSupportEnv, Action
from env.tasks import TASKS


def policy(obs, state: dict) -> Action:
    """
    Deterministic rule-based policy used for validation and testing.

    Turn 0              → classify  (always first)
    Turn 1 (complex)    → ask       (get clarification before replying)
    Next available turn → reply     (task-specific keywords guaranteed)
    After reply, if escalatable and not yet escalated → escalate
    Once resolved       → close
    """
    turn      = obs.turn
    task_type = state.get("task_type", "refund")
    status    = state.get("status", "open")

    # ── Step 1: Classify ──────────────────────────────────────────────
    if turn == 0:
        return Action(action_type="classify", content=task_type)

    # ── Step 2: Ask (complex/edge_case only) ─────────────────────────
    if turn == 1 and task_type in ("complex", "edge_case"):
        clarifications = {
            "complex":   "Could you provide your order number and "
                         "your preferred resolution method?",
            "edge_case": "Can you confirm the charge date and amount "
                         "shown on your account statement?",
        }
        return Action(action_type="ask",
                      content=clarifications[task_type])

    # ── Step 3: Reply ─────────────────────────────────────────────────
    if status not in ("resolved", "closed"):
        replies = {
            "refund":    "We will process your refund immediately. "
                         "Please allow 3–5 business days for the money back.",
            "angry":     "We sincerely apologize for the frustration you have "
                         "experienced. We will help resolve this right away.",
            "complex":   "We are sorry for the delay. We will issue a full "
                         "refund and apply a compensation credit to your account.",
            "edge_case": "We are investigating this unauthorized charge and "
                         "will verify the order details on your account now.",
        }
        content = replies.get(
            task_type,
            "We will resolve your issue immediately. Thank you for your patience."
        )
        return Action(action_type="reply", content=content)

    # ── Step 4: Escalate (once, if appropriate) ───────────────────────
    already_escalated = any(
        a["action"] == "escalate" for a in state.get("actions_log", [])
    )
    if (task_type in ("complex", "edge_case")
            and status == "resolved"
            and not already_escalated):
        return Action(
            action_type="escalate",
            content="Escalating to senior support team for final review."
        )

    # ── Step 5: Close ────────────────────────────────────────────────
    return Action(action_type="close",
                  content="Issue fully resolved. Closing the ticket.")


# ──────────────────────────────────────────────────────────────────────
#  Episode runner
# ──────────────────────────────────────────────────────────────────────
def run_episode(env: CustomerSupportEnv, verbose: bool = True) -> dict:
    """Run a single episode and return the episode summary dict."""
    obs          = env.reset()
    done         = False
    total_reward = 0.0

    if verbose:
        print("\n" + "═" * 64)
        print(f"  TASK      : {env.state['task_id']}  "
              f"[{env.state['difficulty']}]")
        print(f"  CUSTOMER  : {obs.customer_message}")
        print(f"  MAX TURNS : {obs.max_turns}")
        print("═" * 64)

    while not done:
        action               = policy(obs, env.state)
        obs, reward, done, info = env.step(action)
        total_reward        += reward

        if verbose:
            print(f"\n  Turn {obs.turn:>2}  │  {action.action_type.upper()}")
            print(f"          │  Content : {action.content[:70]}")
            print(f"          │  Reward  : {reward:+.4f}   "
                  f"Total: {total_reward:+.4f}")
            print(f"          │  Status  : {env.state['status']}")
            print(f"          │  Grader  : {info.get('grader_score', '-')}")

            if info.get("efficiency_bonus"):
                print("          │  ★ Efficiency bonus earned!")
            if info.get("timeout_penalty"):
                print("          │  ✗ Timeout penalty applied")
            if info.get("classify"):
                print(f"          │  classify: {info['classify']}")
            if info.get("reply"):
                print(f"          │  reply   : {info['reply']}")
            if info.get("escalate"):
                print(f"          │  escalate: {info['escalate']}")
            if info.get("close"):
                print(f"          │  close   : {info['close']}")
            if info.get("error"):
                print(f"          │  ERROR   : {info['error']}")

    episode = info.get("episode", {})
    grade   = episode.get("grade", {})

    if verbose:
        symbol = "✓" if episode.get("success") else "✗"
        print("\n" + "─" * 64)
        print(f"  {symbol}  {'SUCCESS' if episode.get('success') else 'FAILED'}")
        print(f"     Total reward : {total_reward:+.4f}")
        print(f"     Grade        : {grade.get('grade', '?').upper()}")
        print(f"     Score        : {grade.get('normalised_score', 0):.4f}")
        print(f"     Efficiency   : {grade.get('efficiency', 0):.4f}")
        print(f"     Passed       : {grade.get('passed', False)}")
        print("─" * 64)

    return episode


# ──────────────────────────────────────────────────────────────────────
#  Main entry point (required hackathon format)
# ──────────────────────────────────────────────────────────────────────
def run_inference(task_id: str = None, verbose: bool = True) -> list:
    """
    Required entry point for hackathon validation.

    Parameters
    ----------
    task_id : str or None — specific task, or None to run all 4
    verbose : bool        — print step-by-step output

    Returns
    -------
    list of episode summary dicts
    """
    tasks_to_run = (
        [task_id] if task_id else [t["id"] for t in TASKS]
    )

    results = []
    for tid in tasks_to_run:
        env     = CustomerSupportEnv(task_id=tid)
        summary = run_episode(env, verbose=verbose)
        results.append(summary)

    if verbose and len(results) > 1:
        passed  = sum(1 for r in results
                      if r.get("grade", {}).get("passed"))
        avg_r   = sum(r.get("total_reward", 0) for r in results) / len(results)
        print(f"\n{'═'*64}")
        print("  OVERALL SUMMARY")
        print(f"{'═'*64}")
        print(f"  Tasks run   : {len(results)}")
        print(f"  Passed      : {passed} / {len(results)}")
        print(f"  Avg reward  : {avg_r:+.4f}")
        print(f"{'═'*64}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv — Day 2 inference runner")
    parser.add_argument("--task",  type=str, default=None,
                        help="task_id to run (default: all)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress step-by-step output")
    args = parser.parse_args()
    run_inference(task_id=args.task, verbose=not args.quiet)