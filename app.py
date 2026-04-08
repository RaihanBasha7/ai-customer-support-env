import gradio as gr
import requests
from env.environment import CustomerSupportEnv

env = CustomerSupportEnv()
env.reset()

# Store session state
state_data = {}

# -------------------------
# RESET FUNCTION
# -------------------------
def reset_env():
    global env
    env = CustomerSupportEnv()
    response = env.reset()

    return (
        response["customer_message"],  # 1
        "",                            # 2
        "Reward: 0",                   # 3
        "Final Score: 0",              # 4
        "Logs will appear here"        # 5
    )

# -------------------------
# STEP FUNCTION
# -------------------------
def take_action(action_type, content):
    global env

    try:
        if env.current_task is None:
            env.reset()

        state, reward, done, info = env.step({
            "action_type": action_type,
            "content": content
        })

        conversation = "\n".join(state["conversation"])
        final_score = info.get("final_score", "")
        logs = "\n".join(f"• {log}" for log in info.get("grading_logs", []))
        reason = info.get("reason", "")

        return (
            state.get("customer_message", ""),   # 1
            conversation,                        # 2
            f"Reward: {reward}",                 # 3
            f"Final Score: {final_score if final_score else 'Not finished'}",       # 4
            f"""
            📌 Reason: {reason}

            🧠 Evaluation:
            {logs if logs else "No evaluation yet"}
            """# 5
        )

    except Exception as e:
        return (
            "Error",
            "",
            "Error",
            "Error",
            str(e)
        )
    
def auto_solve():
    global env

    try:
        if env.current_task is None:
            state = env.reset()
        else:
            state = env.reset()

        steps_output = []

        # -------- STEP 1: CLASSIFY --------
        message = state["customer_message"].lower()

        if "refund" in message:
            action = {"action_type": "classify", "content": "refund"}
        elif "angry" in message or "not happy" in message:
            action = {"action_type": "classify", "content": "angry"}
        else:
            action = {"action_type": "classify", "content": "complex"}

        state, reward, done, info = env.step(action)
        steps_output.append(f"Step 1 → classify → {reward}")

        # -------- STEP 2: REPLY / ESCALATE --------
        task_type = env.current_task["type"]

        if task_type in ["complex", "edge_case"]:
            action = {"action_type": "escalate", "content": ""}
        else:
            if task_type == "refund":
                action = {"action_type": "reply", "content": "Your refund has been initiated and will be processed within 3-5 business days."}
            elif task_type == "angry":
                action = {"action_type": "reply", "content": "We sincerely apologize for the inconvenience. We understand your frustration and are here to help."}
            else:
                action = {"action_type": "reply", "content": "issue resolved"}

        state, reward, done, info = env.step(action)
        steps_output.append(f"Step 2 → {action['action_type']} → {reward}")

        # -------- STEP 3: CLOSE (if not done) --------
        if not done:
            action = {"action_type": "close", "content": ""}
            state, reward, done, info = env.step(action)
            steps_output.append(f"Step 3 → close → {reward}")

        conversation = "\n".join(state["conversation"])
        final_score = info.get("final_score", "")
        logs = "\n".join(info.get("grading_logs", []))
        reason = info.get("reason", "")

        return (
            state["customer_message"],
            conversation,
            f"Auto Agent Completed ✅",
            f"Final Score: {final_score} / 1.0 ⭐",
            f"📌 {reason}\n\n🧠 Logs:\n{logs}"
        )

    except Exception as e:
        return ("Error", "", "Error", "Error", str(e))

# -------------------------
# UI LAYOUT
# -------------------------

with gr.Blocks(title="AI Customer Support Simulator 🚀") as demo:

    gr.Markdown("## AI Customer Support Simulator")
    gr.Markdown("Deterministic AI Evaluation • Real-time Scoring")
    gr.Markdown("⚡ Try Auto Solve to see AI complete the task instantly")

    # -------------------------
    # CUSTOMER SECTION
    # -------------------------
    with gr.Group(elem_classes="card"):
        customer_msg = gr.Textbox(label="📩 Customer Message", interactive=False)
        conversation_box = gr.Textbox(label="💬 Conversation History", lines=10)

    # -------------------------
    # ACTION SECTION
    # -------------------------
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 🎮 Agent Controls")

        with gr.Row():
            action_type = gr.Dropdown(
                ["classify", "ask", "reply", "escalate", "close"],
                value="classify",
                label="Action Type"
            )
            content = gr.Textbox(label="Content")

        with gr.Row():
            submit_btn = gr.Button("🚀 Submit")
            reset_btn = gr.Button("🔄 Reset")
            auto_btn = gr.Button("🤖 Auto Solve")

    # -------------------------
    # RESULTS SECTION
    # -------------------------
    with gr.Group(elem_classes="card"):
        gr.Markdown("### 📊 Evaluation Results")

        reward_box = gr.Textbox(label="⚡ Reward")
        score_box = gr.Textbox(label="🏆 Final Score")
        logs_box = gr.Textbox(label="🧠 Grader Logs", lines=8)

    gr.Markdown("---")
    gr.Markdown("⚡ Built for Meta x Scaler PyTorch Hackathon • By Team 🚀")

    # BUTTON CONNECTIONS
    submit_btn.click(
        take_action,
        inputs=[action_type, content],
        outputs=[customer_msg, conversation_box, reward_box, score_box, logs_box]
    )

    reset_btn.click(
        reset_env,
        outputs=[customer_msg, conversation_box, reward_box, score_box, logs_box]
    )

    auto_btn.click(
        auto_solve,
        outputs=[customer_msg, conversation_box, reward_box, score_box, logs_box]
    )

# Run
demo.launch(theme=gr.themes.Soft())