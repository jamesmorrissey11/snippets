import os

from langchain.callbacks import wandb_tracing_enabled


# unset the environment variable and use a context manager instead
def run_agent_with_context_manager(agent, text):
    if "LANGCHAIN_WANDB_TRACING" in os.environ:
        del os.environ["LANGCHAIN_WANDB_TRACING"]
    with wandb_tracing_enabled():
        result = agent.run(text)
    return result


def enable_tracing(wandb_project):
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = wandb_project
