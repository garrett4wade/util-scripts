#!/usr/bin/env python3
"""
Nightly Experiment Runner

This script automates the process of:
1. Pulling the latest code from the main branch
2. Running an experiment with a date+commit ID in the trial name
3. Automatically killing the experiment after 24 hours
4. Cleaning up all processes when manually interrupted
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import psutil

from realhf.base import logging

logger = logging.getLogger("nightly runner", "system")

# Configuration Constants
# REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DURATION = 12 * 60 * 60  # pull code every 12 hours


def get_experiment_cmd(trial_name):
    EXPERIMENT_CMD = [
        "python3",
        "-m",
        "realhf.apps.quickstart",
        "ppo-math",
        "mode=slurm",
        "experiment_name=fw-Nightly",
        f"trial_name={trial_name}",
        "wandb.mode=online",
        "exp_ctrl.total_train_epochs=5",
        "exp_ctrl.ckpt_freq_secs=3600",
        "group_size=16",
        "actor.type._class=qwen2",
        "actor.path=/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        "actor.bf16=False",
        "critic.type._class=qwen2",
        "critic.type.is_critic=True",
        "critic.init_critic_from_actor=True",
        "critic.path=/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        "ref.type._class=qwen2",
        "ref.path=/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        "ref.bf16=False",
        "rew.type._class=qwen2",
        "rew.type.is_critic=True",
        "rew.init_critic_from_actor=True",
        "rew.path=/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        "dataset.path=/storage/openpsi/users/xushusheng.xss/training_data/boba_106k_0319.jsonl",
        "dataset.max_prompt_len=1024",
        "dataset.train_bs_n_seqs=512",
        "ppo.gen.max_new_tokens=27648",
        "ppo.gen.min_new_tokens=0",
        "ppo.disable_value=True",
        "ppo.gen.top_p=1",
        "ppo.gen.top_k=1000000",
        "ppo.ppo_n_minibatches=4",
        "ppo.gen.temperature=1.0",
        "ppo.kl_ctl=0.0",
        "ppo.value_eps_clip=0.2",
        "ppo.reward_output_scaling=5",
        "ppo.reward_output_bias=0.0",
        "ppo.adv_norm=True",
        "ppo.value_norm=True",
        "ppo.discount=1.0",
        "actor.optimizer.lr=2e-5",
        "actor.optimizer.lr_scheduler_type=constant",
        "actor.optimizer.eps=1e-5",
        "actor.optimizer.warmup_steps_proportion=0.001",
        "actor.sglang.triton_attention_num_kv_splits=16",
        "actor.sglang.mem_fraction_static=0.7",
        "actor.vllm.max_seq_len_to_capture=32768",
        "ref_inf.mb_spec.max_tokens_per_mb=30720",
        "actor_train.mb_spec.max_tokens_per_mb=30720",
        "actor.optimizer.hysteresis=2",
        "cache_clear_freq=1",
        "n_nodes=16",
        "allocation_mode=sglang.d64m1p1+d64p1m1",
        "n_gpus_per_node=8",
        "recover_mode=auto",
        "recover_retries=10",
        "torch_cache_mysophobia=True",
    ]
    # EXPERIMENT_CMD = ["echo", trial_name]
    return EXPERIMENT_CMD


def terminate_process_and_children(pid: int, s=None):
    if s is None:
        s = signal.SIGKILL
    if isinstance(s, str):
        s = getattr(signal, s)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid)
        parent.send_signal(s)
    except psutil.NoSuchProcess:
        pass


class ExperimentRunner:
    def __init__(self):
        self.running = True
        self.tracked_pids = set()

    def _get_current_commit_id(self):
        """Get the short commit hash of the current HEAD."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                # cwd=REPO_DIR,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def _pull_latest_changes(self):
        """Pull the latest changes from the main branch."""
        try:
            subprocess.run(
                ["git", "pull", "origin", "main"],
                # cwd=REPO_DIR,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _generate_trial_name(self):
        """Generate a trial name with date and commit ID."""
        today = datetime.now().strftime("%Y%m%d")
        commit_id = self._get_current_commit_id()
        return f"{today}@1.5b-16node-{commit_id}"

    def _run_experiment(self):
        """Run the experiment with proper environment and timeout."""
        trial_name = self._generate_trial_name()
        logger.info(f"Starting experiment with trial name: {trial_name}")

        # Prepare environment variables
        env = os.environ.copy()
        env.update(
            {
                "CLUSTER_SPEC_PATH": "/storage/realhf/examples/cluster_config_etcd.json",
                "REAL_ETCD_ADDR": "etcd-client.openpsi-etcd.svc.sigma-su18-01.hn01.su18-hn.local:2379",
                "REAL_GPU_MEMORY_KILL_THRESHOLD": "1",
                "WANDB_API_KEY": "local-5dd08fc1894114d0bea728566d5c35c5b31ee608",
                "WANDB_BASE_URL": "http://8.150.1.98:8080",
            }
        )

        # Start the experiment process
        process = subprocess.Popen(
            get_experiment_cmd(trial_name),
            # cwd=REPO_DIR,
            env=env,
            start_new_session=True,
        )
        self.tracked_pids.add(process.pid)

        # Wait for experiment duration or until interrupted
        start_time = time.time()
        while self.running and (time.time() - start_time) < EXPERIMENT_DURATION:
            time.sleep(1)

            if process.poll() is not None:
                logger.critical("Experiment process terminated unexpectedly")
                return

        # Clean up if still running
        if process.poll() is None:
            logger.info("24-hour duration reached, terminating experiment...")
            terminate_process_and_children(process.pid)

    def run(self):
        """Main execution loop that runs indefinitely."""
        while self.running:
            # Pull latest changes
            if not self._pull_latest_changes():
                logger.warning(
                    "Failed to pull latest changes, retrying in 60 seconds..."
                )
                time.sleep(60)
                continue

            # Run experiment
            self._run_experiment()

            # Short delay before restarting (if not interrupted)
            if self.running:
                time.sleep(10)


if __name__ == "__main__":
    runner = ExperimentRunner()
    try:
        runner.run()
    finally:
        terminate_process_and_children(os.getpid())
