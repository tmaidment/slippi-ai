import functools
import os

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import appo
from ray.rllib.algorithms import callbacks as rllib_callbacks
from ray.rllib.examples._old_api_stack.policy import random_policy
from ray.rllib.policy import policy as rllib_policy
from ray.tune import CLIReporter
from ray.tune.result import (
    EPISODE_REWARD_MEAN,
    MEAN_ACCURACY,
    MEAN_LOSS,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
)
from ray.tune.search.hyperopt import HyperOptSearch

from slippi_ai.rllib.callbacks import add_policies, winrates
from slippi_ai.rllib.utils import matchmaking
from slippi_ai.rllib.env import SlippiMultiAgentEnv, SlippiEnvConfig
from slippi_ai.rllib.models import lstm_model

def eval_policy_mapping_fn(*args, **kwargs): ...

def multiagent_env_creator(env_config):
    """Create SlippiMultiAgentEnv from config."""
    config = SlippiEnvConfig(
        dolphin_path=env_config.get("dolphin_path"),
        iso_path=env_config.get("iso_path"),
    )
    return SlippiMultiAgentEnv(config)


class Experiment:

    NUM_ENVS_PER_ENV_RUNNER = 1

    def __init__(self, config: dict | None = None):

        config = config or {}

        self.config = config

    def construct_run_config(self):
        reporter = CLIReporter(
            metric_columns={
                MEAN_ACCURACY: "acc",
                MEAN_LOSS: "loss",
                # TRAINING_ITERATION: "iter",
                TIME_TOTAL_S: "total time (s)",
                TIMESTEPS_TOTAL: "ts",
                EPISODE_REWARD_MEAN: "reward",
            }
        )

        run_config = air.config.RunConfig(
            name=self.config["experiment_name"],
            stop={
                "num_agent_steps_trained": self.config.get(
                    "agent_steps_to_train", 300_000_000
                )
            },
            callbacks=(
                [WandbLoggerCallback(project="Melee-v0")]
                if not self.config.get("debug", False)
                else None
            ),
            failure_config=air.config.FailureConfig(
                max_failures=self.config.get("max_failures", 0),
                fail_fast=self.config.get("fail_fast", False),
            ),
            checkpoint_config=air.config.CheckpointConfig(
                checkpoint_frequency=self.config.get("checkpoint_freq", 50),
                # num_to_keep=self.config.get("num_to_keep", 5),
                checkpoint_at_end=True,
            ),
            progress_reporter=reporter,
            verbose=1,
        )
        return run_config

    def construct_tune_config(self):

        if self.config.get("tune"):
            tune_config = tune.TuneConfig(
                num_samples=self.config.get("num_trials", 20),
                max_concurrent_trials=self.config.get(
                    "max_concurrent_trials", 1
                ),
                metric="env_runners/policy_reward_mean/focal_policy",
                mode="max",
                search_alg=HyperOptSearch(),
                scheduler=tune.schedulers.ASHAScheduler(
                    time_attr="num_agent_steps_trained",
                    max_t=50_000_000,
                    grace_period=5_000_000,
                ),
            )
        else:
            tune_config = tune.TuneConfig(
                num_samples=self.config.get("num_trials", 1),
                max_concurrent_trials=self.config.get(
                    "max_concurrent_trials", 1
                ),
                trial_name_creator=lambda trial: f"{self.config.get('experiment_name')}-{str(trial).split('_')[-1]}",
            )
        return tune_config

    def construct_model_config(self, as_dict=True):

        # Create a dummy environment to get observation and action spaces
        dummy_config = SlippiEnvConfig(
            dolphin_path=self.config.get("dolphin_path"),
            iso_path=self.config.get("iso_path"),
        )
        dummy_env = SlippiMultiAgentEnv(dummy_config)
        
        policy_observation_space = dummy_env.observation_space["p1"]
        policy_action_space = dummy_env.action_space["p1"]
        
        dummy_env.close()

        # Add policy names stored in the ModuleRepository here to evaluate against them
        eval_policies = []

        config = (
            appo.APPOConfig()
            .environment(
                "SlippiMultiAgentEnv",
                env_config={
                    "dolphin_path": self.config.get("dolphin_path"),
                    "iso_path": self.config.get("iso_path"),
                },
            )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .resources(
                num_learner_workers=1,
                num_gpus_per_learner_worker=(
                    1 if not self.config.get("debug", False) else 0
                ),
                num_gpus=(1 if not self.config.get("debug", False) else 0),
                num_cpus_for_local_worker=1,
            )
            .env_runners(
                num_env_runners=(
                    40 if not self.config.get("debug", False) else 1
                ),
                # Must be 1 unless the port configuration is changed
                # in footsies_env.py, which finds the port according
                # to the worker index.
                num_envs_per_env_runner=self.NUM_ENVS_PER_ENV_RUNNER,
            )
            .env_runners(
                rollout_fragment_length=128,
                batch_mode="truncate_episodes",
            )
            .multi_agent(
                policies={
                    "focal_policy": rllib_policy.PolicySpec(
                        config={
                            "model": {
                                "custom_model": lstm_model.LSTMModel,
                                "custom_model_config": {
                                    "lstm_cell_size": 256,
                                    "policy_dense_widths": [256, 256],
                                },
                            },
                            "max_seq_len": 32,
                        },
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "random": rllib_policy.PolicySpec(
                        policy_class=random_policy.RandomPolicy,
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                },
                policy_mapping_fn=matchmaking.Matchmaker(
                    [matchmaking.Matchup("focal_policy", "focal_policy", 1.0)]
                ).policy_mapping_fn,
                policies_to_train=["focal_policy"],
            )
            .evaluation(
                evaluation_num_env_runners=(
                    8 if not self.config.get("debug", False) else 1
                ),
                evaluation_interval=1,
                evaluation_duration="auto",
                evaluation_duration_unit="timesteps",
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "env_config": {
                        "dolphin_path": self.config.get("dolphin_path"),
                        "iso_path": self.config.get("iso_path"),
                        "evaluation": True,
                    },
                    "multiagent": {
                        "policy_mapping_fn": matchmaking.Matchmaker(
                            [
                                matchmaking.Matchup(
                                    "focal_policy",
                                    eval_policy,
                                    1 / (len(eval_policies) + 1),
                                )
                                for eval_policy in eval_policies + ["random"]
                            ]
                        ).policy_mapping_fn,
                    },
                },
            )
            .callbacks(
                rllib_callbacks.make_multi_callbacks(
                    [
                        winrates.Winrates,
                        functools.partial(
                            add_policies.AddPolicies, policies=eval_policies
                        ),
                        # script_metrics.ScriptMetrics,
                    ]
                )
            )
        )

        if self.config.get("tune"):
            config.training(
                lr=tune.choice([1e-3, 3e-4, 5e-5, 1e-5]),
                train_batch_size=1024,
                entropy_coeff=tune.choice(
                    [
                        # 0,
                        0.001,
                        0.005,
                        0.01,
                        0.03,
                        0.05,
                        0.1,
                    ]
                ),
                gamma=0.995,
                vf_loss_coeff=tune.choice([0.5, 1.0]),
                # tau=tune.choice([1, 0.1, 4e-4]),
            )
        else:

            config.training(
                train_batch_size=2048,
                # lr_schedule=[[0, 0.001], [5_000_000, 0.00075], [10_000_000, 3e-4]],
                lr=3e-5,
                entropy_coeff=0.01,
                # entropy_coeff_schedule=[[0, 0.03], [200_000_000, 0.01]],
                gamma=0.99,
                vf_loss_coeff=0.5,
            )

        return config

    def run(self):
        ray.init(
            local_mode=self.config.get("debug", False),
        )

        ray.tune.registry.register_env(
            "SlippiMultiAgentEnv",
            env_creator=multiagent_env_creator,
        )

        experiment_name = self.config.get("experiment_name")
        results_path = f"/home/c/ray_results/{experiment_name}"
        experiment_exists = os.path.exists(results_path)
        if experiment_exists and experiment_name != "test":
            print("Experiment already exists, restoring...")
            tuner = tune.Tuner.restore(
                results_path,
                trainable=appo.APPO,
            )

        else:
            model_config = self.construct_model_config()
            tune_config = self.construct_tune_config()
            run_config = self.construct_run_config()

            tuner = tune.Tuner(
                trainable=appo.APPO,
                param_space=model_config,
                tune_config=tune_config,
                run_config=run_config,
            )

        tuner.fit()
