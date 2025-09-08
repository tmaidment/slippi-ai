"""Training script for pure self-play using RLlib."""

import argparse
import os
import ray
from ray import tune
from ray.rllib.algorithms import appo

from slippi_ai.rllib.config import env_creator, get_self_play_config, get_multiagent_self_play_config


def main():
    parser = argparse.ArgumentParser(description="Train Slippi AI using RLlib self-play")
    parser.add_argument("--dolphin-path", type=str, default=None,
                        help="Path to Dolphin executable")
    parser.add_argument("--iso-path", type=str, required=True,
                        help="Path to Melee ISO file")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of rollout workers")
    parser.add_argument("--num-envs-per-worker", type=int, default=1,
                        help="Number of environments per worker")
    parser.add_argument("--train-batch-size", type=int, default=4000,
                        help="Training batch size")
    parser.add_argument("--sgd-minibatch-size", type=int, default=128,
                        help="SGD minibatch size")
    parser.add_argument("--num-sgd-iter", type=int, default=10,
                        help="Number of SGD iterations per training step")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Checkpoint frequency (iterations)")
    parser.add_argument("--num-iterations", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--multiagent", action="store_true",
                        help="Use multi-agent self-play setup")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--restore-checkpoint", type=str, default=None,
                        help="Path to checkpoint to restore from")
    
    args = parser.parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create configuration
    if args.multiagent:
        config = get_multiagent_self_play_config(
            dolphin_path=args.dolphin_path,
            iso_path=args.iso_path,
            num_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            checkpoint_freq=args.checkpoint_freq,
        )
    else:
        config = get_self_play_config(
            dolphin_path=args.dolphin_path,
            iso_path=args.iso_path,
            num_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            checkpoint_freq=args.checkpoint_freq,
        )
    
    ray.tune.registry.register_env("SlippiEnv-v0", env_creator)

    # Create algorithm
    algo = appo.APPO(config=config)
    
    # Restore from checkpoint if specified
    if args.restore_checkpoint:
        algo.restore(args.restore_checkpoint)
        print(f"Restored from checkpoint: {args.restore_checkpoint}")
    
    # Training loop
    try:
        for i in range(args.num_iterations):
            result = algo.train()
            
            print(f"Iteration {i + 1}:")
            
            # Handle different possible result keys
            if 'episode_reward_mean' in result:
                print(f"  Episode reward mean: {result['episode_reward_mean']:.3f}")
            elif 'env_runners' in result and 'episode_reward_mean' in result['env_runners']:
                print(f"  Episode reward mean: {result['env_runners']['episode_reward_mean']:.3f}")
            else:
                print(f"  Episode reward mean: N/A (key not found in result)")
                
            if 'episode_len_mean' in result:
                print(f"  Episode length mean: {result['episode_len_mean']:.1f}")
            elif 'env_runners' in result and 'episode_len_mean' in result['env_runners']:
                print(f"  Episode length mean: {result['env_runners']['episode_len_mean']:.1f}")
            else:
                print(f"  Episode length mean: N/A")
                
            if 'time_total_s' in result:
                print(f"  Training time: {result['time_total_s']:.1f}s")
            else:
                print(f"  Training time: N/A")
                
            # Debug: print available keys
            print(f"  Available result keys: {list(result.keys())}")
            print()
            
            if 'info' in result and 'learner' in result['info'] and 'default_policy' in result['info']['learner'] and 'learner_stats' in result['info']['learner']['default_policy']:
                print(f"  Policy loss: {result['info']['learner']['default_policy']['learner_stats']['policy_loss']:.6f}")
                print(f"  Value function loss: {result['info']['learner']['default_policy']['learner_stats']['vf_loss']:.6f}")
            
            # Save checkpoint
            if (i + 1) % args.checkpoint_freq == 0:
                checkpoint_path = algo.save(args.checkpoint_dir)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Save final checkpoint
        final_checkpoint = algo.save(args.checkpoint_dir)
        print(f"Final checkpoint saved: {final_checkpoint}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
