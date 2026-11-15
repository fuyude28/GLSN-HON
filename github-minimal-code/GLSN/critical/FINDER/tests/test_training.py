import csv

from finder_pytorch.agent import FinderDQNAgent
from finder_pytorch.training import TrainingConfig, load_training_checkpoint, train_finder_nd


def test_training_writes_csv_log_and_checkpoint(tmp_path):
    agent = FinderDQNAgent(hidden_dim=8, batch_size=2, device="cpu")
    log_file = tmp_path / "train.csv"
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainingConfig(
        episodes=3,
        warmup_episodes=0,
        num_min=6,
        num_max=7,
        ba_m=2,
        seed=3,
        log_interval=1,
        log_file=log_file,
        checkpoint_interval=2,
        checkpoint_dir=checkpoint_dir,
    )

    result = train_finder_nd(agent, config)

    assert result.episodes_completed == 3
    with log_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["episode"]) for row in rows] == [1, 2, 3]
    assert {"episode", "epsilon", "loss", "replay_size", "elapsed_seconds"}.issubset(rows[0])

    checkpoint = checkpoint_dir / "checkpoint_ep_000002.pt"
    assert checkpoint.exists()
    loaded_agent, loaded_state = load_training_checkpoint(checkpoint, device="cpu")
    assert loaded_state.episodes_completed == 2
    assert loaded_agent.optimization_steps > 0


def test_training_resume_continues_from_checkpoint(tmp_path):
    first_agent = FinderDQNAgent(hidden_dim=8, batch_size=2, device="cpu")
    checkpoint_dir = tmp_path / "checkpoints"
    first_config = TrainingConfig(
        episodes=2,
        warmup_episodes=0,
        num_min=6,
        num_max=7,
        ba_m=2,
        seed=11,
        checkpoint_interval=2,
        checkpoint_dir=checkpoint_dir,
    )
    first = train_finder_nd(first_agent, first_config)
    assert first.episodes_completed == 2

    checkpoint = checkpoint_dir / "checkpoint_ep_000002.pt"
    resumed_agent, state = load_training_checkpoint(checkpoint, device="cpu")
    resumed_config = TrainingConfig(
        episodes=4,
        warmup_episodes=0,
        num_min=6,
        num_max=7,
        ba_m=2,
        seed=11,
    )
    second = train_finder_nd(resumed_agent, resumed_config, initial_state=state)

    assert second.episodes_completed == 4
    assert second.start_episode == 2
