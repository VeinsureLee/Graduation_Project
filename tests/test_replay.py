import numpy as np

from algorithms.common.buffers import JointReplayBuffer, ReplayBuffer


def test_replay_buffer_basic():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(np.zeros(5, dtype=np.float32), 0, 1.0, np.zeros(5, dtype=np.float32), False)
    assert len(buf) == 10
    batch = buf.sample(4)
    assert len(batch) == 5
    assert batch[0].shape == (4, 5)


def test_replay_buffer_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.array([i], dtype=np.float32), 0, 0.0, np.array([i + 1], dtype=np.float32), False)
    assert len(buf) == 5


def test_joint_replay_buffer_basic():
    buf = JointReplayBuffer(capacity=100, num_agents=2, state_dim=5)
    for _ in range(10):
        joint_s = {0: np.zeros(5, dtype=np.float32), 1: np.zeros(5, dtype=np.float32)}
        joint_a = {0: 0, 1: 1}
        joint_r = {0: 1.0, 1: 0.5}
        joint_s_next = {0: np.zeros(5, dtype=np.float32), 1: np.zeros(5, dtype=np.float32)}
        joint_done = {0: False, 1: False}
        buf.push(joint_s, joint_a, joint_r, joint_s_next, joint_done)
    assert len(buf) == 10
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert states.shape == (4, 2, 5)
    assert actions.shape == (4, 2)
    assert rewards.shape == (4, 2)
