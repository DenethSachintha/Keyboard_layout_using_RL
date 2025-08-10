# rl_keyboard.py
"""
RL Keyboard layout generator (REINFORCE) - CPU optimized
Features added per user request:
 - EPISODES = 20000
 - Balance (home-row equalized load) reward scaled to +30
 - Start every training episode from QWERTY layout (deterministic init)
 - Save best model (best_score) to disk and resume if exists
 - Detailed logging (CSV) and progress prints
 - Periodic validation / "accuracy" checks over time (avg score on greedy rollouts)
 - Small improvements: running baseline for variance reduction, grad clipping
"""

import os
import csv
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# ------------------ CONFIG ------------------
EPISODES = 20000             # as requested
EPISODE_LEN = 120            # steps per episode
GAMMA = 0.99
LR = 1e-3
VALIDATE_EVERY = 500         # validate every N episodes
VALIDATION_RUNS = 10         # number of greedy rollouts during validation
CHECKPOINT_PATH = "best_policy.pth"
LOG_CSV = "training_log.csv"
SEED = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------ CONSTANTS ------------------
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
LETTER_TO_IDX = {c: i for i, c in enumerate(LETTERS)}
IDX_TO_LETTER = {i: c for c, i in LETTER_TO_IDX.items()}

TOP_RANGE = range(0, 10)      # 10 slots
HOME_RANGE = range(10, 19)    # 9 slots
BOTTOM_RANGE = range(19, 26)  # 7 slots

LEFT_POSITIONS = set([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23])
RIGHT_POSITIONS = set(range(26)) - LEFT_POSITIONS

# All possible swap actions (i < j)
ACTION_PAIRS = [(i, j) for i in range(26) for j in range(i + 1, 26)]
N_ACTIONS = len(ACTION_PAIRS)  # 325


def get_hand_and_finger(pos):
    """Return (hand, finger_index) for given slot."""
    if pos in range(0, 10):  # top row
        if pos <= 4:
            return "L", pos
        else:
            return "R", pos - 5
    if pos in range(10, 19):  # home row
        if pos <= 14:
            return "L", pos - 10
        else:
            return "R", pos - 15
    if pos in range(19, 26):  # bottom row
        if pos <= 23:
            return "L", pos - 19
        else:
            return "R", pos - 24
    raise ValueError("Invalid position")


# QWERTY mapping for 26 letters into slots:
# top (10): q w e r t y u i o p
# home (9): a s d f g h j k l
# bottom (7): z x c v b n m
QWERTY_ORDER = list("qwertyuiopasdfghjklzxcvbnm")


# ------------------ ENVIRONMENT ------------------
class KeyboardEnv:
    def __init__(self, letter_freqs, bigram_freqs, top9_list=None, max_steps=200):
        freqs = np.zeros(26, dtype=np.float32)
        for c, v in letter_freqs.items():
            c = c.lower()
            if c in LETTER_TO_IDX:
                freqs[LETTER_TO_IDX[c]] = float(v)

        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()

        self.letter_freqs = freqs
        self.bigram_freqs = {k.lower(): float(v) for k, v in bigram_freqs.items()}
        if top9_list:
            self.top9 = [c.lower() for c in top9_list]
        else:
            top_idxs = np.argsort(-self.letter_freqs)[:9]
            self.top9 = [IDX_TO_LETTER[i] for i in top_idxs]

        self.max_steps = max_steps
        self.reset(start_layout=self._qwerty_layout_indices())

    def _qwerty_layout_indices(self):
        """Return the indices list that correspond to QWERTY_ORDER above."""
        return [LETTER_TO_IDX[c] for c in QWERTY_ORDER]

    def reset(self, start_layout=None, randomize=False):
        """
        start_layout: list of 26 letter indices. If None -> random.
        randomize: if True, will slightly shuffle QWERTY to inject exploration restarts.
        """
        if start_layout is None:
            self.layout = list(range(26))
            random.shuffle(self.layout)
        else:
            self.layout = start_layout.copy()
            if randomize:
                # small shuffles to encourage exploration from good start
                for _ in range(3):
                    i, j = random.randrange(26), random.randrange(26)
                    self.layout[i], self.layout[j] = self.layout[j], self.layout[i]

        self.step_count = 0
        self.prev_score = self._compute_score()
        return self._get_obs()

    def _get_obs(self):
        # one-hot layout + slot freq vector
        one_hot = np.zeros((26, 26), dtype=np.float32)
        for slot, letter_idx in enumerate(self.layout):
            one_hot[slot, letter_idx] = 1.0
        slot_freqs = np.array([self.letter_freqs[self.layout[s]] for s in range(26)], dtype=np.float32)
        return np.concatenate([one_hot.flatten(), slot_freqs])

    def step(self, action_idx):
        i, j = ACTION_PAIRS[action_idx]
        self.layout[i], self.layout[j] = self.layout[j], self.layout[i]
        new_score = self._compute_score()
        reward = float(new_score - self.prev_score)
        self.prev_score = new_score
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {"score": new_score}

    def _compute_score(self):
        """Compute total score using rules. Balance reward scaled to +30 as requested."""
        pos_of_letter = {letter_idx: pos for pos, letter_idx in enumerate(self.layout)}

        # Rule 1: top9 in home row (+5 per top9 placed in home row)
        top9_idxs = [LETTER_TO_IDX[c] for c in self.top9 if c in LETTER_TO_IDX]
        top9_in_home = sum(1 for pos in HOME_RANGE if self.layout[pos] in top9_idxs)
        top9_reward = 5.0 * top9_in_home

        # Rule 2: bigram rewards
        bigram_reward = 0.0
        for bigram, freq in self.bigram_freqs.items():
            if len(bigram) != 2:
                continue
            a, b = bigram[0], bigram[1]
            if a not in LETTER_TO_IDX or b not in LETTER_TO_IDX:
                continue
            ia, ib = LETTER_TO_IDX[a], LETTER_TO_IDX[b]
            pa, pb = pos_of_letter[ia], pos_of_letter[ib]
            hand_a, finger_a = get_hand_and_finger(pa)
            hand_b, finger_b = get_hand_and_finger(pb)
            if hand_a != hand_b:
                bigram_reward += freq * 2.0
            elif finger_a != finger_b:
                bigram_reward += freq * 1.0

        # Rule 3: balance home row -> scaled to +30
        left_home_positions = list(range(10, 15))
        right_home_positions = list(range(15, 19))
        L = sum(self.letter_freqs[self.layout[p]] for p in left_home_positions)
        R = sum(self.letter_freqs[self.layout[p]] for p in right_home_positions)
        if L + R > 0:
            balance_score = 1.0 - abs(L - R) / (L + R)
        else:
            balance_score = 0.0
        balance_reward = balance_score * 30.0   # changed from 10 -> 30

        # Add small penalty for placing very frequent letters on bottom row (encourage top/home)
        bottom_penalty = 0.0
        for pos in BOTTOM_RANGE:
            idx = self.layout[pos]
            # penalize proportionally to letter frequency (small penalty)
            bottom_penalty -= 0.1 * float(self.letter_freqs[idx])

        total = top9_reward + bigram_reward + balance_reward + bottom_penalty
        return float(total)

    def render_layout(self):
        top = "".join(IDX_TO_LETTER[self.layout[i]] for i in TOP_RANGE)
        home = "".join(IDX_TO_LETTER[self.layout[i]] for i in HOME_RANGE)
        bottom = "".join(IDX_TO_LETTER[self.layout[i]] for i in BOTTOM_RANGE)
        return f"TOP:    {top}\nHOME:   {home}\nBOTTOM: {bottom}"


# ------------------ POLICY MODEL (CPU-Optimized) ------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # moderate size for CPU
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ------------------ TRAIN / EVAL UTILITIES ------------------
def evaluate_policy(policy, letter_freqs, bigram_freqs, top9_list=None, runs=10, steps=300):
    """Run greedy rollouts and return average score and best layout string."""
    policy.eval()
    scores = []
    best_score = -1e9
    best_render = None
    for _ in range(runs):
        env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=steps)
        # initialize from QWERTY for fair evaluation
        obs = env.reset(start_layout=env._qwerty_layout_indices())
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1))
            obs, _, done, info = env.step(action)
        score = info["score"]
        scores.append(score)
        if score > best_score:
            best_score = score
            best_render = env.render_layout()
    policy.train()
    return float(np.mean(scores)), float(best_score), best_render


# ------------------ TRAINING (REINFORCE with running baseline) ------------------
def train_agent(letter_freqs, bigram_freqs, top9_list=None,
                episodes=EPISODES, episode_len=EPISODE_LEN, gamma=GAMMA, lr=LR,
                checkpoint_path=CHECKPOINT_PATH, validate_every=VALIDATE_EVERY):
    # env to get obs_dim
    tmp_env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=episode_len)
    obs_dim = tmp_env._get_obs().shape[0]
    policy = PolicyNet(obs_dim, N_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # load checkpoint if exists
    best_score = -1e9
    start_episode = 0
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            policy.load_state_dict(ckpt["policy_state_dict"])
            best_score = ckpt.get("best_score", best_score)
            start_episode = ckpt.get("episode", 0) + 1
            print(f"> Resumed checkpoint '{checkpoint_path}' at episode {start_episode} with best_score={best_score:.4f}")
        except Exception as e:
            print("Could not load checkpoint:", e)

    # prepare logging file
    header = ["episode", "episode_return", "episode_score", "best_score", "time_elapsed_s", "val_mean_score", "val_best_score"]
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    running_baseline = 0.0
    baseline_alpha = 0.01  # slow-moving baseline

    policy.train()
    total_start = time.time()
    for ep in trange(start_episode, episodes, desc="Training"):
        env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=episode_len)
        # Start from QWERTY each episode (with occasional tiny randomization for exploration)
        obs = env.reset(start_layout=env._qwerty_layout_indices(), randomize=(ep % 10 == 0))
        log_probs = []
        rewards = []
        episode_reward_sum = 0.0

        for t in range(episode_len):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # CPU tensor
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            episode_reward_sum += reward
            if done:
                break

        # compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # baseline for variance reduction (running baseline of episode return)
        running_baseline = (1 - baseline_alpha) * running_baseline + baseline_alpha * float(returns.sum().item())
        returns = returns - running_baseline  # simple baseline subtraction

        # normalize returns to unit variance for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy gradient loss
        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss = loss - lp * G
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        episode_score = env.prev_score
        # update best and checkpoint
        if episode_score > best_score:
            best_score = float(episode_score)
            # save checkpoint
            try:
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "best_score": best_score,
                    "episode": ep
                }, checkpoint_path)
                # also save a human-readable best layout
                with open("best_layout.txt", "w") as f:
                    f.write(env.render_layout() + "\n")
            except Exception as e:
                print("Warning: failed to save checkpoint:", e)

        # validation periodically
        val_mean = None
        val_best = None
        val_render = None
        if (ep + 1) % validate_every == 0 or ep == episodes - 1:
            val_mean, val_best, val_render = evaluate_policy(policy, letter_freqs, bigram_freqs, top9_list, runs=VALIDATION_RUNS)
            # print validation summary
            print(f"\n=== Validation @episode {ep+1}: mean_score={val_mean:.4f} best_score={val_best:.4f} ===")
            if val_render:
                print(val_render)
        # log to CSV
        elapsed = time.time() - total_start
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, float(sum(rewards)), float(episode_score), float(best_score), int(elapsed), val_mean or "", val_best or ""])

    total_time = time.time() - total_start
    print(f"\nTraining complete. Best score: {best_score:.4f}. Time elapsed: {total_time:.1f}s")
    return policy, best_score


# ------------------ GENERATE FINAL LAYOUT ------------------
def generate_layout(policy, letter_freqs, bigram_freqs, top9_list=None, steps=300, start_layout_qwerty=True):
    env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=steps)
    if start_layout_qwerty:
        obs = env.reset(start_layout=env._qwerty_layout_indices())
    else:
        obs = env.reset()
    best_score = env.prev_score
    best_layout = env.layout.copy()

    for _ in range(steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1))
        obs, _, _, info = env.step(action)
        if info["score"] > best_score:
            best_score = info["score"]
            best_layout = env.layout.copy()

    mapping = [IDX_TO_LETTER[i] for i in best_layout]
    return mapping, best_score, env.render_layout()


# ------------------ MAIN ------------------
if __name__ == "__main__":
    # Example input frequencies (you will replace these with real inputs)
    letter_freqs = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7,
        's': 6.3, 'h': 6.1, 'r': 6.0
    }
    for c in LETTERS:
        letter_freqs.setdefault(c, 0.5)

    bigram_freqs = {'th': 3.5, 'he': 2.8, 'in': 2.0, 'er': 1.8, 'an': 1.6}
    top9 = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']

    # Train (or resume)
    policy_model, best_score = train_agent(letter_freqs, bigram_freqs, top9_list=top9,
                                          episodes=EPISODES, episode_len=EPISODE_LEN, lr=LR,
                                          checkpoint_path=CHECKPOINT_PATH, validate_every=VALIDATE_EVERY)

    # Final greedy generation and print
    mapping, score, pretty = generate_layout(policy_model, letter_freqs, bigram_freqs, top9_list=top9, steps=500)
    print("\nFinal generated layout (greedy from QWERTY):")
    print(pretty)
    print("Mapping (slot -> letter):", mapping)
    print("Final score:", score)

    print(f"\nLogs saved to: {LOG_CSV}")
    print(f"Best policy checkpoint: {CHECKPOINT_PATH}")
    print("Best layout (human-readable) in best_layout.txt (if saved).")
