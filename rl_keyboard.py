# rl_keyboard.py
"""
RL Keyboard layout generator (REINFORCE) - phased training with penalties
Features:
 - EPISODES = 20000 total, train 20% per --phase invocation (4000 episodes per phase)
 - Balance reward scaled to +30 but only applied if ALL top9 letters are in home row
 - Penalties appended:
     * -0.1 per letter position change from QWERTY
     * -0.5 penalty for each top9 letter placed on TOP or BOTTOM rows (encourage home row)
     * -0.2 penalty if frequent bigram letters use the SAME FINGER (in addition to existing rewards)
 - QWERTY initialization for episodes (small randomization sometimes)
 - Save partial checkpoint at each best improvement and at end of phase
 - Resume automatically if checkpoint exists
 - Validation and CSV logging per episode
"""

import os
import csv
import argparse
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# ------------------ CONFIG ------------------
EPISODES = 20000             # total episodes across all phases
PHASE_FRACTION = 0.2         # 20% per phase
EPISODES_PER_PHASE = int(EPISODES * PHASE_FRACTION)  # 4000
EPISODE_LEN = 120            # steps per episode
GAMMA = 0.99
LR = 1e-3
VALIDATE_EVERY = 500         # validation every N episodes during a phase
VALIDATION_RUNS = 10
CHECKPOINT_PATH = "best_policy_phase.pth"
LOG_CSV = "training_log_phased.csv"
SEED = 42

# penalty constants (tweakable)
QWERTY_DEVIATION_PENALTY = -0.1   # per letter moved from QWERTY slot
TOP9_ROW_PENALTY = -0.5           # per top9 letter placed on top row or bottom row
BIGRAM_SAME_FINGER_PENALTY = -0.2 # penalty when bigram letters use same finger

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
        # store bigram freqs as provided (no normalization) but cast to floats
        self.bigram_freqs = {k.lower(): float(v) for k, v in bigram_freqs.items()}
        if top9_list:
            self.top9 = [c.lower() for c in top9_list]
        else:
            top_idxs = np.argsort(-self.letter_freqs)[:9]
            self.top9 = [IDX_TO_LETTER[i] for i in top_idxs]

        self.max_steps = max_steps
        # store qwerty mapping indices for penalties & initialization
        self.qwerty_indices = self._qwerty_layout_indices()
        self.reset(start_layout=self.qwerty_indices)

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
                # small local shuffle to encourage exploration
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
        """Compute total score using rules. Balance reward scaled to +30 but only when all top9 in home row.
        Appended penalties are also applied here.
        """
        pos_of_letter = {letter_idx: pos for pos, letter_idx in enumerate(self.layout)}

        # Rule 1: top9 in home row (+5 per top9 placed in home row)
        top9_idxs = [LETTER_TO_IDX[c] for c in self.top9 if c in LETTER_TO_IDX]
        top9_in_home = sum(1 for pos in HOME_RANGE if self.layout[pos] in top9_idxs)
        top9_reward = 5.0 * top9_in_home

        # Rule 2: bigram rewards (existing scheme) + same-finger penalty appended
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
            # existing positive rewards
            if hand_a != hand_b:
                bigram_reward += freq * 2.0
            elif finger_a != finger_b:
                bigram_reward += freq * 1.0
            # appended penalty if same finger
            if finger_a == finger_b and hand_a == hand_b:
                bigram_reward += BIGRAM_SAME_FINGER_PENALTY * freq  # scaled by freq

        # Rule 3: balance home row -> scaled to +30 but only if all top9 are in home row
        left_home_positions = list(range(10, 15))
        right_home_positions = list(range(15, 19))
        L = sum(self.letter_freqs[self.layout[p]] for p in left_home_positions)
        R = sum(self.letter_freqs[self.layout[p]] for p in right_home_positions)
        if L + R > 0:
            balance_score = 1.0 - abs(L - R) / (L + R)
        else:
            balance_score = 0.0
        if top9_in_home == len(self.top9):
            balance_reward = balance_score * 30.0   # only if all top9 in home row
        else:
            balance_reward = 0.0

        # Existing small penalty for placing very frequent letters on bottom row (kept)
        bottom_penalty = 0.0
        for pos in BOTTOM_RANGE:
            idx = self.layout[pos]
            bottom_penalty -= 0.1 * float(self.letter_freqs[idx])

        # APPENDED PENALTIES (as requested) without removing existing rewards:
        # 1) Penalty for deviation from QWERTY positions: -0.1 per letter that's not on the QWERTY slot
        qwerty_dev_penalty = 0.0
        for slot, letter_idx in enumerate(self.layout):
            q_idx = self.qwerty_indices[slot]  # which letter QWERTY expects at this slot
            # we check if the letter currently at this slot equals the QWERTY letter for this slot
            # If not equal, penalty applies (letter moved from its QWERTY position)
            if letter_idx != q_idx:
                qwerty_dev_penalty += QWERTY_DEVIATION_PENALTY

        # 2) Penalty for placing top9 letters on TOP or BOTTOM rows (encourage them to be on HOME)
        top9_row_penalty = 0.0
        top_positions = set(TOP_RANGE)
        bottom_positions = set(BOTTOM_RANGE)
        for pos in list(TOP_RANGE) + list(BOTTOM_RANGE):
            letter_idx = self.layout[pos]
            letter = IDX_TO_LETTER[letter_idx]
            if letter in self.top9:
                top9_row_penalty += TOP9_ROW_PENALTY

        total = top9_reward + bigram_reward + balance_reward + bottom_penalty + qwerty_dev_penalty + top9_row_penalty
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
        obs = env.reset(start_layout=env._qwerty_layout_indices())  # evaluate from QWERTY
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


# ------------------ TRAINING (PHASED REINFORCE with baseline) ------------------
def train_phase(letter_freqs, bigram_freqs, top9_list=None,
                phase_idx=1, episodes_per_phase=EPISODES_PER_PHASE, episode_len=EPISODE_LEN,
                gamma=GAMMA, lr=LR, checkpoint_path=CHECKPOINT_PATH):
    """
    Train one phase. phase_idx is 1-based (1..int(1/PHASE_FRACTION))
    """

    # Sanity check
    max_phases = int(1.0 / PHASE_FRACTION)
    if not (1 <= phase_idx <= max_phases):
        raise ValueError(f"phase must be between 1 and {max_phases}")

    total_episodes = EPISODES
    episodes_per_phase = episodes_per_phase
    start_episode = (phase_idx - 1) * episodes_per_phase
    end_episode = phase_idx * episodes_per_phase

    # create temporary env to determine observation dimension
    tmp_env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=episode_len)
    obs_dim = tmp_env._get_obs().shape[0]
    policy = PolicyNet(obs_dim, N_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Try to load checkpoint if exists (resume)
    best_score = -1e9
    last_saved_episode = -1
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            policy.load_state_dict(ckpt["policy_state_dict"])
            best_score = ckpt.get("best_score", best_score)
            last_saved_episode = ckpt.get("episode", -1)
            print(f"> Loaded checkpoint '{checkpoint_path}' (best_score={best_score:.4f}, saved_episode={last_saved_episode})")
        except Exception as e:
            print("Warning: failed to load checkpoint:", e)

    # If checkpoint indicates we've already passed the target phase end, do nothing
    if last_saved_episode >= end_episode:
        print(f"> Checkpoint already at episode {last_saved_episode} which is >= requested phase end {end_episode}. Nothing to do.")
        return policy, best_score, last_saved_episode

    # If checkpoint inside phase range, resume from next episode, else start from start_episode
    current_start = max(start_episode, last_saved_episode + 1)
    if current_start > end_episode:
        print(f"> Nothing to train in this phase (current_start={current_start} > end_episode={end_episode}).")
        return policy, best_score, last_saved_episode

    # prepare logging CSV header if not exists
    header = ["phase", "global_episode", "episode_return", "episode_score", "best_score", "time_elapsed_s", "val_mean_score", "val_best_score"]
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    running_baseline = 0.0
    baseline_alpha = 0.01

    policy.train()
    global_start = time.time()
    # iterate global episodes from current_start to end_episode-1 inclusive
    for global_ep in trange(current_start, end_episode, desc=f"Phase {phase_idx} training"):
        # create env for episode and initialize from QWERTY (randomize a bit every 10th episode for exploration)
        env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=episode_len)
        obs = env.reset(start_layout=env._qwerty_layout_indices(), randomize=(global_ep % 10 == 0))
        log_probs = []
        rewards = []
        ep_return_sum = 0.0

        for t in range(episode_len):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            ep_return_sum += reward
            if done:
                break

        # compute returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # running baseline
        running_baseline = (1 - baseline_alpha) * running_baseline + baseline_alpha * float(returns.sum().item())
        returns = returns - running_baseline

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy gradient step
        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss = loss - lp * G
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        episode_score = env.prev_score

        # save checkpoint if improved best_score
        if episode_score > best_score:
            best_score = float(episode_score)
            last_saved_episode = global_ep
            try:
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "best_score": best_score,
                    "episode": int(last_saved_episode)
                }, checkpoint_path)
                # human-readable best layout
                with open("best_layout_phase.txt", "w") as f:
                    f.write(env.render_layout() + "\n")
            except Exception as e:
                print("Warning: failed to save checkpoint:", e)

        # validation at intervals inside the phase (mean greedy score across runs)
        val_mean = None
        val_best = None
        val_render = None
        if (global_ep + 1) % VALIDATE_EVERY == 0 or global_ep == end_episode - 1:
            val_mean, val_best, val_render = evaluate_policy(policy, letter_freqs, bigram_freqs, top9_list, runs=VALIDATION_RUNS)
            print(f"\n=== Validation (phase {phase_idx}) @global_ep {global_ep+1}: mean_score={val_mean:.4f} best_score={val_best:.4f} ===")
            if val_render:
                print(val_render)

        # append CSV log
        elapsed = time.time() - global_start
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([phase_idx, global_ep + 1, float(sum(rewards)), float(episode_score), float(best_score), int(elapsed), val_mean or "", val_best or ""])

    phase_time = time.time() - global_start
    print(f"\nPhase {phase_idx} complete. Phase episodes [{current_start}..{end_episode-1}]. Best score now: {best_score:.4f}. Phase time: {phase_time:.1f}s")
    return policy, best_score, last_saved_episode


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
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=None, help="Phase index to run (1..{:.0f})".format(1.0 / PHASE_FRACTION))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase is None:
        print("Please run with --phase N where N is 1..{}".format(int(1.0 / PHASE_FRACTION)))
        exit(1)

    # Example letter frequencies: fill small values for missing letters
    letter_freqs = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7,
        's': 6.3, 'h': 6.1, 'r': 6.0
    }
    for c in LETTERS:
        letter_freqs.setdefault(c, 0.5)

    # Doubling bigram list length as requested: added additional common English bigrams
    bigram_freqs = {
        'th': 3.5, 'he': 2.8, 'in': 2.0, 'er': 1.8, 'an': 1.6,
        're': 1.5, 'ed': 1.4, 'on': 1.3, 'es': 1.2, 'st': 1.1
    }

    # Top9 list (explicit)
    top9 = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']

    # Run phase training
    phase = args.phase
    if phase < 1 or phase > int(1.0 / PHASE_FRACTION):
        raise ValueError(f"phase must be in 1..{int(1.0 / PHASE_FRACTION)}")

    print(f"Starting phase {phase} training (episodes per phase = {EPISODES_PER_PHASE})...")
    policy_model, best_score, saved_ep = train_phase(letter_freqs, bigram_freqs, top9_list=top9, phase_idx=phase, episodes_per_phase=EPISODES_PER_PHASE, episode_len=EPISODE_LEN, lr=LR, checkpoint_path=CHECKPOINT_PATH)

    # After phase, produce a greedy final layout from QWERTY using current policy
    mapping, score, pretty = generate_layout(policy_model, letter_freqs, bigram_freqs, top9_list=top9, steps=500)
    print("\nGreedy layout after this phase (starting from QWERTY):")
    print(pretty)
    print("Mapping (slot -> letter):", mapping)
    print("Greedy score:", score)

    print(f"\nPhase {phase} done. Checkpoint: {CHECKPOINT_PATH}, logs: {LOG_CSV}, best layout file: best_layout_phase.txt (if created).")
