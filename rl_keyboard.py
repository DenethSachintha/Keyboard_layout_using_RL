# rl_keyboard.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

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


# ------------------ ENVIRONMENT ------------------
class KeyboardEnv:
    def __init__(self, letter_freqs, bigram_freqs, top9_list=None, max_steps=200):
        freqs = np.zeros(26, dtype=np.float32)
        for c, v in letter_freqs.items():
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
        self.reset()

    def reset(self):
        self.layout = list(range(26))
        random.shuffle(self.layout)
        self.step_count = 0
        self.prev_score = self._compute_score()
        return self._get_obs()

    def _get_obs(self):
        one_hot = np.zeros((26, 26), dtype=np.float32)
        for slot, letter_idx in enumerate(self.layout):
            one_hot[slot, letter_idx] = 1.0
        slot_freqs = np.array(
            [self.letter_freqs[self.layout[s]] for s in range(26)], dtype=np.float32
        )
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
        pos_of_letter = {letter_idx: pos for pos, letter_idx in enumerate(self.layout)}

        # Rule 1: top9 in home row
        top9_idxs = [LETTER_TO_IDX[c] for c in self.top9]
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

        # Rule 3: balance home row
        left_home_positions = list(range(10, 15))
        right_home_positions = list(range(15, 19))
        L = sum(self.letter_freqs[self.layout[p]] for p in left_home_positions)
        R = sum(self.letter_freqs[self.layout[p]] for p in right_home_positions)
        if L + R > 0:
            balance_score = 1.0 - abs(L - R) / (L + R)
        else:
            balance_score = 0.0
        balance_reward = balance_score * 10.0

        return top9_reward + bigram_reward + balance_reward

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
            nn.Linear(input_dim, 256),  # reduced size for CPU
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ------------------ TRAINING ------------------
def train_agent(letter_freqs, bigram_freqs, top9_list=None, n_episodes=500, episode_len=80, gamma=0.99, lr=1e-3):
    env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=episode_len)
    obs_dim = env.reset().shape[0]
    policy = PolicyNet(obs_dim, N_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    best_score = -1e9
    best_layout = None

    for ep in trange(n_episodes, desc="Training"):
        obs = env.reset()
        log_probs = []
        rewards = []

        for _ in range(episode_len):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)  # keep on CPU
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            if done:
                break

        # Compute returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss = loss - lp * G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if info["score"] > best_score:
            best_score = info["score"]
            best_layout = env.layout.copy()

    return policy, best_layout, best_score


# ------------------ GENERATE FINAL LAYOUT ------------------
def generate_layout(policy, letter_freqs, bigram_freqs, top9_list=None, steps=200):
    env = KeyboardEnv(letter_freqs, bigram_freqs, top9_list, max_steps=steps)
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
    # Example input
    letter_freqs = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7,
        's': 6.3, 'h': 6.1, 'r': 6.0
    }
    for c in LETTERS:
        letter_freqs.setdefault(c, 0.5)

    bigram_freqs = {'th': 3.5, 'he': 2.8, 'in': 2.0, 'er': 1.8, 'an': 1.6}
    top9 = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']

    policy, best_layout, best_score = train_agent(letter_freqs, bigram_freqs, top9, n_episodes=400)
    print("Best score during training:", best_score)

    mapping, score, layout_str = generate_layout(policy, letter_freqs, bigram_freqs, top9)
    print("Final Layout Mapping:", mapping)
    print("Final Score:", score)
    print(layout_str)
