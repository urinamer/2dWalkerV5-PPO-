

# 2D Walker V5 - PPO

**Biped Robot learning to walk with reinforcement learning**

In this project I trained a **MuJoCo environment** of a biped robot to **walk on two feet** and **reach as far as possible** without falling over. I used the **PPO algorithm** to achieve **optimal and efficient training** that fits the **infinite action space** of the robot.

---

# Demo




---

# Installation

1. **Clone the repository:**
```bash
git clone https://github.com/urinamer/2dWalkerV5-PPO-.git
cd 2dWalkerV5-PPO-
```

2. **Install Dependencies:**
```bash
pip install gymnasium mujoco torch numpy

```


3. **Run Trainer to train or Tester to View Results:**
```bash
python trainer.py
python tester.py

```



---

# PPO Algorithm Short Summary

## What is the problem with other on-policy based methods?

Other **on-policy based methods** collect experiences (**rewards, actions, values**) and then use the **value** (*how much an action is worth*) to calculate the **advantage** of the action (*how much better the action was compared to what we thought*).

Because the advantage estimation is calculated using the value which is itself calculated using a neural network called **The Critic**, we will have to **throw away the experience every time we use it** because next time the value neural network updates, the advantage estimation will no longer be correct for the current policy.

## How PPO optimizes that?

**PPO** uses the **ratio between the new probability of an action and the old probability of an action** to determine if it changed too much such that it will lead to a wrong advantage calculation. If it is, **it clips it** so it will not change the probability of the action.

This allows us to **use the same experience 3-10 times** until the probability changes past the point where using it to calculate the advantage will inflict badly on training.

```

```
