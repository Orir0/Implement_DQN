import gymnasium as gym
import ale_py

# Create environment
env = gym.make('ALE/Breakout-v5', frameskip=1)

print("Action space:", env.action_space)
print("Number of actions:", env.action_space.n)
print("\nAction meanings:")

# Try to get action meanings from the unwrapped environment
unwrapped = env.unwrapped
if hasattr(unwrapped, 'get_action_meanings'):
    action_meanings = unwrapped.get_action_meanings()
    for i, meaning in enumerate(action_meanings):
        print(f"  Action {i}: {meaning}")
else:
    # Standard Breakout actions
    print("  Action 0: NOOP (No operation - do nothing)")
    print("  Action 1: FIRE (Start/launch the ball)")
    print("  Action 2: UP (Not used in Breakout)")
    print("  Action 3: RIGHT (Move paddle right)")
    print("  Action 4: LEFT (Move paddle left)")
    print("  Action 5: DOWN (Not used in Breakout)")
    print("  Action 6: UPRIGHT (Not used in Breakout)")
    print("  Action 7: UPLEFT (Not used in Breakout)")
    print("  Action 8: DOWNRIGHT (Not used in Breakout)")
    print("  Action 9: DOWNLEFT (Not used in Breakout)")
    print("\nNote: In Breakout, only actions 0, 1, 3, and 4 are typically used")
    print("  - Action 0: Do nothing")
    print("  - Action 1: Fire/launch ball")
    print("  - Action 3: Move right")
    print("  - Action 4: Move left")

env.close()

