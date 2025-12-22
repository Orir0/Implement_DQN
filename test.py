import gymnasium as gym
import ale_py



try:
    env = gym.make('ALE/Breakout-v5')  # Try v5 first
    print("✅ Environment created!")
    
    obs, info = env.reset()
    print(f"✅ Reset successful! Observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful! Reward: {reward}")
    
    env.close()
    print("✅ Everything works! No OpenCV needed.")
    
except Exception as e:
    print(f"❌ Error: {e}")