import gym
try:
    # Test if render_mode parameter works
    env = gym.make('CartPole-v0', render_mode='human')
    print("✅ render_mode='human' works!")
    obs, info = env.reset()
    print("✅ Environment created successfully")
    env.close()
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying without render_mode...")
    env = gym.make('CartPole-v0')
    print("✅ Works without render_mode")
    env.close()