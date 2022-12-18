from tqdm import trange

def fill_memory(agent, env, num_episodes=500 ):
    print("Filling up memory....")
    for _ in trange(num_episodes):
        state = env.reset()
        done = False 
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.cache(state, next_state, action, reward, done)
            state = next_state


def train(agent, env, logger):
    episodes = 5000
    for e in range(episodes):

        state = env.reset()
        # Play the game!
        while True:
        
            # Run agent on the state
            action = agent.act(state)
            
            # Agent performs action
            next_state, reward, done, info = env.step(action)
            
            # Remember
            agent.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = agent.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state
            
            # Check if end of game
            if done:
                break

        logger.log_episode(e)

        if e % 20 == 0:
            logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
