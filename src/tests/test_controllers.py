#tests/test_controllers.py, Running a Quick controller test
import gymnasium as gym
from controllers.cpg import CPGController
from controllers.reservoir import ReservoirController
def test_controllers():
    for env_name in ("Ant-v5", "Swimmer-v5"):
        # Step 1: create the environment and make its cleanup reliable.
        env = gym.make(env_name)
        try:
            # Step 2: reset the environment to get its first observation.
            observation, _ = env.reset(seed=42)

            # Step 3: initialize both controllers from the environment spaces.
            cpg = CPGController(action_space=env.action_space)
            reservoir = ReservoirController(
                action_space=env.action_space,
                input_dim=observation.shape[0],
                reservoir_size=100,
                seed=42,
            )

            # Step 4: generate one action from each controller.
            cpg_action = cpg.act(observation)
            reservoir_action = reservoir.act(observation)

            # Step 5: confirm actions match the environment shape and bounds.
            assert cpg_action.shape == env.action_space.shape
            assert reservoir_action.shape == env.action_space.shape
            assert env.action_space.contains(cpg_action)
            assert env.action_space.contains(reservoir_action)

            # Step 6: report results only after all checks have passed.
            print(f"\n{env_name}: PASS")
            print(f"Observation shape: {observation.shape}")
            print(f"Action shape: {env.action_space.shape}")
            print(f"CPG action: {cpg_action}")
            print(f"Reservoir action: {reservoir_action}")
        finally:
            env.close()

    print("\nAll controller checks passed.")


if __name__ == "__main__":
    test_controllers()


