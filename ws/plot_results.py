import json

import matplotlib.pyplot as plt


def main():
    # Load the results from the JSON file
    with open("./results/frozen_lake_1/results.json", "r") as f:
        results = json.load(f)

    # Extract the total rewards
    total_rewards = results["total_rewards"]

    # Plot the total rewards

    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards over Episodes")
    plt.show()


if __name__ == "__main__":
    main()
