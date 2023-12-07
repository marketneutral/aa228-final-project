using Random

# Define the gridworld environment
grid_size = (5, 5)
goal_state = (5, 5)
start_state = (1, 1)
obstacles = [(3, 3)]

# Define actions
actions = ["up", "down", "left", "right"]
num_actions = length(actions)

# Initialize theta, a parameter vector for our linear approximator, with small random values
theta = randn(grid_size[1] * grid_size[2] * num_actions)

# Feature vector construction
function feature_vector(state, action)
    # One-hot encoding for state and action
    s_idx = (state[1] - 1) * grid_size[2] + state[2]
    a_idx = findfirst(isequal(action), actions)
    features = zeros(grid_size[1] * grid_size[2] * num_actions)
    features[(a_idx - 1) * grid_size[1] * grid_size[2] + s_idx] = 1
    return features
end

# Approximate Q-function
function Q_grad(state, action, theta)
    features = feature_vector(state, action)
    return dot(theta, features)
end

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration rate

# Rest of the functions (valid_actions, step, choose_action) remain the same

# Gradient update for the linear function approximator
function update_theta!(theta, state, action, reward, next_state, alpha, gamma)
    features = feature_vector(state, action)
    q_predict = Q_grad(state, action, theta)
    q_target = reward
    if next_state != goal_state
        q_target += gamma * maximum([Q_grad(next_state, a, theta) for a in actions])
    end
    # Gradient descent step
    theta -= alpha * (q_predict - q_target) * features
end

# Training loop
for episode in 1:1000
    state = start_state
    while state != goal_state
        action = choose_action(state)
        next_state, reward = step(state, action)
        update_theta!(theta, state, action, reward, next_state, alpha, gamma)
        state = next_state
    end
end

println("Learned weights for theta:")
println(theta)

# get the optimal action for each state
policy = Dict()
for state in [(i, j) for i in 1:grid_size[1], j in 1:grid_size[2]]
    qs = [Q_grad(state, a, theta) for a in actions]
    max_q = maximum(qs)
    best_actions = [a for a in actions if Q_grad(state, a, theta) == max_q]
    policy[state] = rand(best_actions)
end

# print the policy as a grid
for i in 1:grid_size[1]
    for j in 1:grid_size[2]
        print(policy[(i, j)], " ")
    end
    println()
end