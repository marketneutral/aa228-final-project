using Random

# Define the gridworld environment
grid_size = (5, 5)
goal_state = (5, 5)
start_state = (1, 1)
obstacles = [(3,3)]

# Define actions
actions = ["up", "down", "left", "right"]

# Initialize Q-table
Q = Dict(((i, j), a) => 0.0 for i in 1:grid_size[1], j in 1:grid_size[2], a in actions)

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration rate

# # Function to return the valid actions from a state
function valid_actions(state)
    i, j = state
    valid = []
    if i > 1 push!(valid, "up") end
    if i < grid_size[1] push!(valid, "down") end
    if j > 1 push!(valid, "left") end
    if j < grid_size[2] push!(valid, "right") end
    return valid
end

# Function to take an action in the environment
function step(state, action)
    i, j = state
    if action == "up" i -= 1 end
    if action == "down" i += 1 end
    if action == "left" j -= 1 end
    if action == "right" j += 1 end
    next_state = (max(min(i, grid_size[1]), 1), max(min(j, grid_size[2]), 1))
    reward = -1.0
    if next_state == goal_state
        reward = 100.0
    elseif next_state in obstacles
        reward = -100.0
        next_state = state # Hit obstacle, stay in place
    end
    return next_state, reward
end

# Function to choose an action using epsilon-greedy policy
function choose_action(state)
    if rand() < epsilon
        return rand(valid_actions(state))
    else
        qs = [Q[(state, a)] for a in valid_actions(state)]
        max_q = maximum(qs)
        best_actions = [a for a in valid_actions(state) if Q[(state, a)] == max_q]
        return rand(best_actions)
    end
end

# Q-learning algorithm
for episode in 1:1000
    state = start_state
    while state != goal_state
        action = choose_action(state)
        next_state, reward = step(state, action)
        old_value = Q[(state, action)]
        next_max = maximum([Q[(next_state, a)] for a in valid_actions(next_state)])
        Q[(state, action)] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        #Q[(state, action)] = alpha * ((reward + gamma * next_max) - old_value)

        state = next_state
    end
end

# get the optimal action for each state
policy = Dict()
for state in [(i, j) for i in 1:grid_size[1], j in 1:grid_size[2]]
    qs = [Q[(state, a)] for a in valid_actions(state)]
    max_q = maximum(qs)
    best_actions = [a for a in valid_actions(state) if Q[(state, a)] == max_q]
    policy[state] = rand(best_actions)
end

# print the policy as a grid
for i in 1:grid_size[1]
    for j in 1:grid_size[2]
        print(policy[(i, j)], "\t")
    end
    println()
end

println("Q-values after learning:")
for (k, v) in Q
    println("$k => $v")
end
