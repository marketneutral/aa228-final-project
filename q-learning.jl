using Random
using Plots
using LinearAlgebra: dot, norm

struct GridWorldState
    i::Int64
    j::Int64
    done::Bool
end

struct GridWorld
    size_i::Int64
    size_j::Int64
    reward_goal::Float64
    reward_penalty::Float64
    penalty_zone::Tuple{Int64, Int64}
end

gridworld = GridWorld(5, 5, 100.0, -100.0, (3, 3))
actions = [:up, :down, :left, :right]
states = [GridWorldState(i, j, false) for i in 1:gridworld.size_i, j in 1:gridworld.size_j]
initialstate = GridWorldState(1, 1, false)

function gen(m::GridWorld, s::GridWorldState, a::Symbol) #, rng::AbstractRNG)
    # if s.done
    #     return (sp=s, r=0.0)
    # end

    i = s.i
    j = s.j

    if a == :down
        i = min(i+1, m.size_i) # stay in the grid 
    elseif a == :up
        i = max(i-1, 1)
    elseif a == :left
        j = max(j-1, 1)
    elseif a == :right
        j = min(j+1, m.size_j)
    end

    done = (i == m.size_i) && (j == m.size_j)
    reward = done ? m.reward_goal : (i == m.penalty_zone[1] && j == m.penalty_zone[2] ? m.reward_penalty : -1.0)

    # stay in place if hit the obstacle
    if (i == m.penalty_zone[1] && j == m.penalty_zone[2])
        i = s.i
        j = s.j
    end

    return (sp=GridWorldState(i, j, done), r=reward)
end


mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # action value function
    alpha # learning rate
end


function lookahead(model::QLearning, s, a)
    return model.Q[((s.i, s.j), a)]
end



mutable struct EpsilonGreedyExploration
    epsilon # probability of random action
end

function valid_actions(state::GridWorldState)
    i, j = state.i, state.j
    valid = []
    if i > 1 push!(valid, :up) end
    if i < 5 push!(valid, :down) end
    if j > 1 push!(valid, :left) end
    if j < 5 push!(valid, :right) end
    return valid
end

# Define the epsilon-greedy policy
function π(model, s, exploration::EpsilonGreedyExploration)
    valids = valid_actions(s)
    if rand() < exploration.epsilon
        return rand(valids) # Explore: take a random action
    else
        values = [lookahead(model, s, a) for a in valids]
        return valids[argmax(values)] # Exploit: take the best action
    end
end

# Greedy policy
function π(model, s)
    values = [lookahead(model, s, a) for a in actions]
    return actions[argmax(values)]
end


function update!(model::QLearning, s, a, r, s_prime)
    gamma, Q, alpha = model.gamma, model.Q, model.alpha

    # the update should only look at valid actions in s_prime
    valid_s_prime_actions = valid_actions(s_prime)

    # Q[((s.i, s.j), a)] +=
    #     alpha*(
    #         r + gamma*maximum(Q[((s_prime.i, s_prime.j), a_prime)] for a_prime in valid_s_prime_actions)
    #         - Q[((s.i, s.j), a)]
    #     )

    Q[((s.i, s.j), a)] = (1-alpha)*Q[((s.i, s.j), a)] + alpha*(r + gamma*maximum(Q[((s_prime.i, s_prime.j), a_prime)] for a_prime in valid_s_prime_actions))
    return model
end



# h is the horizon
function simulate(P, model, π, h, s)
    for i in 1:h
        a = π(model, s)
        s_prime, r = gen(P, s, a)
        update!(model, s, a, r, s_prime)
        s = s_prime
    end
end


# Initialize the gridworld
Q = Dict{Tuple{Tuple{Int64, Int64}, Symbol}, Float64}()
for s in states
    for a in actions
        Q[((s.i, s.j), a)] = 0.0
    end
end

alpha = 0.1

model = QLearning(states, actions, 0.90, Q, alpha)

exploration = EpsilonGreedyExploration(0.1)
s = initialstate

# simulate fuction
simulate(gridworld, model, (m, s) -> π(m, s, exploration), 1_000, s)


# loop over all states and print the optimal policy
function get_optimal_policy(model)
    policy = Dict{GridWorldState, Symbol}()
    for s in states
        valids = valid_actions(s)
        a = valids[argmax(model.Q[((s.i, s.j), a)] for a in valids)]
        policy[s] = a
    end
    return policy
end

get_optimal_policy(model)

# print grid of optimal actions
function print_grid(model)
    grid = fill(" ", 5, 5)
    policy = get_optimal_policy(model)
    for s in states
        grid[s.i, s.j] = string(policy[s])
    end
    return grid  # Display the first letter of the optimal action
end

print_grid(model)





# Gradient-based Q-learning

struct GradientQLearning
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # parameterized action value function Q(theta,s,a)
    grad_Q # gradient of action value function
    theta # action value function parameters
    alpha # learning rate
    lambda # regularization parameter
end

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.theta, s, a)
end

function update!(model::GradientQLearning, s, a, r, s_prime)
    A, gamma, Q, theta, alpha, lambda = 
        model.A, model.gamma, model.Q, model.theta, model.alpha, model.lambda
    u = maximum(Q(θ, s_prime, a_prime) for a_prime in A)
    Δ = (r + gamma*u - Q(theta, s, a)) * model.grad_Q(theta, s, a)
    theta[:] += alpha*(scale_gradient(Δ, 1) - lambda*theta)
    return model
end


#basis(s, a) = [s.x, s.y, s.x*s.y, s.x^2, s.y^2, a == :up, a == :down, a == :left, a == :right]

function basis(s, a)
    state_features = [s.x, s.y, 5 - s.x, 5 - s.y, s.x*s.y, s.x^2, s.y^2]
    action_features = [a == :up, a == :down, a == :left, a == :right]
    # all combinations of state and action features
    combs =  [s_f * a_f for s_f in state_features, a_f in action_features]
    return vcat(state_features, action_features, combs[:])
end

# s = GridWorldState(1, 1, false)
# a = :up
# length(basis(s, a))


Q_func(θ, s, a) = dot(θ, basis(s, a))
grad_Q_func(θ, s, a) = basis(s, a)

# intialize the theta vector with unform random values between -1 and 1
θ = 2 .* rand(39) .- 1
alpha = 0.05
lambda = 0.05

model_gql = GradientQLearning(1:n_actions, 0.95, Q_func, grad_Q_func, θ, alpha, lambda)

simulate(gridworld, model_gql, (m, s) -> π(m, s, exploration), 500_000, s)

#print_Q_table(model_gql)
print_grid(model_gql)




# Vizualization


function visualize_gridworld(gridworld::GridWorld, state::GridWorldState)
    grid = fill(" ", gridworld.size_x, gridworld.size_y)
    grid[gridworld.penalty_zone...] = "P"
    grid[gridworld.size_x, gridworld.size_y] = "G"
    grid[state.x, state.y] = "A"
    
    plot = heatmap(grid, yflip = false, colorbar = false)
    plot
end

function visualize_gridworld(gridworld::GridWorld, model::QLearning)
    grid = fill(" ", gridworld.size_x, gridworld.size_y)
    grid[gridworld.penalty_zone...] = "P"
    grid[gridworld.size_x, gridworld.size_y] = "G"
    
    for state in POMDPs.states(gridworld)
        if !state.done
            values = [model.Q[state_to_index(state, gridworld.size_x, gridworld.size_y), action_to_index(a)] for a in POMDPs.actions(gridworld)]
            optimal_action = index_to_action(argmax(values))
            grid[state.x, state.y] = string(optimal_action)  # Display the first letter of the optimal action
        end
    end
    
    plot = heatmap(grid, yflip = true, colorbar = false)
    plot
end


visualize_gridworld(gridworld, model)
# Test the function
state = POMDPs.initialstate(gridworld)
visualize_gridworld(gridworld, state)