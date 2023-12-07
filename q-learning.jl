using POMDPs
using POMDPModelTools: Deterministic
using Random
using Plots

struct GridWorldState
    x::Int64
    y::Int64
    done::Bool
end

struct GridWorld <: POMDPs.MDP{GridWorldState, Symbol}
    size_x::Int64
    size_y::Int64
    reward_goal::Float64
    reward_penalty::Float64
    penalty_zone::Tuple{Int64, Int64}
end

POMDPs.actions(::GridWorld) = [:up, :down, :left, :right]
POMDPs.states(mdp::GridWorld) = [GridWorldState(x, y, false) for x in 1:mdp.size_x, y in 1:mdp.size_y]
POMDPs.initialstate(mdp::GridWorld) = GridWorldState(1, 1, false)

function POMDPs.gen(m::GridWorld, s::GridWorldState, a::Symbol) #, rng::AbstractRNG)
    if s.done
        return (sp=s, r=0.0)
    end

    x = s.x
    y = s.y

    if a == :up
        y = min(y+1, m.size_y)
    elseif a == :down
        y = max(y-1, 1)
    elseif a == :left
        x = max(x-1, 1)
    elseif a == :right
        x = min(x+1, m.size_x)
    end

    done = (x == m.size_x) && (y == m.size_y)
    reward = done ? m.reward_goal : (x == m.penalty_zone[1] && y == m.penalty_zone[2] ? m.reward_penalty : -1.0)

    return (sp=GridWorldState(x, y, done), r=reward)
end

# Initialize the gridworld
gridworld = GridWorld(5, 5, 10.0, -10.0, (3, 3))

mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # action value function
    alpha # learning rate
end


function state_to_index(state::GridWorldState, size_x::Int64, size_y::Int64)
    return (state.y - 1) * size_x + state.x
end

function state_to_index(state::GridWorldState, size_x::Int64, size_y::Int64)
    return (state.y - 1) * size_x + state.x
end

function action_to_index(action::Symbol)
    return Dict(:up => 1, :down => 2, :left => 3, :right => 4)[action]
end

function index_to_action(index::Int64)
    return [:up, :down, :left, :right][index]
end




function lookahead(model::QLearning, s, a)
    idx_state = state_to_index(s, 5, 5)
    idx_action = action_to_index(a)
    return model.Q[idx_state, idx_action]
end



mutable struct EpsilonGreedyExploration
    epsilon # probability of random action
end


# Define the epsilon-greedy policy
function π(model::QLearning, s, exploration::EpsilonGreedyExploration)
    if rand() < exploration.epsilon
        return rand(POMDPs.actions(gridworld)) # Explore: take a random action
    else
        values = [lookahead(model, s, a) for a in POMDPs.actions(gridworld)]
        return POMDPs.actions(gridworld)[argmax(values)] # Exploit: take the best action
    end
end

function update!(model::QLearning, s, a, r, s_prime)
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    idx_state = state_to_index(s, 5, 5)
    idx_action = action_to_index(a)
    idx_state_prime = state_to_index(s_prime, 5, 5)

    Q[idx_state, idx_action] += alpha*(r + gamma*maximum(Q[idx_state_prime,:]) - Q[idx_state, idx_action])
    return model
end



# h is the horizon

function simulate(P::MDP, model, π, h, s)
    for i in 1:h
        a = π(model, s)
        s_prime, r = POMDPs.gen(P, s, a) #, rng)
        update!(model, s, a, r, s_prime)
        s = s_prime
    end
end


n_states = length(POMDPs.states(gridworld))
n_actions = length(POMDPs.actions(gridworld))
Q = zeros(n_states, n_actions)

model = QLearning(1:n_states, 1:n_actions, 0.95, Q, 0.5)

exploration = EpsilonGreedyExploration(0.1)

s = POMDPs.initialstate(gridworld)

# simulate fuction
simulate(gridworld, model, (m, s) -> π(m, s, exploration), 100_000, s)



function visualize_gridworld(gridworld::GridWorld, state::GridWorldState)
    grid = fill(" ", gridworld.size_x, gridworld.size_y)
    grid[gridworld.penalty_zone...] = "P"
    grid[gridworld.size_x, gridworld.size_y] = "G"
    grid[state.x, state.y] = "A"
    
    plot = heatmap(grid, yflip = true, colorbar = false)
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