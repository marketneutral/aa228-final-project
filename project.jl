# Gradient-based Q-learning
# General code

struct GradientQLearning
    A # discrete action space
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

function simulate(model, π, h, s)
    for i in 1:h
        a = π(model, s)
        s_prime, r = generate(s, a)
        update!(model, s, a, r, s_prime)
        s = s_prime
    end
end

mutable struct EpsilonGreedyExploration
    epsilon # probability of random action
end


function π(::GradientQLearning, s, exploration::EpsilonGreedyExploration)
    A = model.A
    if rand() < exploration.epsilon
        return rand(A) # Explore: take a random action
    else
        values = [lookahead(model, s, a) for a in A]
        return A[argmax(values)] # Exploit: take the best action
    end
end



# Problem-specific code



struct Heuristic

end

function π(::Heuristic, s)
    if s == 1
        return 1
    else
        return 2
    end
end






function generate(s, a)
    s_prime = s + a
    r = 0
    if s_prime == 10
        r = 1
    end
    return s_prime, r
end

function basis(s, a)
    return []
end

exploration = EpsilonGreedyExploration(0.1)

Q_func(θ, s, a) = dot(θ, basis(s, a))
grad_Q_func(θ, s, a) = basis(s, a)

θ = 2 .* rand(39) .- 1
alpha = 0.05
lambda = 0.05

model_gql = GradientQLearning(
    1:n_actions,
    0.95,
    Q_func,
    grad_Q_func,
    θ,
    alpha,
    lambda
)


