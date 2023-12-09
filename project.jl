using Plots
using ProgressBars
import Random

Random.seed!(17)

# -----------------------------------------------------------------
# Problem domain defintion
# -----------------------------------------------------------------

mutable struct private_investment
    committed::Float64
    called::Float64
    distributed::Float64
    nav::Float64
    age::Int64
    max_age::Int64
    rate_of_contrib::Float64
    bow::Float64
    is_alive::Bool
    private_investment(committed::Float64) = 
        new(committed, 0.0, 0.0, 0.0, 0, 120, 0.25, 3, true)
end

mutable struct investment_pool
    time_step::Int64
    total_wealth::Float64
    bonds::Float64
    stocks::Float64
    private_investments::Array{private_investment,1}
    max_wealth::Float64
    max_drawdown::Float64
    is_bankrupt::Bool
    spent::Float64
    distributed::Float64
    # Define a constructor with default values
    investment_pool() = new(1, 100., 40, 60, private_investment[], 100, 0, false, 0, 0)
end

struct market_parameters
    stock_volatility::Float64
    stock_expected_return::Float64
    bond_return::Float64
    privates_beta::Float64
    privates_expected_alpha::Float64
    privates_idiosyncratic_volatility::Float64
    spending_rate::Float64
end


function step!(pool::investment_pool, mp::market_parameters)
    # make the spending distribution
    spending = pool.total_wealth * mp.spending_rate/12
    pool.spent = spending
    pool.bonds -= spending
    
    # Step the bonds forward
    pool.bonds *= 1 + mp.bond_return/12
    # Step the stocks forward
    stock_return = mp.stock_expected_return/12 + (mp.stock_volatility/sqrt(12))*randn()
    pool.stocks *= 1 + stock_return

    # Step the private investments forward
    if !isempty(pool.private_investments)
        dist = 0
        for i in length(pool.private_investments):-1:1
            if pool.private_investments[i].age >= pool.private_investments[i].max_age
                # distribute the remaining nav
                pool.bonds += pool.private_investments[i].nav
                # set the nav to zero
                pool.private_investments[i].nav = 0
                pool.private_investments[i].is_alive = false
            else
                pool.private_investments[i].age += 1
                # Step the private investment forward
                priv_fund_alpha_return = 
                    mp.privates_expected_alpha/12 +
                    (mp.privates_idiosyncratic_volatility/sqrt(12))*randn()
                pool.private_investments[i].nav *= (1 + mp.privates_beta*stock_return + priv_fund_alpha_return)

                # make distributions
                rd = (pool.private_investments[i].age/(pool.private_investments[i].max_age))^pool.private_investments[i].bow
                dist = rd * pool.private_investments[i].nav
                pool.private_investments[i].distributed += dist
                pool.bonds += dist
                dist += dist

                # make calls
                uncalled = pool.private_investments[i].committed - pool.private_investments[i].called
                call = (pool.private_investments[i].rate_of_contrib/12) * uncalled
                pool.private_investments[i].called += call
                pool.bonds -= call

                pool.private_investments[i].nav += (call - dist)
            end
        end
        pool.distributed = dist
    end
    
    # calculate total wealth
    total_wealth = pool.bonds + pool.stocks + sum([x.nav for x in pool.private_investments])
    pool.total_wealth = total_wealth
    pool.max_wealth = max(pool.max_wealth, total_wealth)
    pool.time_step += 1
end



function buy_stocks!(pool::investment_pool, amount::Float64)
    # If there is not enough cash, return
    if amount > pool.bonds
        amount = pool.bonds
    end
    # Buy stocks
    pool.stocks += amount
    # Sell bonds
    pool.bonds -= amount
end


function sell_stocks!(pool::investment_pool, amount::Float64)
    # If there are not enough stocks, return
    if amount > pool.stocks
        amount = pool.stocks
    end
    # Sell stocks
    pool.stocks -= amount
    # Buy bonds
    pool.bonds += amount
end


function commit!(pool::investment_pool, amount::Float64)
    # Commit to the private investment
    push!(pool.private_investments, private_investment(amount))
end


actions_space = [
    :do_nothing,
    :buy_stocks_5,
    :sell_stocks_5,
    :commit_5,
]


function generate(s::investment_pool, a, mp::market_parameters)
    # process action
    # if a is 1, then do nothing, else

    s_prime = deepcopy(s)
    if s_prime.is_bankrupt
        return s_prime, 0
    end

    if a == :buy_stocks_5
        trade_size = s_prime.total_wealth * 0.05
        buy_stocks!(s_prime, trade_size)
    end
    
    if a == :sell_stocks_5
        trade_size = s_prime.total_wealth * 0.05
        sell_stocks!(s_prime, trade_size)
    end

    if a == :commit_5
        commitment_size = s_prime.total_wealth * 0.05
        commit!(s_prime, commitment_size)
    end

    step!(s_prime, mp)

    # log
    # push!(endowments, deepcopy(endowment))

    # calculate reward
    r = s_prime.total_wealth - s.total_wealth
    if (s_prime.total_wealth < 0) || (s_prime.bonds < 0)
        s_prime.is_bankrupt = true
        r = r - 1000
        return s_prime, r
    end

    # add neg reward for being in a drawdown 
    current_drawdown = (s_prime.max_wealth - s_prime.total_wealth) / s_prime.max_wealth
    r = r - 10 * current_drawdown
    max_drawdown = max(s_prime.max_drawdown, current_drawdown)
    s_prime.max_drawdown = max_drawdown

    return s_prime, r
end


function run_one_path(model, S, π, mp, n_months=240)
    per_step_evolution = []
    push!(per_step_evolution, deepcopy(S))

    for i in 1:n_months
        a = π(model, S)
        S_prime, r = generate(S, a, mp)
        S = S_prime
        push!(per_step_evolution, deepcopy(S))
    end
    return per_step_evolution
end

function run_many_paths(model, S, π, mp, n_paths=1000, n_months=240)
    terminal_states = []
    for i in ProgressBar(1:n_paths)
        state = deepcopy(S)
        state = run_one_path(model, state, π, mp, n_months)[end]
        push!(terminal_states, deepcopy(state))
    end
    return terminal_states
end



#------------------------------------------------------------
# Define rules-based baseline
#------------------------------------------------------------

struct Heuristics
    max_privates::Float64
    max_equity_net::Float64
    max_uncalled::Float64
    target_equity_net::Float64
    rebalance_band::Float64
    pacing_months::Int64
    min_cash::Float64
end

function π(model::Heuristics, s)
    if s.is_bankrupt
        return :do_nothing
    end

    max_privates = model.max_privates
    target_equity_net = model.target_equity_net
    rebalance_band = model.rebalance_band
    pacing_months = model.pacing_months
    max_uncalled = model.max_uncalled
    min_cash = model.min_cash

    bonds_pct = s.bonds / s.total_wealth
    privates_nav = sum([p.nav for p in s.private_investments])
    uncalled_total = sum([p.committed - p.called for p in s.private_investments])
    stocks_pct = s.stocks / s.total_wealth
    uncalled_pct = uncalled_total / s.total_wealth
    privates_pct = privates_nav / s.total_wealth

    a = :do_nothing

    if stocks_pct + privates_pct - target_equity_net > rebalance_band
        return :sell_stocks_5
    end

    if stocks_pct + privates_pct - target_equity_net < -rebalance_band
        return :buy_stocks_5
    end

    if s.time_step % pacing_months == 0
        if uncalled_pct < max_uncalled && bonds_pct > min_cash && privates_pct < max_privates
            return :commit_5
        end
    end

    return a
end

#------------------------------------------------------------
# Visualize results
#------------------------------------------------------------

function plot_one_path(one_path)
    total_wealths = [x.total_wealth for x in one_path]
    stocks = [x.stocks for x in one_path]
    bonds = [x.bonds for x in one_path]
    total_spent = cumsum([x.spent for x in one_path])
    total_distributed = cumsum([x.distributed for x in one_path])
    private_navs = [sum([x.nav for x in y.private_investments]) for y in one_path]
    uncalleds = [sum([x.committed - x.called for x in y.private_investments]) for y in one_path]

    plot(total_wealths, label = "Total Wealth", title = "Investment Paths", xlabel = "Months", ylabel = "\$")
    plot!(stocks, label = "Stocks")
    plot!(bonds, label = "Bonds")
    plot!(private_navs, label = "Privates Total NAV")
    plot!(uncalleds, label = "Uncalled Capital") 
    plot!(total_spent, label = "Cumulative Spending")
    plot!(total_distributed, label = "Cumulative Distributions")
end


function plot_private_investment(one_path, private_investment_index)
    committeds = [
        e.private_investments[private_investment_index].committed 
        for e in one_path 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    calleds = [
        e.private_investments[private_investment_index].called 
        for e in one_path 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    navs = [
        e.private_investments[private_investment_index].nav 
        for e in one_path 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    distributions = [
        e.private_investments[private_investment_index].distributed 
        for e in one_path 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]

    plot(committeds, label = "Committed", title = "Private Investment", xlabel = "Months", ylabel = "\$")
    plot!(calleds, label = "Called")
    plot!(navs, label = "NAV")
    plot!(distributions, label = "Distributions")
end





#------------------------------------------------------------
# Apply heuristic policy in simulation
#------------------------------------------------------------

mp = market_parameters(
    0.17,   # stock volatility
    0.08,   # stock expected return
    0.03,   # bond return
    1.0,   # privates beta
    0.05,   # privates expected alpha
    0.10,   # privates idiosyncratic volatility
    0.0   # spending rate
)

model = Heuristics(
    0.25,   # max privates
    0.70,   # max equity net
    0.25,   # max uncalled
    0.60,   # target equity net
    0.05,   # rebalance band
    3,      # pacing months
    0.05    # min cash
)

S = investment_pool()
one_path = run_one_path(model, S, π, mp, 240);

# plot one_path path and save to file
plot_one_path(one_path)
savefig("one_path.png")

plot_private_investment(one_path, 3)

paths = run_many_paths(model, S, π, mp, 100, 240);




#------------------------------------------------------------
# Summarize path results
#------------------------------------------------------------






#------------------------------------------------------------
# Learn Q* via gradient-based Q-learning with function approximation
#------------------------------------------------------------

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
    Δ = (r + gamma*u - Q(theta, s, a)) * model.grad_Q(theta, s, a) - lambda*theta
    theta[:] += alpha*scale_gradient(Δ, 1)
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
        return rand(A) # Explore: take a random action; constrain to valid actions
    else
        values = [lookahead(model, s, a) for a in A]
        return A[argmax(values)] # Exploit: take the best action
    end
end



function basis(s, a)
    bias = [1.0]
    state_features = [1.0] # [s.i, s.j, 5 - s.i, 5 - s.j, s.i*s.j, s.i^2, s.j^2]
    action_features = [a == :up, a == :down, a == :left, a == :right]
    # all combinations of state and action features
    combs =  [s_f * a_f for s_f in state_features, a_f in action_features]
    return vcat(bias, state_features, action_features, combs[:])
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

