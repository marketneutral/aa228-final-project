using DelimitedFiles
using LinearAlgebra
using Plots
using ProgressBars
import Random
using Statistics
using DataFrames
using PrettyTables

default(dpi = 300)

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
        new(committed, 0.0, 0.0, 0.0, 0, 144, 0.40, 2, true)
end


mutable struct investment_pool
    time_step::Int64
    horizon::Int64
    begin_wealth::Float64
    total_wealth::Float64
    bonds::Float64
    stocks::Float64
    private_investments::Array{private_investment,1}
    total_uncalled::Float64
    max_wealth::Float64
    max_drawdown::Float64
    is_bankrupt::Bool
    spent::Float64
    distributed::Float64

    # Define a constructor with default values
    investment_pool() = new(
        1, 240, 100., 100., 40, 60, private_investment[], 0, 100, 0, false, 0, 0
    )
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
    pool.spent += spending
    pool.bonds -= spending
    
    # Step the bonds forward
    pool.bonds *= 1 + mp.bond_return/12
    # Step the stocks forward
    stock_return = mp.stock_expected_return/12 + (mp.stock_volatility/sqrt(12))*randn()
    pool.stocks *= 1 + stock_return

    # Step the private investments forward
    if !isempty(pool.private_investments)
        dist = 0
        for i in 1:length(pool.private_investments)
            if pool.private_investments[i].age >= pool.private_investments[i].max_age
                # distribute the remaining nav
                pool.bonds += pool.private_investments[i].nav
                # set the nav to zero
                pool.private_investments[i].nav = 0
                pool.private_investments[i].is_alive = false
            else
                pri_inv = pool.private_investments[i]
   
                uncalled = pri_inv.committed - pri_inv.called
                if pri_inv.age == 0 || pri_inv.age % 11 == 0
                    call = (pri_inv.rate_of_contrib) * uncalled
                else
                    call = 0
                end

                priv_fund_alpha_return = 
                    mp.privates_expected_alpha/12 +
                    (mp.privates_idiosyncratic_volatility/sqrt(12))*randn()
                g = priv_fund_alpha_return + mp.privates_beta*stock_return

                #g = 0.12/12

                # make distributions
                rd = ((pri_inv.age-1)/(pri_inv.max_age))^pri_inv.bow
                dist = rd * (pri_inv.nav * (1 + g))
                pri_inv.distributed += dist

                pri_inv.called += call
                pri_inv.nav = pri_inv.nav * (1 + g) + call - dist
                pri_inv.age += 1

                # process the call
                pool.bonds -= call
                pool.bonds += dist
            end
        end
        pool.total_uncalled = sum([x.committed - x.called for x in pool.private_investments])
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
    if amount > pool.bonds
        amount = pool.bonds
    end
    push!(pool.private_investments, private_investment(amount))
end



function generate(s::investment_pool, a, mp::market_parameters)
    # process action
    # if a is 1, then do nothing, else

    r = -1
    s_prime = deepcopy(s)

    step!(s_prime, mp)

    if (s_prime.total_wealth < 0) || (s_prime.bonds < 0)
        s_prime.is_bankrupt = true
        r = r - 1_000_000
        return s_prime, r
    end

    # end of game! 
    if s_prime.time_step == s_prime.horizon
        r = r + s_prime.total_wealth - 100
        return s_prime, r
    end

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

    r = r + (s_prime.total_wealth - s.total_wealth)

    if s.bonds/s.total_wealth < 0.15
        r = r - 100_000
    end

    # add neg reward for being in a drawdown 
    current_drawdown = (s_prime.max_wealth - s_prime.total_wealth) / s_prime.max_wealth
    max_drawdown = max(s_prime.max_drawdown, current_drawdown)
    s_prime.max_drawdown = max_drawdown

    return s_prime, r
end


function run_one_path(model, S, π, mp)
    n_months = S.horizon
    per_step_evolution = []
    push!(per_step_evolution, deepcopy(S))

    for i in 1:n_months
        a = π(model, S)
        S_prime, r = generate(S, a, mp)
        S = S_prime
        push!(per_step_evolution, deepcopy(S))
        # check for Bankrupcy
        if S_prime.is_bankrupt
          break
        end
    end
    return per_step_evolution
end

function run_many_paths(model, S, π, mp, n_paths=1000)
    terminal_states = []
    for i in ProgressBar(1:n_paths)
        state = deepcopy(S)
        state = run_one_path(model, state, π, mp)[end]
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
        if uncalled_pct < max_uncalled &&
            bonds_pct > min_cash &&
            privates_pct < max_privates
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
    total_spent = [x.spent for x in one_path]
    private_navs = [sum([x.nav for x in y.private_investments]) for y in one_path]
    distrbutions = [sum([x.distributed for x in y.private_investments]) for y in one_path]
    uncalleds = [sum([x.committed - x.called for x in y.private_investments]) for y in one_path]

    plot(total_wealths, label = "Total Wealth", title = "Single Path", xlabel = "Months", ylabel = "\$")
    plot!(stocks, label = "Stocks")
    plot!(bonds, label = "Bonds")
    plot!(private_navs, label = "Privates Total NAV")
    plot!(uncalleds, label = "Uncalled Capital") 
    plot!(total_spent, label = "Cumulative Spending")
    plot!(distrbutions, label = "Cumulative Distributions")
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
# Summarize path results
#------------------------------------------------------------

function summarize_many_paths(paths)
    avg_wealth = mean([x.total_wealth for x in paths])
    wealth_90 = quantile([x.total_wealth for x in paths], 0.9)
    wealth_10 = quantile([x.total_wealth for x in paths], 0.1)
    avg_max_drawdown = mean([x.max_drawdown for x in paths])
    avg_spent = mean([x.spent for x in paths])
    count_bankrupt = sum([x.is_bankrupt for x in paths])
    n_sims = length(paths)

    df = DataFrame(
        metric = ["N Simulation", "Average Wealth", "Wealth 90th Percentile", "Wealth 10th Percentile", "Average Max Drawdown", "Average Spent", "Times Bankrupt"],
        baseline = [n_sims, avg_wealth, wealth_90, wealth_10, avg_max_drawdown, avg_spent, count_bankrupt],
    )

    pretty_table(df, header=["Metric", "Baseline"])
end




#------------------------------------------------------------
# Market parameters for all simulations
#------------------------------------------------------------

mp = market_parameters(
    0.15,   # stock volatility
    0.08,   # stock expected return
    0.03,   # bond return
    1.0,   # privates beta
    0.025,   # privates expected alpha
    0.05,   # privates idiosyncratic volatility
    0.055   # spending rate
)

# make pretty table of the market_parameters
df = DataFrame(
    parameter = ["Stock Volatility", "Stock Expected Return", "Bond Return",  "Privates Expected Alpha", "Privates Idiosyncratic Volatility", "Spending Rate"],
    value = [mp.stock_volatility, mp.stock_expected_return, mp.bond_return, mp.privates_expected_alpha, mp.privates_idiosyncratic_volatility, mp.spending_rate],
)
pretty_table(df, header=["Parameter", "Value"])

#------------------------------------------------------------
# Heuristic model, no privates
#------------------------------------------------------------

model = Heuristics(
    0.00,   # max privates
    0.75,   # max equity net
    0.25,   # max uncalled
    0.70,   # target equity net
    0.05,   # rebalance band
    99999,      # pacing months
    0.05    # min cash
)


Random.seed!(42)


S = investment_pool()
one_path = run_one_path(model, S, π, mp);

# plot one_path path and save to file
plot_one_path(one_path)
savefig("one_path.png")

paths = run_many_paths(model, S, π, mp, 500);
summarize_many_paths(paths)



#------------------------------------------------------------
# Heuristic model, with privates
#------------------------------------------------------------

model = Heuristics(
    0.25,   # max privates
    0.75,   # max equity net
    0.25,   # max uncalled
    0.70,   # target equity net
    0.05,   # rebalance band
    6,      # pacing months
    0.05    # min cash
)


Random.seed!(42)

S = investment_pool()
one_path = run_one_path(model, S, π, mp);

# plot one_path path and save to file
plot_one_path(one_path)
savefig("one_path.png")

paths = run_many_paths(model, S, π, mp, 500);
summarize_many_paths(paths)

plot_private_investment(one_path, 4)
savefig("one_pi.png")





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
    u_ = [Q(theta, s_prime, a_prime) for a_prime in A]
    u = maximum(u_)
    Δ = (r + gamma*u - Q(theta, s, a)) * model.grad_Q(theta, s, a) - lambda*theta
    theta[:] += alpha*scale_gradient(Δ, 10)
    return model
end


function simulate(model, π, s, gen_func)
    h = s.horizon
    per_step_evolution = []
    per_step_reward = []
    push!(per_step_evolution, deepcopy(s))
    for i in 1:h
        a = π(model, s)
        s_prime, r = gen_func(s, a)
        update!(model, s, a, r, s_prime)
        s = s_prime
        push!(per_step_evolution, deepcopy(s))
        push!(per_step_reward, r)
        if s.is_bankrupt
            break
        end
    end
    return per_step_evolution, per_step_reward
end




mutable struct EpsilonGreedyExploration
    epsilon # probability of random action
end


function π(model::GradientQLearning, s, exploration::EpsilonGreedyExploration)
    A = model.A
    if rand() < exploration.epsilon
        action = rand(A)
    else
        values = [lookahead(model, s, a) for a in A]
        action = A[argmax(values)] # Exploit: take the best action
    end
    return action
end


function Q_func(θ, s, a)
    q_val = dot(θ, basis(s, a))
    if isnan(q_val)
        println("NaN in Q_func!")
        println("    $s")
    end
    return dot(θ, basis(s, a))
end

grad_Q_func(θ, s, a) = basis(s, a)


#------------------------------------------------------------
# Problem specific Gradient Q-learning
#------------------------------------------------------------

action_space = [
    :do_nothing,
    :buy_stocks_5,
    :sell_stocks_5,
    :commit_5,
]


function basis(s, a)
    cosine_transform = cos((2.0 * pi / 12) * (s.time_step .- 12))
    sin_transform = sin((2.0 * pi / 12) * (s.time_step .- 12))

    total_wealth = s.total_wealth + 1e-6
    bonds = max(s.bonds, 1e-6)

    bonds_pct = bonds / total_wealth

    state_features = [
        (total_wealth/s.begin_wealth) - 1,
        s.stocks/total_wealth,
        bonds_pct,
        s.total_uncalled/total_wealth,
        s.total_uncalled/bonds,
        (total_wealth - s.stocks - bonds)/total_wealth,
        s.max_drawdown,
        cosine_transform,
        sin_transform,
    ]

    # polynomial features
    p2 = [s_f^2 for s_f in state_features]
    p3 = [s_f^3 for s_f in state_features]
    p4 = [s_f^4 for s_f in state_features]

    state_features = vcat(bonds_pct < 0.10, state_features, p2, p3, p4)
    
    action_OHE_features = [a == action for action in action_space]

    # all combinations of state and action features
    combs =  [s_f * a_f for s_f in state_features, a_f in action_OHE_features]

    feature_vector = vcat(action_OHE_features, combs[:])

    # if the feature vector contains any NaNs, return a vector of zeros
    if any(isnan.(feature_vector))
       println("NaNs in feature vector!")
    end

    return feature_vector
end



s = investment_pool()
bf = basis(s, :do_nothing);
n_features = length(bf)

Random.seed!(42)

θ = 2 .* (rand(n_features) .- 0.5);
alpha = 0.01
lambda = 0.1


model_gql = GradientQLearning(
    action_space,
    0.999,
    Q_func,
    grad_Q_func,
    θ,
    alpha,
    lambda
);

exploration = EpsilonGreedyExploration(0.2)

s = investment_pool()
one_path_gql, rewards = simulate(model_gql, (m, s) -> π(m, s, exploration), s, (s, a) -> generate(s, a, mp));
plot_one_path(one_path_gql)
plot(rewards, label = "Reward", title = "Single Path", xlabel = "Months", ylabel = "Reward")


terminal_states = []
rewards_lengths = []
thetas = []
push!(thetas, deepcopy(model_gql.theta));
for i in ProgressBar(1:1_000)
    s = investment_pool()
    one_path, rewards = simulate(model_gql, (m, s) -> π(m, s, exploration), s, (s, a) -> generate(s, a, mp));
    if i % 250 == 0
        push!(terminal_states, deepcopy(one_path[end]))
        push!(rewards_lengths, length(rewards))    
    end
    if i % 2500 == 0
        # save checkpoint
        writedlm("theta_checkpoint_$i.csv", model_gql.theta, ',')
        push!(thetas, deepcopy(model_gql.theta))
    end
end

plot(rewards_lengths, legend=false, title = "Time Before Bankrupcy", xlabel = "Record", ylabel = "Reward")
# write theta vector to file
writedlm("theta.csv", model_gql.theta, ',')

# plot thetas
plot([norm(x) for x in diff(thetas)], legend=false, title = "diff(thetas) Norm", xlabel = "Record", ylabel = "Norm")


# lets look at the q-values for each action for a given state
s = one_path[1];
Q_values = [lookahead(model_gql, s, a) for a in action_space]

S = investment_pool()
greedy = EpsilonGreedyExploration(0.0)
path = run_one_path(model_gql, S, (m, s) -> π(m, s, greedy), mp);
plot_one_path(path)

s = path[8];
Q_values = Dict(a => lookahead(model_gql, s, a) for a in action_space)

paths = run_many_paths(model_gql, S, (m, s) -> π(m, s, greedy), mp, 500);
summarize_many_paths(paths)
writedlm("theta.csv", model_gql.theta, ',')
