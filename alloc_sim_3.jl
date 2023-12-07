using Plots

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
        new(committed, 0.0, 0.0, 0.0, 0, 120, 0.25, 2.5, true)
end

mutable struct investment_pool
    total_wealth::Float64
    bonds::Float64
    stocks::Float64
    private_investments::Array{private_investment,1}
    max_wealth::Float64
    bankrupt::Bool
    # Define a constructor with default values
    investment_pool() = new(100., 30, 70, private_investment[], 100, false)
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


"""
Function which steps the investment pool forward one month:
"""
function step!(pool::investment_pool, mp::market_parameters)
    # make the spending distribution
    spending = pool.total_wealth * mp.spending_rate/12
    pool.bonds -= spending
    
    # Step the bonds forward
    pool.bonds *= 1 + mp.bond_return/12
    # Step the stocks forward
    stock_return = mp.stock_expected_return/12 + (mp.stock_volatility/sqrt(12))*randn()
    pool.stocks *= 1 + stock_return

    # Step the private investments forward
    if !isempty(pool.private_investments)
        # reveresed so that we can splice out private investments
        for i in length(pool.private_investments):-1:1
            # If the private investment is past its max age, remove it
            if pool.private_investments[i].age >= pool.private_investments[i].max_age
                # distribute the remaining nav
                pool.bonds += pool.private_investments[i].nav
                # set the nav to zero
                pool.private_investments[i].nav = 0
                pool.private_investments[i].is_alive = false
            else
                pool.private_investments[i].age += 1
                # Step the private investment forward
                priv_fund_alpha_return = mp.privates_expected_alpha/12 + (mp.privates_idiosyncratic_volatility/sqrt(12))*randn()
                pool.private_investments[i].nav *= 1 + mp.privates_beta*stock_return + priv_fund_alpha_return

                # make distributions
                rd = (pool.private_investments[i].age/(pool.private_investments[i].max_age))^pool.private_investments[i].bow
                dist = rd * pool.private_investments[i].nav
                pool.private_investments[i].distributed += dist
                pool.bonds += dist
                
                # make calls
                uncalled = pool.private_investments[i].committed - pool.private_investments[i].called
                call = (pool.private_investments[i].rate_of_contrib/12) * uncalled
                pool.private_investments[i].called += call
                pool.bonds -= call

                pool.private_investments[i].nav += (call - dist)
            end
        end
    end
    
    # calculate total wealth
    total_wealth = pool.bonds + pool.stocks + sum([x.nav for x in pool.private_investments])
    pool.total_wealth = total_wealth
    pool.max_wealth = max(pool.max_wealth, total_wealth)
end

"""
Function to buy stocks and sell bonds
"""
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


"""
Function to sell stocks and buy bonds
"""
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


"""
Function to commit to a private investment
"""
function commit!(pool::investment_pool, amount::Float64)
    # Commit to the private investment
    push!(pool.private_investments, private_investment(amount))
end


# Simulate the baseline heuristic policy

# equity_risks = zeros(N_MONTHS)
# stocks = zeros(N_MONTHS)
# bonds = zeros(N_MONTHS)
# private_commitments = zeros(N_MONTHS)
# total_wealths = zeros(N_MONTHS)
# private_navs = zeros(N_MONTHS)
# distributions = zeros(N_MONTHS)
# uncalleds = zeros(N_MONTHS)


mutable struct StateSpace
    time_step::Int64
    total_wealth::Int64
    equity_pct::Int64
    bonds_pct::Int64
    privates_pct::Int64
    uncalled_pct::Int64
    wgt_avg_age_of_privates_nav::Int64
    current_drawdown::Int64
    StateSpace() = new(0, 0, 0, 0, 0, 0, 0, 0)
end

A = (0, 1, 2, 3, 4, 5, 6)


# chapter 17, Model-Free Methods
mutable struct ModelFreeProblem
    S::StateSpace
    A # action space (assumes 1:nactions)
    gamma::Float64 # discount
    Q # action value function
    alpha::Float64 # learning rate
end
    


# A: actions
#  1: do nothing
#  2: buy stocks, sell bonds 1
#  3: buy stocks, sell bonds 5
#  4: sell stocks, buy bonds 1
#  5: sell stocks, buy bonds 5   
#  6: commit to private investment 1
#  7: commit to a private investment 5


function discretize(endowment::investment_pool)::StateSpace
    states = StateSpace()
    states.total_wealth = round(endowment.total_wealth)
    states.equity_pct = round(endowment.stocks/endowment.total_wealth * 100)
    states.bonds_pct = round(endowment.bonds/endowment.total_wealth * 100)
    states.privates_pct = round(sum([x.nav for x in endowment.private_investments])/endowment.total_wealth * 100)
    states.uncalled_pct = round(
        sum([x.committed - x.called for x in endowment.private_investments if x.is_alive])
        /endowment.total_wealth * 100
    )

    if isempty(endowment.private_investments)
        states.wgt_avg_age_of_privates_nav = 0
    else
        states.wgt_avg_age_of_privates_nav = round(sum([x.age * x.nav for x in endowment.private_investments])/sum([x.nav for x in endowment.private_investments]))
    end

    states.current_drawdown = round(endowment.max_wealth - endowment.total_wealth)
    return states
end


function baseline_policy(S::StateSpace)

    a = 1   # default, do nothing

    if S.equity_pct + S.privates_pct - 80 > 1
        a = 4   # sell stocks, buy bonds 1%
    end

    if S.equity_pct + S.privates_pct - 80 > 5
        a = 5   # sell stocks, buy bonds 5%
    end

    if S.equity_pct + S.privates_pct - 70 < -1
        a = 2   # buy stocks, sell bonds 1%
    end

    if S.equity_pct + S.privates_pct - 70 < -5
        a = 3   # buy stocks, sell bonds 5%
    end

    # every 3 months, we can commit to a private investment
    if S.time_step % 3 == 0
        if S.uncalled_pct < 25 && S.bonds_pct > 5 && S.privates_pct < 25
            a = 7   # commit to private investment 5%
        end
    end

    return a
end


function run_one_sim(endowment, mp, N_MONTHS)
    S = StateSpace()
    S_prime = StateSpace()
    actions = []    
    endowments = []

    for i in 1:N_MONTHS
        S = discretize(endowment)
        S.time_step = i

        a = baseline_policy(S)
        push!(actions, a)

        # process action

        # if a is 1, then do nothing, else

        if a == 2
            trade_size = endowment.total_wealth * 0.025
            buy_stocks!(endowment, trade_size)
        end
        
        if a == 3
            trade_size = endowment.total_wealth * 0.10
            buy_stocks!(endowment, trade_size)
        end

        if a == 4
            trade_size = endowment.total_wealth * 0.025
            sell_stocks!(endowment, trade_size)
        end

        if a == 5
            trade_size = endowment.total_wealth * 0.10
            sell_stocks!(endowment, trade_size)
        end

        if a == 6
            commitment_size = endowment.total_wealth * 0.025
            commit!(endowment, commitment_size)
        end

        if a == 7
            commitment_size = endowment.total_wealth * 0.10
            commit!(endowment, commitment_size)
        end


        # starting_wealth = endowment.total_wealth

        step!(endowment, mp)
        push!(endowments, deepcopy(endowment))

        S_prime = discretize(endowment)
        S_prime.time_step = i + 1

        # gain_loss = endowment.total_wealth - starting_wealth

    end

    return actions, endowments
end


N_MONTHS = 240

# instantiate market parameters
mp = market_parameters(
    0.17,   # stock volatility
    0.08,   # stock expected return
    0.025,   # bond return
    1.0,   # privates beta
    0.08,   # privates expected alpha
    0.15,   # privates idiosyncratic volatility
    0.055   # spending rate
)


endowment = investment_pool()
actions, endowments = run_one_sim(endowment, mp, N_MONTHS)
println(endowments[end])

function plot_paths(endowments)
    total_wealths = [x.total_wealth for x in endowments]
    stocks = [x.stocks for x in endowments]
    bonds = [x.bonds for x in endowments]
    private_navs = [sum([x.nav for x in y.private_investments]) for y in endowments]
    uncalleds = [sum([x.committed - x.called for x in y.private_investments]) for y in endowments]

    plot(total_wealths, label = "Total Wealth", title = "Investment Paths")
    plot!(stocks, label = "Stocks")
    plot!(bonds, label = "Bonds")
    plot!(private_navs, label = "Private NAVs")
    plot!(uncalleds, label = "Uncalled Capital") 
end

function plot_paths_normalized(endowments)
    total_wealths = [x.total_wealth for x in endowments]
    stocks = [(x.stocks / total_wealths[i]) * 100 for (i, x) in enumerate(endowments)]
    bonds = [(x.bonds / total_wealths[i]) * 100 for (i, x) in enumerate(endowments)]
    private_navs = [(sum([x.nav for x in y.private_investments]) / total_wealths[i]) * 100 for (i, y) in enumerate(endowments)]
    uncalleds = [(sum([x.committed - x.called for x in y.private_investments]) / total_wealths[i]) * 100 for (i, y) in enumerate(endowments)]

    plot(stocks, label = "Stocks (%)")
    plot!(bonds, label = "Bonds (%)")
    plot!(private_navs, label = "Private NAVs (%)")
    plot!(uncalleds, label = "Uncalled Capital (%)") 
end

# TODO: the path doesn't seem right here; check geometric brownian motion
# TODO: when we commit nothing to privates we do the best; does't make sense
plot_paths(endowments)
plot_paths_normalized(endowments)


function plot_wgt_avg_age_of_privates(endowments)
    ages = [sum([y.age * y.nav for y in x.private_investments])/sum([y.nav for y in x.private_investments]) for x in endowments]
    plot(ages, label = "Weighted Average Age of Private NAVs", title = "Weighted Average Age of Private NAVs")
end

plot_wgt_avg_age_of_privates(endowments)

function plot_private_investment(endowments, private_investment_index)
    committeds = [
        e.private_investments[private_investment_index].committed 
        for e in endowments 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    calleds = [
        e.private_investments[private_investment_index].called 
        for e in endowments 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    navs = [
        e.private_investments[private_investment_index].nav 
        for e in endowments 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    distributions = [
        e.private_investments[private_investment_index].distributed 
        for e in endowments 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]

    plot(committeds, label = "Committed", title = "Private Investment")
    plot!(calleds, label = "Called")
    plot!(navs, label = "NAV")
    plot!(distributions, label = "Distributions")
end

plot_private_investment(endowments, 1)
   
"""
for a given private investment, plot the difference between the total called in this time step and the prior time step
"""
function plot_private_investment_calls(endowments, private_investment_index)
    calleds = [
        e.private_investments[private_investment_index].called 
        for e in endowments 
        if !isempty(e.private_investments) && length(e.private_investments) >= private_investment_index && e.private_investments[private_investment_index].age < e.private_investments[private_investment_index].max_age
    ]
    calleds = [0; diff(calleds)]
    plot(calleds, label = "Called", title = "Private Investment Calls")
end

# TODO: this shows that the full amount never gets called
# because it looks like each year the call is 10% of the uncalled capital
plot_private_investment_calls(endowments, 1)



"""
this function plots the actions taken by the baseline policy
the actions are labeled on the y axis, and there is a dot for each time step
"""
function plot_actions(actions)
    plot(actions, seriestype = :scatter, legend=false, markershape = :square, markersize = 2, title = "Actions")
end



plot_actions(actions)