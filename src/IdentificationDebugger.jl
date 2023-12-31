"""
Placeholder for a short summary about IdentificationDebugger.
"""
module IdentificationDebugger

export known_parameter, endogeneous_parameter, identification_problem, check_identification

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using FillArrays: Fill
using ADNLPModels: ADNLSModel
using JSOSolvers: tron
using Logging: @error
using Percival: percival
using ObjConsNLPModels: objcons_nlpmodel
using TransformVariables: dimension, transform, inverse, as, asℝ
using Random: default_rng
using Statistics: mean

###
### parameters
###

Base.@kwdef struct Parameter{TK,TT,TL,TU}
    known_value::TK
    transformation::TT
    lower_bound::TL
    upper_bound::TU
    is_endogeneous::Bool
end

_default_transformation(a::Real) = asℝ

_default_transformation(a::AbstractArray) = as(Array, size(a)...)

_default_lower_bound(transformation) = fill(-Inf, dimension(transformation))

_default_upper_bound(transformation) = fill(Inf, dimension(transformation))

"""
$(SIGNATURES)

Description of a known parameter. The exercise is to recover the `known_value` in a
simulated setting.

# Keyword parameters

- `transformation`: a transformation from ``ℝⁿ``, if applicable, implicitly also provides
  the dimension `n`. Defaults to the identity, reshaped.
- `lower_bound`, `upper_bound`: lower- and upper bound in ``ℝⁿ`` for the estimation
"""
function known_parameter(known_value;
                   transformation = _default_transformation(known_value),
                   lower_bound::AbstractVector = _default_lower_bound(transformation),
                   upper_bound::AbstractVector = _default_upper_bound(transformation))
    @argcheck dimension(transformation) == length(lower_bound) == length(upper_bound)
    x = inverse(transformation, known_value)
    @argcheck all(lower_bound .≤ x .≤ upper_bound)
    Parameter(; known_value, transformation, lower_bound, upper_bound,
              is_endogeneous = false)
end

"""
$(SIGNATURES)

An endogeneous parameter. Will be solved for when the problem is created.
"""
function endogeneous_parameter(transformation;
                               lower_bound::AbstractVector = _default_lower_bound(transformation),
                               upper_bound::AbstractVector = _default_upper_bound(transformation))
    Parameter(; known_value = fill(NaN, dimension(transformation)),
              transformation, lower_bound, upper_bound, is_endogeneous = true)
end

###
### objective
###

struct LeastSquaresObjective{T<:NamedTuple}
    target_moments::T
end

ls_diff(a::Real, b::Real) = abs2(a - b)
ls_diff(a::AbstractArray, b::AbstractArray) = mapreduce(ls_diff, +, a, b)
ls_diff(a::NamedTuple{T}, b::NamedTuple{T}) where T = mapreduce(ls_diff, +, a, b)

function (lso::LeastSquaresObjective)(model_moments::NamedTuple{TT}) where TT
    (; target_moments) = lso
    m = NamedTuple{TT}(model_moments)       # pick the variables we want, in the same order
    ls_diff(m, target_moments)
end

####
#### identification problem framework
####

mean_abs2(x::AbstractVector, y::AbstractVector) = mean(((x, y),) -> abs2(x - y), zip(x, y))

"The type we accept for return values of `moments_and_constraint_calculator`."
const MC_TYPE = Tuple{NamedTuple,AbstractVector}

Base.@kwdef struct IdentificationProblem{TP<:NamedTuple,TM,TO}
    parameters::TP
    moments_and_constraint_calculator::TM
    constraint_dimension::Int = 0
    objective::TO = nothing
    solution_tol::Float64 = 1e-4
end

function classify_parameters(problem::IdentificationProblem)
    endogeneous = Symbol[]
    exogeneous = Symbol[]
    for (name, parameter) in pairs(problem.parameters)
        push!(parameter.is_endogeneous ? endogeneous : exogeneous, name)
    end
    (endogeneous = Tuple(endogeneous), exogeneous = Tuple(exogeneous))
end

function free_parameters(problem::IdentificationProblem, free_variables::Val{S}) where S
    NamedTuple{S}(problem.parameters)
end

function free_lower_bound(problem::IdentificationProblem, free_variables::Val{S}) where S
    mapreduce(p -> p.lower_bound, vcat, free_parameters(problem, free_variables))
end

function free_upper_bound(problem::IdentificationProblem, free_variables::Val{S}) where S
    mapreduce(p -> p.upper_bound, vcat, free_parameters(problem, free_variables))
end

function free_dimension(problem::IdentificationProblem, free_variables::Val{S}) where S
    mapreduce(p -> dimension(p.transformation), +, free_parameters(problem, free_variables))
end

function free_transformation(problem::IdentificationProblem, free_variables::Val{S}) where S
    as(map(p -> p.transformation, free_parameters(problem, free_variables)))
end

function known_values(problem::IdentificationProblem, free_variables::Val{S}) where S
    map(p -> p.known_value, Base.structdiff(problem.parameters, NamedTuple{S}))
end

"""
$(SIGNATURES)
"""
function solve_endogeneous_parameters!(problem::IdentificationProblem,
                                      endogeneous_variables::Val{S}) where S
    isempty(S) && return problem
    (; moments_and_constraint_calculator, parameters, solution_tol) = problem
    N = free_dimension(problem, endogeneous_variables)
    transformation = free_transformation(problem, endogeneous_variables)
    lb = free_lower_bound(problem, endogeneous_variables)
    ub = free_upper_bound(problem, endogeneous_variables)
    κ = known_values(problem, endogeneous_variables)
    function F(x)
        θ = transform(transformation, x)
        (moments_and_constraint_calculator(merge(κ, θ))::MC_TYPE)[2]
    end
    x0 = zeros(N)
    PF = ADNLSModel(F, x0, length(F(x0)), lb, ub)
    stats = tron(PF; Fatol = solution_tol)
    @argcheck stats.solution_reliable
    θ = transform(transformation, stats.solution)
    for (k, v) in pairs(θ)
        getproperty(parameters, k).known_value .= v
    end
    nothing
end

"$(SIGNATURES)"
function identification_problem(moments_and_constraint_calculator,
                                parameters::NamedTuple; solution_tol = 1e-4)
    @argcheck solution_tol > 0
    parameters = deepcopy(parameters) # don't modify caller's variables
    problem0 = IdentificationProblem(; parameters, moments_and_constraint_calculator)
    solve_endogeneous_parameters!(problem0,
                                  Val(classify_parameters(problem0).endogeneous))
    moments, constraint = moments_and_constraint_calculator(map(p -> p.known_value,
                                                                problem0.parameters))::MC_TYPE
    @argcheck all(abs.(constraint) .≤ solution_tol)
    objective = LeastSquaresObjective(moments)
    IdentificationProblem(; parameters, moments_and_constraint_calculator, objective,
                          solution_tol, constraint_dimension = length(constraint))
end

###
### partial problem we pass to optimizers
###

struct PartialProblem{S,TP,TT}
    parent_problem::TP
    transformation::TT
    function PartialProblem{S}(parent_problem::TP, transformation::TT) where {S,TP,TT}
        new{S,TP,TT}(parent_problem, transformation)
    end
end

"$(SIGNATURES)"
function partial_problem(parent_problem, free_variables::Val{S}) where {S}
    @argcheck S isa Tuple{Vararg{Symbol}}
    transformation = free_transformation(parent_problem, free_variables)
    PartialProblem{S}(parent_problem, transformation)
end

lower_bound(pp::PartialProblem{S}) where S = free_lower_bound(pp.parent_problem, Val(S))

upper_bound(pp::PartialProblem{S}) where S = free_upper_bound(pp.parent_problem, Val(S))

function known_x(pp::PartialProblem{S}) where S
    (; parent_problem, transformation) = pp
    inverse(transformation, map(p -> p.known_value, free_parameters(parent_problem, Val(S))))
end

function calculate_objcons(pp::PartialProblem{S}, x::AbstractVector) where {S}
    (; parent_problem, transformation) = pp
    _known = known_values(parent_problem, Val(S))
    _free = transform(transformation, x)
    (; objective, moments_and_constraint_calculator) = parent_problem
    θ = merge(_free, _known)
    moments, constraint =  moments_and_constraint_calculator(θ)::MC_TYPE
    pushfirst!(copy(constraint), objective(moments))
end

###
###
###

function random_starting_point(lower_bound::AbstractVector, upper_bound::AbstractVector;
                               rng = default_rng())
    # NOTE: scaling is assumed to be more or less [-2, -2] for infinite intervals, x+[0,2]
    # etc for finite ones
    @argcheck length(lower_bound) == length(upper_bound)
    function _r(x, y)
        @argcheck !isnan(x) && !isnan(y)
        @argcheck x ≤ y
        if x == y
            x
        elseif isfinite(x) && isfinite(y)
            clamp(rand(rng) * (y - x) + x, x, y) # clamp for numerical error
        elseif isfinite(x)
            abs(randn(rng)) + x
        elseif isfinite(y)
            y - abs(randn(rng))
        else
            randn(rng)
        end
    end
    map(_r, lower_bound, upper_bound)
end

function solve_and_status(pp::PartialProblem, x0;
                          lb = lower_bound(pp), ub = upper_bound(pp),
                          catch_errors = true, solution_norm)
    (; solution_tol, constraint_dimension) = pp.parent_problem
    oc = Base.Fix1(calculate_objcons, pp)
    F = objcons_nlpmodel(oc; x0, lvar = lb, uvar = ub)
    x̃ = known_x(pp)
    try
        stats = percival(F)
        x = stats.solution
        Δ = solution_norm(x̃, x)
        oc_x = oc(x)
        objective = oc_x[1]
        constraints = oc_x[2:end]
        status = if !stats.solution_reliable
            :nonconvergence
        elseif Δ ≤ solution_tol && maximum(abs, constraints) ≤ solution_tol
            :convergence_correct
        else
            :convergence_incorrect
        end
        (; status, x0, x, objective, constraints)
    catch e
        if catch_errors
            @warn "error message" sprint(showerror, e)
            (; status = :error, x0, objective = nothing, constraints = nothing)
        else
            rethrow(e)
        end
    end
end

function solutions_from_random_points(pp::PartialProblem{S}; catch_errors = true,
                         solution_norm, random_x0_count) where S
    lb = lower_bound(pp)
    ub = upper_bound(pp)
    # NOTE parallelize this step
    [begin
         x0 = random_starting_point(lb, ub)
         solve_and_status(pp, x0; lb, ub, catch_errors, solution_norm)
     end
     for _ in 1:random_x0_count]
end

"""
$(SIGNATURES)

Solve the given problem step by step, checking that the known parameters are identified.

# Keyword arguments (width defaults)

- `catch_errors = true`: errors in the objective function are caught and treated as
  non-convergence

- `solution_norm = mean_abs2`: the solution nom

- `random_x0_count = 10`: random points to evaluate convergence

- `minimum_convergence_ratio = 1.0`: stop below this.
"""
function check_identification(problem::IdentificationProblem; catch_errors = true,
                              solution_norm = mean_abs2,
                              random_x0_count = 10,
                              minimum_convergence_ratio = 1.0)
    (; exogeneous, endogeneous) = classify_parameters(problem)
    @argcheck random_x0_count > 0
    @argcheck 0 ≤ minimum_convergence_ratio ≤ 1.0
    known_parameters = exogeneous
    free_parameters = endogeneous
    results = []
    while !isempty(known_parameters)
        v, known_parameters... = known_parameters
        free_parameters = (free_parameters..., v)
        pp = partial_problem(problem, Val(free_parameters)) # non-inferrable, but that's ok
        @info "solving partial model" free_parameters
        solutions = solutions_from_random_points(pp; catch_errors, solution_norm, random_x0_count)
        convergence_ratio = mean(s -> s.status ≡ :convergence_correct, solutions)
        if convergence_ratio < minimum_convergence_ratio
            @error "insufficient convergence" last_variable = v convergence_ratio
            error("insufficient convergence")
        end
        push!(results, (added_variable = v, pp.transformation, known_parameters, solutions))
    end
    @info "model is identified correctly"
    results
end

end # module
