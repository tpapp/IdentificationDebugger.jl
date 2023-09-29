"""
Placeholder for a short summary about IdentificationDebugger.
"""
module IdentificationDebugger

export parameter, identification_problem, check_identification

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using FillArrays: Fill
using Logging: @error
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
end

_default_transformation(a::Real) = asℝ

_default_transformation(a::AbstractArray) = as(Array, size(a)...)

_default_lower_bound(transformation) = fill(-Inf, dimension(transformation))

_default_upper_bound(transformation) = fill(Inf, dimension(transformation))

"""
$(SIGNATURES)

Description of a parameter. The exercise is to recover the `known_value` in a simulated
setting.

# Keyword parameters

- `transformation`: a transformation from ``ℝⁿ``, if applicable, implicitly also provides
  the dimension `n`. Defaults to the identity, reshaped.
- `lower_bound`, `upper_bound`: lower- and upper bound in ``ℝⁿ`` for the estimation
"""
function parameter(known_value;
                   transformation = _default_transformation(known_value),
                   lower_bound::AbstractVector = _default_lower_bound(transformation),
                   upper_bound::AbstractVector = _default_upper_bound(transformation))
    @argcheck dimension(transformation) == length(lower_bound) == length(upper_bound)
    x = inverse(transformation, known_value)
    @argcheck all(lower_bound .≤ x .≤ upper_bound)
    Parameter(; known_value, transformation, lower_bound, upper_bound)
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

Base.@kwdef struct IdentificationProblem{TP<:NamedTuple,TM,TO,TN,}
    parameters::TP
    moments_and_constraint_calculator::TM
    objective::TO
    solution_norm::TN
    solution_tol::Float64
    random_x0_count::Int
    minimum_convergence_ratio::Float64
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


"$(SIGNATURES)"
function identification_problem(moments_and_constraint_calculator,
                                parameters::NamedTuple,
                                solution_norm = mean_abs2, solution_tol = 1e-4,
                                random_x0_count = 10, minimum_convergence_ratio = 1.0)
    moments, constraint = moments_and_constraint_calculator(map(p -> p.known_value, parameters))
    objective = LeastSquaresObjective(moments)
    IdentificationProblem(; parameters, moments_and_constraint_calculator, objective, solution_norm,
                          solution_tol, random_x0_count, minimum_convergence_ratio)
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

function calculate_objective_and_constraint(pp::PartialProblem{S}, x::AbstractVector) where {S}
    (; parent_problem, transformation) = pp
    _known = known_values(parent_problem, Val(S))
    _free = transform(transformation, x)
    (; objective, moments_and_constraint_calculator) = parent_problem
    moments, constraint =  moments_and_constraint_calculator(merge(_free, _known))
    objective(moments), constraint
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
            rand(rng) * (y - x) + x
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

const SOLVER_DOCS = """
The solver is a function which will be called as

```julia
(; converged, x) = solver(f, x0, lb, ub, lc, uc)
```
where `converged` should be a boolean indicating convergence and `x` is the solution.
"""

function solve_and_status(pp::PartialProblem, solver, x0;
                          lb = lower_bound(pp), ub = upper_bound(pp),
                          catch_errors = true)

    (; solution_norm, solution_tol) = pp.parent_problem
    x̃ = known_x(pp)
    try
        (; converged, x) = solver(Base.Fix1(calculate_objective, pp), x0, lb, ub, lc, uc)
        Δ = solution_norm(x̃, x)
        status = if !converged
            :nonconvergence
        elseif Δ ≤ solution_tol
            :convergence_correct
        else
            :convergence_incorrect
        end
        (; x0, status)
    catch e
        if catch_errors
            @warn "error message" sprint(showerror, e)
            (; x0, status = :error)
        else
            rethrow(e)
        end
    end
end

function solve_and_check(pp::PartialProblem{S}, solver; catch_errors = true) where S
    lb = lower_bound(pp)
    ub = upper_bound(pp)
    (; random_x0_count::Int, minimum_convergence_ratio::Float64) = pp.parent_problem
    # NOTE parallelize this step
    s = [solve_and_status(pp, solver, random_starting_point(lb, ub); lb, ub, catch_errors)
         for _ in 1:pp.parent_problem.random_x0_count]
    s_e = findfirst(s -> s.status ≡ :error, s)
    if s_e ≢ nothing
        @error "starting point errored" S s[s_e].x0
        error("at least one starting point errored, check logs")
    end
    convergence_ratio = mean(s -> s.status ≡ :convergence_correct, s)
    ok = convergence_ratio ≥ minimum_convergence_ratio
    if !ok
        @error "did not reach expected converge" convergence_ratio minimum_convergence_ratio
    end
    ok
end

"""
$(SIGNATURES)

Solve the given problem step by step, checking that the known parameters are identified.

## Solver interface

$(SOLVER_DOCS)
"""
function check_identification(solver, problem::IdentificationProblem; catch_errors = true)
    known_variables = keys(problem.parameters)
    free_variables = ()
    while !isempty(known_variables)
        v, known_variables... = known_variables
        free_variables = (free_variables..., v)
        pp = partial_problem(problem, Val(free_variables)) # non-inferrable, but that's ok
        @info "solving partial model" free_variables
        ok = solve_and_check(pp, solver; catch_errors)
        if !ok
            error("solution failed when adding variable $(v)")
        end
    end
    @info "model is identified correctly"
    true
end

end # module
