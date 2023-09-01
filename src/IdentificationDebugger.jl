"""
Placeholder for a short summary about IdentificationDebugger.
"""
module IdentificationDebugger

export parameter, identification_problem

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES
using FillArrays: Fill
using TransformVariables: dimension, transform, inverse, as, asℝ

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

_default_lower_bound(transformation) = Fill(-Inf, dimension(transformation))

_default_upper_bound(transformation) = Fill(Inf, dimension(transformation))

"$(SIGNATURES)"
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

###
### identification problem framework
###

Base.@kwdef struct IdentificationProblem{TP<:NamedTuple,TM,TO}
    parameters::TP
    moment_calculator::TM
    objective::TO
end

"$(SIGNATURES)"
function identification_problem(moment_calculator, parameters::NamedTuple)
    moments = moment_calculator(map(p -> p.known_value, parameters))
    objective = LeastSquaresObjective(moments)
    IdentificationProblem(; parameters, moment_calculator, objective)
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

function calculate_objective(pp::PartialProblem{S}, x::AbstractVector) where {S}
    (; parent_problem, transformation) = pp
    _known = known_values(parent_problem, Val(S))
    _free = transform(transformation, x)
    (; objective, moment_calculator) = parent_problem
    moments =  moment_calculator(merge(_free, _known))
    objective(moments)
end

end # module
