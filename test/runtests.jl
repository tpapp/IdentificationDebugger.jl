using IdentificationDebugger
using Test
using ADNLPModels, JSOSolvers

using IdentificationDebugger: partial_problem, lower_bound, upper_bound, calculate_objective

parameters = (a = parameter(2.0), b = parameter([1.0, 3.0]))

"Matrix we use for tests, all that matters is that it is not singular & not block diagonal."
const M = [0.5 0.3 0.2;
           0.7 1.9 -5.0
           -0.2 0.5 1.4]

IP = identification_problem(parameters) do θ
    (; a, b) = θ
    r = M * vcat(a, b)
    (c = r[1:2], d = r[3])
end

PP = partial_problem(IP, Val((:a,)))

@test lower_bound(PP) == [-Inf]
@test upper_bound(PP) == [Inf]
@test calculate_objective(PP, [parameters.a.known_value]) == 0

function solver_ADNLP(objective, lb, ub, x0)
    model = ADNLPModel(objective, x0, lb, ub)
    stats = tron(model)
    converged = stats.solution_reliable
    x = stats.solution
    (; x, converged)
end

@test check_identification(solver_ADNLP, IP)


## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     @test isempty(JET.get_reports(report_package(IdentificationDebugger, target_modules=(IdentificationDebugger,))))
# end

## NOTE add Aqua to the test environment, then uncomment
# @testset "QA with Aqua" begin
#     import Aqua
#     Aqua.test_all(IdentificationDebugger; ambiguities = false)
#     # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
#     Aqua.test_ambiguities(IdentificationDebugger)
# end
