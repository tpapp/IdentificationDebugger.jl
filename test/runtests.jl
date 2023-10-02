using IdentificationDebugger
using Test, TransformVariables

using IdentificationDebugger: partial_problem, lower_bound, upper_bound, calculate_objcons

parameters = (a = parameter(2.0), b = parameter([1.0, 3.0]),
              c = endogeneous_parameter(as(Array, 3)))

"Matrix we use for tests, all that matters is that it is not singular & not block diagonal."
const M = [0.5 0.3 0.2;
           0.7 1.9 -5.0
           -0.2 0.5 1.4]

const z = [0.1, 0.2, -0.9]

IP = identification_problem(parameters) do θ
    (; a, b, c) = θ
    m = M * vcat(a, b)
    r = M * c .- z
    (c = m[1:2], d = m[3]), r
end

@test maximum(abs, M * IP.parameters.c.known_value .- z) ≤ 1e-8

PP = partial_problem(IP, Val((:a,)))

@test lower_bound(PP) == [-Inf]
@test upper_bound(PP) == [Inf]
@test maximum(abs, calculate_objcons(PP, [parameters.a.known_value])) ≤ 1e-8


@test check_identification(IP)


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
