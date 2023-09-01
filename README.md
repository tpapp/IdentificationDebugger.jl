# IdentificationDebugger.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/IdentificationDebugger.jl/workflows/CI/badge.svg)](https://github.com/tpapp/IdentificationDebugger.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/IdentificationDebugger.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/IdentificationDebugger.jl?branch=master)
<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/IdentificationDebugger.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/IdentificationDebugger.jl/dev)
-->
<!-- Aqua badge, see test/runtests.jl -->
<!-- [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) -->

Very, very experimental project to check identification of an (economic) model by

1. simulating a bunch of moments with known parameters,

2. adding free parameters one by one to the estimation to see if the correct parameters are obtained from random starting points.

We are using this in a paper; if you want to use it while it is experimental please open an issue and I will document and clean up the API.
