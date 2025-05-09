#!/bin/bash
julia --project=. -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); using MARX"
