SHELL = /bin/bash

.PHONY: test

TEST_OUTPUT_FILES = _testout

test: ## Run tests (requires test/runtests.jl)
	julia --project=. -e 'import Pkg; Pkg.activate("."); Pkg.test()'
