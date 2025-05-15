# MDPI Entropy Multivariate Autoregressive model with Exogeneous Inputs

This is the companion repository to a paper submitted to the MDPI Entropy journal entitled:

`Factor graph-based online Bayesian identification and component evaluation for multivariate autoregressive exogenous input models`

This repository contains the code associated with our submitted paper.
Please note that the code base is currently undergoing cleanup and improvements.
We are actively working to resolve known issues and improve reusability.
Updates will be pushed in the coming days.

## How do I run your experiments?

To install the required Julia packages, run `./install.sh` (OS X and Linux) or `julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'` (Windows).

To run the experiments, run `run-experiments.sh` (OS X and Linux) or `JULIA_NUM_THREADS=4 julia --project=. experiments-MARX.jl` and `JULIA_NUM_THREADS=$NUM_THREADS julia --project=. experiments-dmsds.jl` (Windows).
Results will be stored in a directory called `results`.

## How do I inspect the results of your experiments?

We use [Pluto](https://plutojl.org/) notebooks to interactively inspect our models.

Follow the instructions in the previous section to install required Julia packages.
Currently, the results of our experiments take up considerable storage space (11 GB for one environment and the chosen three models; MARX-UI, MARX-WI, RLS).
We are actively working to minimize the storage space required for each monte carlo experiment.
This means you have to run the experiments (~10 minutes per model and environment) before you can inspect the results stored in the `results` directory.

Then, run `./run-pluto.sh` (OS X and Linux) or `julia --project=. -e 'using Pluto; Pluto.run()'` (Windows).
A new browser tab should open with the url [http://localhost:1234/](http://localhost:1234/).

There are two Pluto notebooks that let you inspect the experiments:

* `inspect-MARX.jl` lets you inspect the results for the MARX system
* `inspect-dmsds.jl` lets you inspect the results for the double mass-spring-damper system

## How to I give feedback?

Please submit an issue.
