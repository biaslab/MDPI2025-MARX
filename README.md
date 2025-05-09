# MDPI Entropy Hierarchical Multivariate ARX-EFE

This is the companion repository to a paper submitted to the MDPI Entropy journal entitled:

`Factor graph-based online Bayesian identification and component evaluation for multivariate autoregressive exogenous input models`

This repository contains the code associated with our submitted paper.
Please note that the code base is currently undergoing cleanup and improvements.
We are actively working to resolve known issues and improve reusability.
Updates will be pushed in the coming days.

## How do I run your experiments?

On OS X and Linux, run `./install.sh` to install packages, `./configure-pluto.sh` to update the Pluto notebooks, then run `./run-pluto.sh` which should open a tab in your browser.
On Windows, please run the commands listed in `install.sh` and `configure-pluto.sh`, followed by the commands listed in `run-pluto.sh`.
Note that the code only has been tested with julia version `lts = 110.7`.
After running (the contents of) `run-pluto.sh`, a new browser tab should open with the url [http://localhost:1234/](http://localhost:1234/).

There are several Pluto notebooks that let you train and evaluate an MARX estimator:

* `experiment-verification-MARX-train.jl` Train an agent for the MARX system (will save the agent, environment, and interaction data to `saves/`)
* `experiment-verification-MARX-offline.jl` Test the MARX estimator and the least-squares estimator on a single run in an offline (batch) manner
* `experiment-verification-MARX-testdream.jl` Test the MARX estimator on a single run, first feeding it real observations then predicted observations to its memory
* `experiment-verification-MARX-montecarlo.jl` Test the MARX estimator and the least-squares estimator in several Monte Carlo experiments across a variety of training sizes
* `experiment-validation-dmsds-train.jl` Train an agent for the double mass spring damper system (will save the agent, environment, and interaction data to `saves/`)
* `experiment-validation-dmsds-offline.jl` Test the MARX estimator and the least-squares estimator on a single run in an offline (batch) manner
* `experiment-validation-dmsds-testdream.jl` Test the MARX estimator and the least-squares estimator on a single run in an offline (batch) manner

## How to I give feedback?

Please submit an issue.
