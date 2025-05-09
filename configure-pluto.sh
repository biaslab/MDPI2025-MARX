#!/bin/bash
# for every pluto notebook, activate this environment to add MARX as a package (thanks Fons van der Plas)
julia --project=. -e "using Pkg; Pkg.update()"
for i in *.jl; do
	#$julia --project=. -e "using Pluto; Pluto.activate_notebook_environment(\"$i\"); println(Base.active_project())"
	#julia --project=. -e "using Pluto; Pluto.activate_notebook_environment(\"$i\"); using Pkg; Pkg.upgrade_manifest(); Pkg.update(); Pkg.add(PackageSpec(path=\".\"))"
	julia --project=. -e "using Pluto; Pluto.activate_notebook_environment(\"$i\"); using Pkg; Pkg.update(); Pkg.develop(PackageSpec(path=\".\"))"
done
