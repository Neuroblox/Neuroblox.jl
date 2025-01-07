# Installing Neuroblox

To install Neuroblox.jl, you need to first add the [NeurobloxRegistry](https://github.com/Neuroblox/NeurobloxRegistry) to your list of registries that julia checks for available packages

```julia
using Pkg
pkg"registry add https://github.com/Neuroblox/NeurobloxRegistry"
```
If this is your first time using Julia, you *may* also need to add the General registry, which can be done with
```
pkg"registry add General"
```

The next step is to install Neuroblox from the NeurobloxRegistry. It is also useful to install some other packages that are commonly used with Neuroblox. These packages are used in the tutorials of the next section. We have included Neuroblox and these other packages into a single `Project.toml` file which you can download and then use it to activate a new environment where all the necessary packages will be installed. To do this first choose a folder where you want this environment to be generated in and then run 

``` julia 
using Downloads

Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/Project.toml", joinpath(@__DIR__, "Project.toml"))
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

Please note that after running these commands `Neuroblox` will also be installed along with all other packages that are used in the tutorials.

> **_NOTE_:**
> If you want to install only Neuroblox and not the other packages used in the tutorials you can run 
> ```julia 
> Pkg.add("Neuroblox")
> ```
