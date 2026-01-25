# [Plotting Figures](@id api_plotting)
```@meta
CurrentModule = Neuroblox
```

The backend for generating plots for Neuroblox is [Makie](https://docs.makie.org/stable/). In order to call these functions, one must have a Makie backend installed, such as CairoMakie or GLMakie.

```@docs
meanfield
meanfield!
rasterplot
rasterplot!
stackplot
stackplot!
frplot
frplot!
powerspectrumplot
powerspectrumplot!
```