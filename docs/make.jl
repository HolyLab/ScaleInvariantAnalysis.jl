using ScaleInvariantAnalysis
using Documenter

DocMeta.setdocmeta!(ScaleInvariantAnalysis, :DocTestSetup, :(using ScaleInvariantAnalysis); recursive=true)

makedocs(;
    modules=[ScaleInvariantAnalysis],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="ScaleInvariantAnalysis.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/ScaleInvariantAnalysis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/ScaleInvariantAnalysis.jl",
    devbranch="main",
)
