using Distributions
using StatsBase
using LinearAlgebra

mutable struct GibbsFixedVar
    points::Array{Float64}
    K::Int64
    zs::Vector{Int64}
    μs::Array{Float64}
    dist::Float64
    dist_change::Float64
    function GibbsFixedVar(points::Array{Float64}, K::Int64)
        n_points = size(points, 1)
        pts = reshape(points, n_points, :)
        n_dims = size(pts, 2)
        new(pts, K, zeros(n_points), zeros(K, n_dims), Inf, Inf)
    end
end


function update!(g::GibbsFixedVar)
    gauss_dists = mapslices(x->MvNormal(x, I), g.μs, dims = 2)
    pdf_gauss(d) = mapslices(x->pdf(d, x), g.points, dims = 2)
    zsi_params = hcat(pdf_gauss.(gauss_dists)...)
    zsi_params ./= sum(zsi_params, dims = 2)
    g.zs = vec(rand.(mapslices(Categorical, zsi_params, dims = 2)))
    ns = countmap(g.zs)
    xs = vcat([mean(g.points[g.zs .== i, :], dims = 1) for i in 1:g.K]...)
    μs_dist = [(nk = ns[i]; MvNormal(nk / (nk + 1) * xs[i, :], 1 / (nk + 1) * I)) for i in 1:g.K]
    g.μs = hcat(rand.(μs_dist)...)'
end

# x = GibbsFixedVar([0.0, 1.0, 2.0, 4.0, 3,2], 2)
# x = GibbsFixedVar([0.0 3.0; 1.0 4.5; 4.3 2.0], 2)
# update!(x)