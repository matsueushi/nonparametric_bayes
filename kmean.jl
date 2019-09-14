using Distributions

mutable struct KMean
    points::Array{Float64}
    K::Int64
    zs::Vector{Int64}
    μs::Array{Float64}
    dist::Float64
    dist_change::Float64
    function KMean(points::Array{Float64}, K::Int64)
        n_points = size(points, 1)
        pts = reshape(points, n_points, :)
        n_dims = size(pts, 2)
        new(pts, K, zeros(n_points), zeros(K, n_dims), Inf, Inf)
    end
end


function update!(km::KMean)
    diff_sq = (km.points .- reshape(km.μs', 1, size(km.μs, 2), :)).^2
    norms = dropdims(sum(diff_sq, dims = 2), dims = 2)
    amin = argmin(norms, dims = 2)
    km.zs = vec([x[2] for x in amin])
    if Base.length(unique(km.zs)) != km.K
        min, max = minimum(km.points, dims = 1), maximum(km.points, dims = 1)
        init_μs = mapslices(x->rand(Uniform(x...), km.K), vcat(min, max), dims = 1)
        km.μs = init_μs
    else
        km.μs = vcat([mean(km.points[km.zs .== i, :], dims = 1) for i in 1:km.K]...)
    end
    new_dist = sum(norms[amin])
    km.dist_change = km.dist - new_dist
    km.dist = new_dist
end

