using Graphs

struct SimpleNeuroGraph
    g :: SimpleDiGraph
    weights :: Dict
    names :: Dict
    blox :: Dict
end

function Graphs.AdjMatrixfromSimpleNeuroGraph(g::SimpleNeuroGraph)
    myadj = map(Float64,adjacency_matrix(g.g))
    for edge in edges(g.g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = g.weights[(s,d)]
    end
    myadj
end

function Graphs.add_edge!(g :: SimpleNeuroGraph, src, dst, weight)
    add_edge!(g.g,src,dst)
    g.weights[(src,dst)] = weight
end

function Graphs.add_vertex!(g :: SimpleNeuroGraph, name, blox)
    add_vertex!(g.g)
    g.names[nv(g.g)] = name # vertices are added to the end
    g.blox[nv(g.g)] = blox
end

function Graphs.add_vertices!(g :: SimpleNeuroGraph, n, names, blox)
    add_vertices!(g.g, n)
    numverts = nv(g.g)
    for i in 1:n
        idx = numverts-n+i # fill them towards the end
        g.names[idx] = names[i]
        g.blox[idx] = blox[i]
    end
end

function Graphs.rem_vertex!(g :: SimpleNeuroGraph, v)
# somewhat tricky to implement since the remaining graph vertex indices
# get shifted down from where the vertex is deleted

    n = nv(g.g) # number of vertices before delete
    for i in v:n-1
        g.names[v] = g.names[v+1]
        g.blox[v] = g.blox[v+1]
    end

    # remove last
    delete!(g.names,n)
    delete!(g.blox,n)

    new_weights = Dict() # dictionary to contain the renamed edge weights
    for (key, value) in g.weights
        src = key[1]
        dst = key[2]
        if src==v || dst ==v
            delete!(g.weights, key)
        elseif  src>v || dst>v
            new_src = (src > v ? src-1 : src)
            new_dst = (dst > v ? dst-1 : dst)
            new_weights[(new_src,new_dst)] = value # updated weight entry
            delete!(g.weights, key) # delete old entry
        end
    end
    merge!(g.weights, new_weights) # add new entries to weights
    rem_vertex!(g.g,v) # now remove vertex
end

