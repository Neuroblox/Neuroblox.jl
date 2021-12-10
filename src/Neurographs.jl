using Graphs
import Graphs.SimpleGraphs.add_edge!
import Graphs.SimpleGraphs.add_vertex!
import Graphs.SimpleGraphs.add_vertices!
import Graphs.SimpleGraphs.rem_vertex!

struct SimpleNeuroGraph
    g :: SimpleDiGraph
    weights :: Dict
    names :: Dict
    blox :: Dict
end

function AdjMatrixfromSimpleNeuroGraph(g::SimpleNeuroGraph)
    myadj = map(Float64,adjacency_matrix(g.g))
    for edge in edges(g.g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = g.weights[(s,d)]
    end
    myadj
end

function add_edge!(g :: SimpleNeuroGraph, src, dst, weight)
    add_edge!(g.g,src,dst)
    g.weights[(src,dst)] = weight
end

function add_vertex!(g :: SimpleNeuroGraph, name, blox)
    add_vertex!(g.g)
    g.names[nv(g.g)] = name # vertices are added to the end
    g.blox[nv(g.g)] = blox
end

function add_vertices!(g :: SimpleNeuroGraph, n, names, blox)
    add_vertices!(g.g, n)
    numverts = nv(g.g)
    for i in 1:n
        idx = numverts-n+i # fill them towards the end
        g.names[idx] = names[i]
        g.blox[idx] = blox[i]
    end
end

function rem_vertex!(g :: SimpleNeuroGraph, v)
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
        if src>v || dst>v
            new_src = (src > v ? src-1 : src)
            new_dst = (dst > v ? dst-1 : dst)
            new_weights[(new_src,new_dst)] = value # updated weight entry
            delete!(g.weights, key) # delete old entry
        end
        if src==v || dst ==v
            delete!(g.weights, key)
        end
    end
    merge!(g.weights, new_weights) # add new entries to weights
    rem_vertex!(g.g,v) # now remove vertex
end

# will move this into the test
function test()
    g = SimpleNeuroGraph(SimpleDiGraph(),Dict(),Dict(),Dict())
    add_vertex!(g,"name1","blox1")
    add_vertex!(g,"name2","blox2")
    add_vertex!(g,"name3","blox3")
    add_edge!(g,1,2,1.0)
    add_edge!(g,2,3,2.0)
    add_edge!(g,3,1,3.0)
    a = AdjMatrixfromSimpleNeuroGraph(g)
    rem_vertex!(g,2)
    b = AdjMatrixfromSimpleNeuroGraph(g)
    return a,b
end

