using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ModelContextProtocol
using TensorQEC
using TensorQEC.QECCore

get_code_stabilizers = MCPTool(
    name = "get_code_stabilizers",
    description = "Get code stabilizers",
    parameters = [ ToolParameter(
        name = "codename",
        type = "string", 
        description = "Code name, supported codes: Unrotated surface code:`Surface(d,d)`, Rotated surface code:`SurfaceCode(d,d)`, `ToricCode(d,d)`, `ShorCode()`, `SteaneCode()`, `Code832()`, `Code422()`, `Code1573()`, [144,12,12] BB code(bbcode, bb code): `BivariateBicycleCode(6,12, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0)))`",
        required = true
    )],
    handler = params -> Dict("code_stabilizers" => string(stabilizers(eval(Meta.parse(params["codename"]))))),
    return_type = TextContent
)

get_code_logicals = MCPTool(
    name = "get_code_logicals",
    description = "Get code logicals",
    parameters = [ ToolParameter(
        name = "codename",
        type = "string", 
        description = "Code name, supported codes: Unrotated surface code:`Surface(d,d)`, Rotated surface code:`SurfaceCode(d,d)`, `ToricCode(d,d)`, `ShorCode()`, `SteaneCode()`, `Code832()`, `Code422()`, `Code1573()`, [144,12,12] BB code(bbcode, bb code): `BivariateBicycleCode(6,12, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0)))`",
        required = true
    )],
    handler = function (params)
        code = eval(Meta.parse(params["codename"]))
        tanner = CSSTannerGraph(code)
        lx,lz = logical_operator(tanner)
        qubitnum = tanner.stgx.nq
        lzs = [PauliString(qubitnum, findall(x->x.x,row)=>Pauli(3)) for row in eachrow(lz)]
        lxs = [PauliString(qubitnum, findall(x->x.x,row)=>Pauli(1)) for row in eachrow(lx)]
        Dict("code_logicals" => string(lxs,lzs))
    end,
    return_type = TextContent
)

compute_code_distance = MCPTool(
    name = "compute_code_distance",
    description = "Compute code distance with integer programming",
    parameters = [ ToolParameter(
        name = "codename",
        type = "string", 
       description = "Code name, supported codes: Unrotated surface code:`Surface(d,d)`, Rotated surface code:`SurfaceCode(d,d)`, `ToricCode(d,d)`, `ShorCode()`, `SteaneCode()`, `Code832()`, `Code422()`, `Code1573()`, [144,12,12] BB code(bbcode, bb code): `BivariateBicycleCode(6,12, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0)))`",
        required = true
    )],
    handler = function (params)
        code = eval(Meta.parse(params["codename"]))
        tanner = CSSTannerGraph(code)
        distance = code_distance(tanner)
        Dict("code_distance" => string(distance))
    end,
    return_type = TextContent
)

server = mcp_server(
    name = "tensorqec-server",
    description = "TensorQEC service",
    tools = [get_code_stabilizers, get_code_logicals, compute_code_distance],
    resources = nothing,
    prompts = nothing
)

# Start the server
start!(server)