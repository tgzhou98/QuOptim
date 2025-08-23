using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ModelContextProtocol
using TensorQEC
using TensorQEC.QECCore

# Define a tool - can now return Dict directly!
get_code_stabilizers = MCPTool(
    name = "get_code_stabilizers",
    description = "Get code stabilizers",
    parameters = [ ToolParameter(
        name = "codename",
        type = "string", 
        description = "Code name, supported codes: SurfaceCode(d,d), ToricCode(d,d), ShorCode(), SteaneCode(), Code832(), Code422(), Code1573()",
        required = true
    )],
    handler = params -> Dict("code_stabilizers" => string(stabilizers(eval(Meta.parse(params["codename"]))))),
    return_type = TextContent  # Explicitly expect single TextContent
)

get_code_logicals = MCPTool(
    name = "get_code_logicals",
    description = "Get code logicals",
    # Supported code names: SurfaceCode(d,d)
    parameters = [ ToolParameter(
        name = "codename",
        type = "string", 
        description = "Code name, supported codes: SurfaceCode(d,d), ToricCode(d,d), ShorCode(), SteaneCode(), Code832(), Code422(), Code1573()",
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
    return_type = TextContent  # Explicitly expect single TextContent
)

server = mcp_server(
    name = "tensorqec-server",
    description = "TensorQEC service",
    tools = [get_code_stabilizers, get_code_logicals],
    resources = nothing,
    prompts = nothing
)

# Start the server
start!(server)