# code reflection entry-points

# forward the rest to GPUCompiler with an appropriate CompilerJob

#
# code_* replacements
#

# Split the keyword arguments that configure compilation (the same ones `@roc`
# accepts, e.g. `unsafe_fp_atomics`) from those that belong to GPUCompiler's
# reflection entry-point itself (e.g. `raw`, `dump_module`). Without this the
# former reach `GPUCompiler.code_*` as unsupported keywords and throw, so a
# kernel could not be inspected under the configuration it is compiled with.
function _split_compiler_kwargs(kwargs)
    nt = values(kwargs)
    ks = keys(nt)
    cfg = filter(k -> k in COMPILER_KWARGS, ks)
    rest = filter(k -> !(k in COMPILER_KWARGS), ks)
    return NamedTuple{cfg}(nt), NamedTuple{rest}(nt)
end

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take an io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    @eval begin
        function $method(
            io::IO, @nospecialize(func), @nospecialize(types);
            kernel::Bool=false, device=HIP.device(), kwargs...,
        )
            config_kwargs, reflect_kwargs = _split_compiler_kwargs(kwargs)
            source = methodinstance(typeof(func), Base.to_tuple_type(types))
            config = Compiler.compiler_config(device; kernel, config_kwargs...)
            job = CompilerJob(source, config)
            GPUCompiler.$method($(args...); reflect_kwargs...)
        end
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

const code_gcn = code_native

#
# @device_code_* macros
#

export @device_code_lowered, @device_code_typed, @device_code_warntype,
       @device_code_llvm, @device_code_gcn, @device_code

# forward the rest to GPUCompiler
@eval $(Symbol("@device_code_lowered")) = $(getfield(GPUCompiler, Symbol("@device_code_lowered")))
@eval $(Symbol("@device_code_typed")) = $(getfield(GPUCompiler, Symbol("@device_code_typed")))
@eval $(Symbol("@device_code_warntype")) = $(getfield(GPUCompiler, Symbol("@device_code_warntype")))
@eval $(Symbol("@device_code_llvm")) = $(getfield(GPUCompiler, Symbol("@device_code_llvm")))
@eval $(Symbol("@device_code_gcn")) = $(getfield(GPUCompiler, Symbol("@device_code_native")))
@eval $(Symbol("@device_code")) = $(getfield(GPUCompiler, Symbol("@device_code")))
