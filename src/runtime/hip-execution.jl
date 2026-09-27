# `map` over a tuple, unrolled for any length. `Base.map(f, ::Tuple)` falls back to a `Vector{Any}` and a
# splat for tuples of 32 or more elements (`Base.Any32`), which makes every launch of a kernel with that
# many parameters type-unstable (host allocations and dynamic dispatch per launch).
# Unrolled with literal indices, so it stays type-stable for heterogeneous tuples (`ntuple` with a
# closure over `t[i]` does not: its index is not a constant inside the closure).
@inline @generated _tmap(f, t::Tuple) = :(($((:(f(t[$i])) for i in 1:fieldcount(t))...),))

# In contrast to `Base.RefValue` we just need a container for both pass-by-ref (Symbol),
# and pass-by-value (immutable structs).
mutable struct ArgBox{T}
    const val::T
end

@inline function Base.unsafe_convert(P::Union{Type{Ptr{T}}, Type{Ptr{Cvoid}}}, box::ArgBox{T})::P where {T}
    return pointer_from_objref(box)
end

"""
    (ker::HIPKernel)(args::Vararg{Any, N}; kwargs...)

Launch compiled HIPKernel by passing arguments to it.

The following kwargs are supported:
- `gridsize::ROCDim = 1`: Size of the grid.
- `groupsize::ROCDim = 1`:  Size of the workgroup.
- `shmem::Integer = 0`: Amount of dynamically-allocated shared memory in bytes.
- `stream::HIP.HIPStream = AMDGPU.stream()`: Stream on which to launch the kernel.
"""
struct HIPKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::HIP.HIPFunction
end

@inline @generated function call(
    kernel::HIPKernel{F, TT}, args::Tuple;
    stream::HIP.HIPStream, call_kwargs...,
) where {F, TT}
    sig = Tuple{F, TT.parameters...} # Base.signature_type with a function type
    args = (:(kernel.f), (:( args[$i] ) for i in 1:length(args.parameters))...)

    # filter out ghost arguments that shouldn't be passed.
    predicate = dt -> GPUCompiler.isghosttype(dt) || Core.Compiler.isconstType(dt)
    # Note: Define a single LLVM context, otherwise it is created per every param.
    to_pass = LLVM.Context() do _
        map(!predicate, sig.parameters)
    end
    call_t = Type[x[1] for x in zip(sig.parameters, to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass) if x[2]]

    # add the kernel state
    pushfirst!(call_t, AMDGPU.KernelState)
    pushfirst!(call_args, :(AMDGPU.KernelState(stream.device, kernel.fun.global_hostcalls)))

    # finalize types
    call_tt = Base.to_tuple_type(call_t)
    quote
        roccall_tuple(kernel.fun, $call_tt, ($(call_args...),); stream, call_kwargs...)
    end
end

function (ker::HIPKernel{F, TT})(
    args::Vararg{Any, N}; stream::HIP.HIPStream = AMDGPU.stream(), call_kwargs...,
) where {F, TT, N}
    # Check if previous kernels threw an exception.
    AMDGPU.throw_if_exception(stream.device)
    GC.@preserve args begin
        converted = _tmap(arg -> AMDGPU.rocconvert(arg, stream), args)
        call(ker, converted; stream, call_kwargs...)
    end
end

@inline @generated function convert_arguments(f::Function, ::Type{tt}, args::Tuple) where tt
    types = tt.parameters
    n = length(args.parameters)

    ex = quote end

    converted_args = Vector{Symbol}(undef, n)
    arg_ptrs = Vector{Symbol}(undef, n)
    for i in 1:n
        converted_args[i] = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($(converted_args[i]) = Base.cconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = Base.unsafe_convert($(types[i]), $(converted_args[i]))))
    end

    append!(ex.args, (quote
        GC.@preserve $(converted_args...) begin
            f(($(arg_ptrs...),))
        end
    end).args)
    return ex
end

roccall(fun::F, tt::Type{T}, args::Vararg{Any, N}; kwargs...) where {F, T, N} = roccall_tuple(fun, tt, args; kwargs...)

function roccall_tuple(fun::F, tt::Type{T}, args::Tuple; kwargs...) where {F, T}
    cvt_fn = pointers -> launch_tuple(fun, pointers; kwargs...)
    convert_arguments(cvt_fn, tt, args)
end

@inline function pack_arguments(f::F, args::Tuple) where F
    boxes = _tmap(ArgBox, args)
    GC.@preserve args boxes begin
        pointers = _tmap(box -> Base.unsafe_convert(Ptr{Cvoid}, box), boxes)
        f(Ref(pointers))
    end
end

launch(fun::HIP.HIPFunction, args::Vararg{Any, N}; kwargs...) where N = launch_tuple(fun, args; kwargs...)

# the argument tuple is passed, never splatted: inference gives up on splats of more than 32
# elements (`max_tuple_splat`), and boxes every argument of such a call
function launch_tuple(
    fun::HIP.HIPFunction, args::Tuple;
    gridsize = 1, groupsize = 1,
    shmem::Integer = 0, stream::HIP.HIPStream,
    cooperative = false,
)
    gd = gridsize isa ROCDim3 ? gridsize : ROCDim3(gridsize)
    bd = groupsize isa ROCDim3 ? groupsize : ROCDim3(groupsize)
    # the device side assumes workgroup indices fit in Int32 (see Device._max_groups)
    (gd.x <= typemax(Int32) && gd.y <= typemax(Int32) && gd.z <= typemax(Int32)) ||
        throw(ArgumentError("gridsize exceeds $(typemax(Int32)) workgroups in a dimension"))
    # TODO guard with try/catch & diagnose a failure
    pack_arguments(args) do kernel_params
        # Inline into `pack_arguments`, so that `kernel_params` does not escape into a
        # call and can be promoted to an `alloca` instead of being heap-allocated.
        @inline
        if cooperative
            HIP.hipModuleLaunchCooperativeKernel(
                fun, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z,
                shmem, stream, kernel_params)
        else
            HIP.hipModuleLaunchKernel(
                fun, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z,
                shmem, stream, kernel_params, C_NULL)
        end
    end

    AMDGPU.LAUNCH_BLOCKING[] && AMDGPU.synchronize(stream)
    return
end
