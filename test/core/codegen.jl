using Test
using AMDGPU
using AMDGPU: Device, ROCArray, @roc
using AMDGPU.Device: sync_workgroup, workitemIdx, workgroupIdx, workgroupDim
using KernelAbstractions: @atomic

@testset "Synchronization" begin
    function synckern()
        sync_workgroup()
        nothing
    end

    iob = IOBuffer()
    AMDGPU.code_gcn(iob, synckern, Tuple{}; kernel=true)
    @test occursin("s_barrier", String(take!(iob)))
end

@testset "Trapping" begin
    function trapkern()
        Device.trap()
        nothing
    end
    function debugtrapkern()
        Device.debugtrap()
        nothing
    end

    iob = IOBuffer()
    AMDGPU.code_gcn(iob, trapkern, Tuple{}; kernel=true)
    @test occursin("s_trap 2", String(take!(iob)))

    iob = IOBuffer()
    AMDGPU.code_gcn(iob, debugtrapkern, Tuple{}; kernel=true)
    @test occursin("s_trap 3", String(take!(iob)))
end

@testset "Hardware FP atomics" begin
    function atomic_fp_ker!(x)
        @inbounds @atomic x[1] += 1f0
        return
    end

    # `global_atomic_add_f32` only exists on CDNA (gfx908, gfx90a, gfx94x, gfx95x)
    # and RDNA3+ (gfx11+), see `FeatureAtomicFaddNoRtnInsts` in LLVM's AMDGPU.td.
    # Vega (gfx900-gfx906) and RDNA1/2 (gfx10xx) lack it, so LLVM must expand
    # the atomic to a CAS loop there.
    arch = first(split(AMDGPU.HIP.gcn_arch(AMDGPU.device()), ':'))
    gen = parse(Int, match(r"^gfx([0-9a-f]+)", arch).captures[1]; base=16)
    has_hw_fadd = 0x908 <= gen < 0x1000 || gen >= 0x1100

    for (T, fp) in ((Float32, "f32"),)
        iob = IOBuffer()
        tt = Tuple{AMDGPU.Device.ROCDeviceVector{T, AMDGPU.Device.AS.Global}}
        AMDGPU.code_gcn(iob, atomic_fp_ker!, tt; kernel=true)
        gcn = String(take!(iob))
        if has_hw_fadd
            @test occursin("global_atomic_add_$fp", gcn)
        else
            @test occursin("global_atomic_cmpswap", gcn)
            @test !occursin("global_atomic_add_$fp", gcn)
        end
    end
end

@testset "Dispatch-packet reads stay scalar" begin
    # workgroupDim reads 16-bit fields of the AQL dispatch packet; those must
    # lower to SMEM (s_load) rather than a per-wave VMEM read of uncached
    # queue memory. The Int64 3-D indexing pattern below used to defeat LLVM's
    # sub-dword load widening and select the VMEM fallback.
    function dim_kern!(A)
        i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
        j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
        k = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z
        n = i + j + k
        n <= length(A) && (@inbounds A[n] = n)
        return
    end

    iob = IOBuffer()
    tt = Tuple{AMDGPU.Device.ROCDeviceVector{Float32, AMDGPU.Device.AS.Global}}
    AMDGPU.code_gcn(iob, dim_kern!, tt; kernel=true)
    gcn = String(take!(iob))
    @test occursin("s_load", gcn)
    @test !occursin("global_load", gcn)
    @test !occursin("flat_load", gcn)

@testset "group_size sizes the register budget" begin
    # Without a declared workgroup size the backend must assume a group may be
    # as large as the 1024-workitem maximum, which caps the kernel at 128 VGPRs
    # (1024 workitems is 16 wave64s over 4 SIMDs, so 4 waves per SIMD, and
    # 512 / 4 = 128). A kernel needing more than that then spills to scratch
    # for an occupancy it never actually runs at. `group_size` states the size
    # the kernel is really launched with, so the budget matches reality.
    function scale_kern!(A)
        i = workitemIdx().x
        @inbounds A[i] = A[i] * 2
        return
    end

    tt = Tuple{AMDGPU.Device.ROCDeviceVector{Float64, AMDGPU.Device.AS.Global}}
    function gcn(; kwargs...)
        source = AMDGPU.methodinstance(typeof(scale_kern!), tt)
        config = AMDGPU.Compiler.compiler_config(
            AMDGPU.HIP.device(); kernel=true, kwargs...)
        iob = IOBuffer()
        AMDGPU.GPUCompiler.code_native(
            iob, AMDGPU.GPUCompiler.CompilerJob(source, config))
        String(take!(iob))
    end

    # the declared size reaches the kernel's metadata, and bounds the register
    # budget the backend allocates against
    @test occursin("max_flat_workgroup_size: 256", gcn(group_size=256))
    # unspecified by default: the backend keeps assuming the 1024 maximum
    @test occursin("max_flat_workgroup_size: 1024", gcn())
end
