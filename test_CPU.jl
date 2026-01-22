""" This is a test for CPU functionality 
    with finite-T PEPO project """

using JLD2, Revise
using TensorKit
using MPSKitModels, ProgressMeter

include("tools.jl");

filename_save = "test_CPU_data.jld2";
Dict_save = Dict();

# J1 = 1.0;
# J2 = 0.5;
# δτ = 0.1;
# χ_trunc = 10;
# N_steps = 10;

J1 = parse(Float64, ARGS[1])
J2 = parse(Float64, ARGS[2])
δτ = parse(Float64, ARGS[3])
N_steps = parse(Int, ARGS[4])
χ_trunc = parse(Int, ARGS[5])


Dict_save["J1"] = J1;
Dict_save["J2"] = J2;
Dict_save["δτ"] = δτ;
Dict_save["χ_trunc"] = χ_trunc;
Dict_save["N_steps"] = N_steps;

save(filename_save, Dict_save);

#### prepare trotter PEPO's for given J1, J2 and δτ ###

function make_J1J2(J1::Float64, J2::Float64, τ::Float64)

    h = real(S_exchange(SU2Irrep))
    H = exp(-τ*h*J1)
    H = permute(H,((1,3),(2,4)))
    U,S,V = svd_full(H)
    L,R = U*sqrt(S), sqrt(S)*V
    @tensor pepo[-1 -2; -3 -4 -5 -6] := L[-2 1;-3] * R[-5; 1 2] * L[2 3; -4]  * R[-6; 3 -1];

    # J2 part
    h = real(S_exchange(SU2Irrep))
    H = exp(-τ*J2*h)
    H = permute(H,((1,3),(2,4)))
    U,S,V = svd_full(H)
    L,R = U*sqrt(S), sqrt(S)*V

    mid_corner = id(Rep[SU₂](0=>1, 1=>1))
    @tensor pepo1_j2[-1 -2; -3 -4 -5 -6] := L[1 -1;-3] * R[-4; -2 1] * mid_corner[-5; -6];
    @tensor pepo2_j2[-1 -2; -4 -5 -6 -3] := L[1 -1;-3] * R[-4; -2 1] * mid_corner[-5; -6];

    return [pepo, pepo1_j2, pepo2_j2]
end

function get_free_energy(O_pepo::AbstractTensorMap, trunc)
    """Compute the free energy from the PEPO"""
    return 1.0
end

function get_energy(O_pepo::AbstractTensorMap, trunc)
    """Compute the energy from the PEPO"""
    return 1.0
end

function get_lnz(O_pepo::AbstractTensorMap, trunc)
    """Compute the log partition function from the PEPO"""
    return 1.0
end

function main()
    
    δpepo_list = make_J1J2(J1, J2, δτ);
    β_list = real((1:(N_steps))*δτ)

    Dict_save["pepo_list_initial"] = δpepo_list;
    Dict_save["β_list"] = β_list;
    Dict_save["free_energy"] = Float64[];
    Dict_save["energy"] = Float64[];
    Dict_save["lnz"] = Float64[];

    save(filename_save, Dict_save);


    ### perform TEBD steps ###

    V_dummy = Rep[SU₂](0=>1);
    V_phys = Rep[SU₂](1/2 => 1);
    O_main = TensorMap(ones, V_phys' ⊗ V_phys ← V_dummy ⊗ V_dummy ⊗ 
                                            V_dummy' ⊗ V_dummy');

    @showprogress for i=1:N_steps
        for _ in 1:2 # have two consecutive δτ/2 steps
            for j in 1:3
                    δO = δpepo_list[j]
                    O_main = update_O!(O_main, δO, 1, truncdim(χ_trunc))[end]
            end

            O_main = permute(O_main, ((1,2), (4,3,6,5))); # permute for next half step
        end

        # record test_CPU_data 
        F = get_free_energy(O_main, χ_trunc);
        E = get_energy(O_main, χ_trunc);
        lnz = get_lnz(O_main, χ_trunc);     

        Dict_save["free_energy"] = push!(get(Dict_save, "free_energy", Float64[]), F);
        Dict_save["energy"] = push!(get(Dict_save, "energy", Float64[]), E);
        Dict_save["lnz"] = push!(get(Dict_save, "lnz", Float64[]), lnz);

        save(filename_save, Dict_save);
    end

    @show "simulation complete!"
end

main()