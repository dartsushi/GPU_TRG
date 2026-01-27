using TensorKit, JLD2, MPSKitModels

## TEBD ##
function ind_pair(T::AbstractTensorMap, p::Tuple)
    p2 = filter(x -> !in(x, p), allind(T))
    return p, p2
end

# QR decomposition
function QR_two_pepo_left(O1::AbstractTensorMap, O2::AbstractTensorMap, ind::Int)
    """ Find R acting on the indices ind of O1 and O2 """
    pb = (1, ind)
    p, q1 = ind_pair(O1, pb)
    _, Rb = leftorth(O1, (q1, p))
    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    _, Rt = leftorth(O2, (q2, p))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, ((3, 4), (1, 2)))
    return R
end

function QR_two_pepo_right(O1::AbstractTensorMap, O2::AbstractTensorMap, ind::Int)
    """ Find R acting on the indices ind of O1 and O2 """
    pb = (1, ind)
    p, q2 = ind_pair(O1, pb)
    Rb, _ = rightorth(O1, (p, q2))
    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    Rt, _ = rightorth(O2, (p, q2))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1; -3] * Rb[1 -2; -4]
    R, _ = rightorth(M, ((1, 2), (3, 4)))
    return R
end

function QR_two_pepo(O1::AbstractTensorMap, O2::AbstractTensorMap, ind::Int; side = :left)
    """ Find R acting on the indices ind of O1 and O2 """
    if side == :left
        return QR_two_pepo_left(O1, O2, ind)
    elseif side == :right
        return QR_two_pepo_right(O1, O2, ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(A1::AbstractTensorMap, A2::AbstractTensorMap, ind1::Int, ind2::Int; check_space = true)
    """ Find R1 and R2 acting on indices ind1 of A1 and ind2 of A2 """
    RA1 = QR_two_pepo(A1, A2, ind1)
    RA2 = QR_two_pepo(A1, A2, ind2; side = :right)
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

# Find the pair of oblique projectors acting on the indices p1 of A1 and p2 of A2
#=
   ┌──┐        ┌──┐   
   │  ├◄──  ─◄─┤  │   
─◄─┤P1│        │P2├◄──
   │  ├◄──  ─◄─┤  │   
   └──┘        └──┘   
=#

function find_P1P2(A1::AbstractTensorMap, A2::AbstractTensorMap, ind1::Int, ind2::Int, trunc; check_space = true)
    """ Find oblique projectors P1 and P2 acting on indices ind1 of A1 and ind2 of A2 """
    R1, R2 = R1R2(A1, A2, ind1, ind2; check_space = check_space)
    return oblique_projector(R1, R2, trunc)
end

function oblique_projector(R1::AbstractTensorMap, R2::AbstractTensorMap, trunc; cutoff = 1e-16)
    """ Find oblique projectors P1 and P2 from R1 and R2"""
    mat = R1 * R2
    U, S, Vt = svd_trunc(mat; trunc = trunc & truncbelow(cutoff))

    P1 = R2 * adjoint(Vt) / sqrt(S)
    P2 = adjoint(U) * R1
    P2 = adjoint(adjoint(P2) / adjoint(sqrt(S)))
    return P1, P2
end

function tr_tensor(T::AbstractTensorMap; inv = false)
    """ Trace over physical indices of tensor T """
    if inv
        @tensoropt tr4 = T[1 2; 3 4] * conj(T[5 2; 3 6]) * conj(T[1 7; 8 4]) * T[5 7; 8 6]
        return (abs(tr4))^(1 / 4)
    else
        return @tensor T[1 2; 2 1]
    end
end

function tr_pepo(T)
    """ Trace over physical indices of PEPO T """
    return @tensor T[1 1; 2 3 2 3]
end

### pepo TEBD

function update_O!(O::AbstractTensorMap, δO::AbstractTensorMap, step::Int, trunc; verbosity = 0, normalize = false)
    """ Perform TEBD update on PEPO O with gate δO for 'step' times.
        Returns the list of O at each step.
    """
    O_list = [O]
    lnz = 0
    @showprogress for i = 1:step
        if verbosity > 0
            println(i, " / ", step)
        end
        P59, P37 = find_P1P2(O, δO, 5, 3, trunc)
  
        P48, P610 = find_P1P2(O, δO, 4, 6, trunc)
     
        @tensor opt = true Onew[-1 -2; -3 -4 -5 -6] :=
            O[1 -2; 7 8 9 10] *
            δO[-1 1; 3 4 5 6] *
            P610[-6; 6 10] *
            P37[-3; 3 7] *
            P48[4 8; -4] *
            P59[5 9; -5]
        O = Onew
        if normalize
            n = tr_pepo(O)
            lnz += log(abs(n))
            O /= abs(n)
        end
        push!(O_list, O)
    end
    if normalize
        return O_list, lnz
    else
        return O_list
    end
end


function to_2d(O::AbstractTensorMap)
    """ Convert PEPO to 2D tensor by contracting physical index with identity """
    m = id(space(O)[1])
    # TNRKit, MPSKit notation
    @tensor O2d[-1 -2; -4 -3] := O[1 2; -1 -2 -3 -4] * m[2; 1]
    return O2d
end