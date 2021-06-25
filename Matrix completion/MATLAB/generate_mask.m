function [H,omega,omega_2d,flag] = generate_mask(n1,n2,nv,r,max_resapmles)

%%% code taken from MatrixIRLS by Christian Kuemmerle %%%

%sample_phi_MatrixCompletion. This function randomly samples a (d1 x d2)
%sparse matrix with ones at m randomly chosen coordinates (uniform without
%replacement). If the 'resample' mode is chosen, this sampling procedure is
%repeated until Phi has at least r non-zero entries in each row and each
%column, where r is a specified positive integer, but this resampling
%procedure is repeated at most max_nr_resample times.

%           Input:
%           n1      = number of rows of Phi
%           n2      = number of columns of Phi 
%           nv       = number of nonzero entries of Phi
%           r       = rank
%     max_resapmles = upper bound on number of resamplings.
%     random_seed
%
%           Output:
%           Phi     = (d1 x d2) sparse matrix: completion mask with 
%                       ones at "seen" entries and
%                       zeros at "unseen" entries.
%         Omega     = (m x 1) vector with linear indices of non-zero
%                       entries of Phi.

rejec_counter=0;
reject=1;
while (reject==1)
    omega = (sort(randperm(n1*n2,nv)))';
    i_Omega=mod(omega,n1);
    i_Omega(i_Omega==0)=n1;
    j_Omega=floor((omega-1)/n1)+1;
    H = sparse(i_Omega,j_Omega,ones(nv,1),n1,n2);
    nr_entr_col = sum(H,1)';
    nr_entr_row = sum(H,2);

    if (isempty(find(nr_entr_row<r+1,1)) == 0) || (isempty(find(nr_entr_col<r+1,1)) == 0)
        rejec_counter=rejec_counter+1;
        if rem(rejec_counter, 1e5) == 0
            disp(['mask counter: ', num2str(rejec_counter)])
        end
    else
        reject=0;
    end
    if rejec_counter >= max_resapmles
        disp('No mask found!');
        break
    end
end
%if rejec_counter >= 1
%    disp(['Rejection counter of sampled matrix completion masks with ',num2str(m),' entries: ',num2str(rejec_counter)]);
%end

omega_2d = zeros(nv,2);
[omega_2d(:,1), omega_2d(:,2)] = ind2sub([n1,n2],omega);
flag = ~reject;

end
