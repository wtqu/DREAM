function K = transpose_derivative(Y)
    % This function calculates the exchange matrix K hat computes the derivative
    % of Y^T with respect to Y, where Y is an m x n matrix.
    %
    % Input:
    %   Y: n x p
    %
    % Returns:
    %   K: The exchange matrix of size (p*n) x (n*p).
    
    % Get the dimensions of Y
    [n, p] = size(Y);
    
    % Initialize the K matrix with zeros
    K = zeros(p * n, n * p);
    
    % Construct the permutation matrix K
    for i = 1:n
        for j = 1:p
            % Calculate the linear index for (i, j) in the original matrix Y (n x p)
            row_index = (j - 1) * n + i;
            % Calculate the linear index for (j, i) in the transposed matrix Y^T (p x n)
            col_index = (i - 1) * p + j;
            K(row_index, col_index) = 1;
        end
    end
end
