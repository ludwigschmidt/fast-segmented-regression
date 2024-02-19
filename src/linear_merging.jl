using Printf
using LinearAlgebra
using StatsBase

struct LinearPiece
  left_index::Int
  right_index::Int
  theta::Array{Float64,1}
end


function fit_linear_piece(X::Array{Float64,2}, y::Array{Float64,1}, left_index::Int, right_index::Int)
  theta = X[left_index:right_index, :] \ y[left_index:right_index]
  return LinearPiece(left_index, right_index, theta)
end

function piece_length(p::LinearPiece)
  return p.right_index - p.left_index + 1
end


function linear_piece_merging_error(p::LinearPiece, X::Array{Float64,2}, y::Array{Float64,1}, sigma::Float64)
  return linear_piece_error(p, X, y) - piece_length(p) * sigma^2
end


function linear_piece_error(p::LinearPiece, X::Array{Float64,2}, y::Array{Float64,1})
  return norm(y[p.left_index : p.right_index] - X[p.left_index:p.right_index, :] * p.theta)^2
end

function linear_fit_error(X::Array{Float64,2}, y::Array{Float64,1}, left_index::Int, right_index::Int)
  p = fit_linear_piece(X, y, left_index, right_index)
  return linear_piece_error(p, X, y)
end


function generate_equal_size_linear_data(endpoint_values, n, sigma)
  # TODO: change
  X = zeros(Float64, (n, 2))
  X[:,1] = ones(Float64, n)
  X[:, 2] = collect(range(0.0, stop=1.0, length=n))
  
  num_segments = length(endpoint_values) - 1
  num_per_bin = floor(Int, n / num_segments)
  num_bins_plusone = n % num_segments

  ystar = Array{Float64}(undef,0)
  for ii = 1:num_bins_plusone
    append!(ystar, collect(range(endpoint_values[ii], stop=endpoint_values[ii + 1], length=num_per_bin + 1)))
  end
  for ii = (num_bins_plusone + 1):num_segments
    append!(ystar, collect(range(endpoint_values[ii], stop=endpoint_values[ii + 1], length=num_per_bin)))
  end

  y = ystar + sigma * randn(n)
  return y, ystar, X
end


function generate_equal_size_random_regression_data(num_segments, n, d, sigma)
  X = randn(n, d)

  num_per_bin = floor(Int, n / num_segments)
  num_bins_plusone = n % num_segments
  
  ystar = Array{Float64}(undef, 0)
  cur_start = 1
  for ii = 1 : num_bins_plusone
    cur_end = cur_start + num_per_bin
    beta = 2 * rand(Float64, d) .+ 1
    append!(ystar, vec(X[cur_start:cur_end,:] * beta))
    cur_start = cur_end + 1
  end
  for ii = (num_bins_plusone + 1) : num_segments
    cur_end = cur_start + num_per_bin - 1
    beta = 2 * rand(Float64, d) .+ 1
    append!(ystar, vec(X[cur_start:cur_end,:] * beta))
    cur_start = cur_end + 1
  end
  y = ystar + sigma * randn(n)
  return y, ystar, X
end


function partition_to_vector(X::Array{Float64,2}, pieces::Array{LinearPiece,1})
  n = pieces[end].right_index
  (rows, cols) = size(X)
  if n != rows
    error("number of rows and rightmost index must match")
  end
  y = Array{Float64}(undef, n)
  for ii = 1 : length(pieces)
    p = pieces[ii]
    y[p.left_index : p.right_index] = X[p.left_index : p.right_index, :] * p.theta
  end
  return y
end


function mse(yhat, ystar)
  return (1.0 / length(yhat)) * sum((yhat - ystar).^2)
end


function compute_errors_fast(X::Array{Float64,2}, y::Array{Float64,1})
  (n, d) = size(X)
  res = -ones(Float64, (n, n))

  for ii = 1:n
    normysq = 0.0
    A = zeros(d, d)
    for jj = ii:min(ii + d - 1, n)
      normysq += y[jj] * y[jj]
      A += X[jj,:] * X[jj,:]'
      theta = X[ii:jj, :] \ y[ii:jj]
      res[ii,jj] = normysq - dot(vec(theta' * A), theta)
    end
    if ii + d - 1 < n
      Ainv = inv(A)
      y2 = X[ii:ii+d-1,:]' * y[ii:ii+d-1]
      for jj = ii+d:n
        currow = vec(X[jj,:])
        normysq += y[jj] *  y[jj]
        A += X[jj,:] * X[jj,:]'
        Ainv -= (Ainv * currow * currow' * Ainv) / (1 + dot(vec(currow' * Ainv), currow))
        y2 += currow * y[jj]
        theta = Ainv * y2
        res[ii,jj] = normysq - dot(vec(theta' * A), theta)
      end
    end
  end
  return res
end


function fit_linear_dp(X::Array{Float64,2}, y::Array{Float64,1}, num_target_pieces::Int)
  n = length(y)
  k = num_target_pieces
  min_error = zeros(Float64, (n, k))
  pred = zeros(Int, (n, k))
  
  fit_error = compute_errors_fast(X, y)

  # Initialization of DP boundary
  for ii = 1:n
    min_error[ii, 1] = fit_error[1, ii]
  end

  for ii = 1:k
    min_error[1, ii] = 0.0
  end

  # Dynamic program
  for ii = 2:k
    for jj = 2:n
      min_error[jj, ii] = Inf
      for ll = 1:(jj-1)
        new_error = min_error[ll, ii-1] + fit_error[ll + 1, jj]
        if new_error < min_error[jj, ii]
          min_error[jj, ii] = new_error
          pred[jj, ii] = ll
        end
      end
    end
  end

  # Reconstruct solution
  sol = Array{LinearPiece}(undef, k)
  cur_pos = n
  for ii = 1:k
    if cur_pos == 0
      @printf("WARNING: ii = %d, cur_pos = 0\n", ii)
#      @printf("ii = %d  cur_pos = %d  pred = %d\n", ii, cur_pos, pred[cur_pos, k - ii + 1] + 1)
      break
    end
    cur_right = cur_pos
    cur_left = pred[cur_right, k - ii + 1] + 1
    sol[k - ii + 1] = fit_linear_piece(X, y, cur_left, cur_right)
    cur_pos = cur_left - 1
  end
  return sol
end


function fit_linear_merging(X::Array{Float64,2}, y::Array{Float64,1}, sigma::Float64, num_target_pieces::Int, num_holdout_pieces::Int; initial_merging_size::Int=-1)

  n = length(y)
  if initial_merging_size <= 0
    initial_merging_size = Int(sqrt(sqrt(n)))
  end

  # Initial partition
  cur_pieces = Array{LinearPiece}(undef, 0)
  num_remaining = n
  cur_left = 1
  while num_remaining > 0
    cur_right = min(cur_left + initial_merging_size - 1, n)
    num_remaining -= initial_merging_size
    tmp_piece = fit_linear_piece(X, y, cur_left, cur_right)
    push!(cur_pieces, tmp_piece)
    cur_left = cur_right + 1
  end
  prev_pieces = Array{LinearPiece}(undef, 0)
  
  while length(cur_pieces) > num_target_pieces && length(cur_pieces) != length(prev_pieces)
    prev_pieces = cur_pieces
    cur_pieces = Array{LinearPiece}(undef, 0)

    # Create a list of merging candidates and compute their errors
    candidate_pieces = Array{LinearPiece}(undef, 0)
    candidate_errors = Array{Float64}(undef, 0)
    for ii = 1:floor(Int, length(prev_pieces) / 2)
      left_piece = prev_pieces[2 * ii - 1]
      right_piece = prev_pieces[2 * ii]
      new_piece = fit_linear_piece(X, y, left_piece.left_index, right_piece.right_index)
      new_error = linear_piece_merging_error(new_piece, X, y, sigma)
      push!(candidate_pieces, new_piece)
      push!(candidate_errors, new_error)
    end

    # For an odd number of pieces, we directly include the last piece as a
    # candidate.
    if length(prev_pieces) % 2 == 1
      last_piece = prev_pieces[end]
      last_error = linear_piece_merging_error(last_piece, X, y, sigma)
      push!(candidate_pieces, last_piece)
      push!(candidate_errors, last_error)
    end

    # Select the num_holdout_pieces'th largest error (counting from the largest
    # error down) as threshold.
    sorted_errors = sort(candidate_errors)
    error_threshold = sorted_errors[max(1, length(candidate_pieces) - num_holdout_pieces + 1)]


    # Count how many of the intervals are exactly at the threshold to avoid
    # corner cases.
    num_at_threshold = 0
    num_above_threshold = 0
    for ii = 1:length(candidate_pieces)
      if candidate_errors[ii] == error_threshold
        num_at_threshold += 1
      elseif candidate_errors[ii] > error_threshold
        num_above_threshold += 1
      end
    end
    num_at_threshold_to_include = num_holdout_pieces - num_above_threshold

    # Form the new partition
    for ii = 1:length(candidate_pieces)
      # Use the merge candidate if the error is small enough.
      if candidate_errors[ii] < error_threshold
        push!(cur_pieces, candidate_pieces[ii])
      # If the error is exactly at the threshold, we make a special check
      elseif candidate_errors[ii] == error_threshold && num_at_threshold_to_include > 0
        num_at_threshold_to_include -= 1
        push!(cur_pieces, candidate_pieces[ii])
      else
        # Otherwise, include the original pieces
        push!(cur_pieces, prev_pieces[2 * ii - 1])
        # Corner case for the last candidate piece, which might not be the
        # result of a merge
        if 2 * ii <= length(prev_pieces)
          push!(cur_pieces, prev_pieces[2 * ii])
        end
      end
    end
  end

  return cur_pieces
end
