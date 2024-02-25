using Printf
using Statistics
using LinearAlgebra
using StatsBase

struct HistogramPiece
  left_index::Int
  right_index::Int
  theta::Float64
end


function piece_length(p::HistogramPiece)
  return p.right_index - p.left_index + 1
end


function histogram_piece_merging_error(p::HistogramPiece, ys::Vector{Float64}, sigma::Float64)
  return norm(ys[p.left_index : p.right_index] .- p.theta)^2 - piece_length(p) * sigma^2
end


struct HistogramStatistics
  sum::Array{Float64, 1}
  sum_sq::Array{Float64, 1}
end


function HistogramStatistics(ys::Array{Float64, 1})
  n = length(ys)
  sum = Array{Float64}(undef, n + 1)
  sum_sq = Array{Float64}(undef, n + 1)
  sum[1] = 0.0
  sum_sq[1] = 0.0
  for ii = 1:n
    sum[ii + 1] = ys[ii] + sum[ii]
    sum_sq[ii + 1] = ys[ii]^2 + sum_sq[ii]
  end
  return HistogramStatistics(sum, sum_sq)
end


function hist_stats_mean(s::HistogramStatistics, left::Int, right::Int)
  return (s.sum[right + 1] - s.sum[left]) / (right - left + 1)
end


function hist_stats_variance(s::HistogramStatistics, left::Int, right::Int)
  return (s.sum_sq[right + 1] - s.sum_sq[left]) - (1.0 / (right - left + 1)) * (s.sum[right + 1] - s.sum[left])^2
end


function generate_equal_size_histogram_data(bin_values, n, sigma)
  num_per_bin = floor(Int, n / length(bin_values))
  num_bins_plusone = n % length(bin_values)
  ystar = Array{Float64}(undef, 0)
  for ii = 1 : num_bins_plusone
    append!(ystar, bin_values[ii] * ones(num_per_bin + 1))
  end
  for ii = (num_bins_plusone + 1) : length(bin_values)
    append!(ystar, bin_values[ii] * ones(num_per_bin))
  end
  y = ystar + sigma * randn(n)
  return y, ystar
end


function partition_to_vector(pieces::Array{HistogramPiece, 1})
  n = pieces[end].right_index
  y = Array{Float64}(undef, n)
  for ii = 1 : length(pieces)
    p = pieces[ii]
    y[p.left_index : p.right_index] .= p.theta
  end
  return y
end


function mse(yhat, ystar)
  return (1.0 / length(yhat)) * sum((yhat - ystar).^2)
end


function fit_histogram_dp(ys::Array{Float64, 1}, num_target_pieces::Int)
  n = length(ys)
  k = num_target_pieces
  min_error = zeros(Float64, (n, k))
  pred = zeros(Int, (n, k))
  stats = HistogramStatistics(ys)

  # Initialization of DP boundary
  for ii = 1:n
    min_error[ii, 1] = hist_stats_variance(stats, 1, ii)
  end

  for ii = 1:k
    min_error[1, ii] = 0.0
  end

  # Dynamic program
  for ii = 2:k
    for jj = 2:n
      min_error[jj, ii] = Inf
      for ll = 1:(jj-1)
        new_error = min_error[ll, ii-1] + hist_stats_variance(stats, ll + 1, jj)
        if new_error < min_error[jj, ii]
          min_error[jj, ii] = new_error
          pred[jj, ii] = ll
        end
      end
    end
  end

  # Reconstruct solution
  sol = Array{HistogramPiece}(undef, k)
  cur_pos = n
  for ii = 1:k
    if cur_pos == 0
      @printf("WARNING: ii = %d, cur_pos = 0\n", ii)
#      @printf("ii = %d  cur_pos = %d  pred = %d\n", ii, cur_pos, pred[cur_pos, k - ii + 1] + 1)
      break
    end
    cur_right = cur_pos
    cur_left = pred[cur_right, k - ii + 1] + 1
    cur_theta = hist_stats_mean(stats, cur_left, cur_right)
    sol[k - ii + 1] = HistogramPiece(cur_left, cur_right, cur_theta)
    cur_pos = cur_left - 1
  end
  return sol
end


function fit_histogram_merging(ys::Array{Float64, 1}, sigma::Float64, num_target_pieces::Int, num_holdout_pieces::Int; initial_merging_size::Int=-1)

  n = length(ys)
  if initial_merging_size <= 0
    initial_merging_size = Int(sqrt(sqrt(n)))
  end

  # Initial partition
  cur_pieces = Array{HistogramPiece}(undef, 0)
  num_remaining = n
  cur_left = 1
  while num_remaining > 0
    cur_right = min(cur_left + initial_merging_size - 1, n)
    num_remaining -= initial_merging_size
    cur_theta = mean(ys[cur_left:cur_right])
    tmp_piece = HistogramPiece(cur_left, cur_right, cur_theta)
    push!(cur_pieces, tmp_piece)
    cur_left = cur_right + 1
  end
  prev_pieces = Array{HistogramPiece}(undef, 0)
  
  while length(cur_pieces) > num_target_pieces && length(cur_pieces) != length(prev_pieces)
    prev_pieces = cur_pieces
    cur_pieces = Array{HistogramPiece}(undef, 0)

    # Create a list of merging candidates and compute their errors
    candidate_pieces = Array{HistogramPiece}(undef, 0)
    candidate_errors = Array{Float64}(undef, 0)
    for ii = 1:floor(Int, length(prev_pieces) / 2)
      left_piece = prev_pieces[2 * ii - 1]
      right_piece = prev_pieces[2 * ii]
      new_theta = mean(ys[left_piece.left_index : right_piece.right_index])
      new_piece = HistogramPiece(left_piece.left_index, right_piece.right_index, new_theta)
      new_error = histogram_piece_merging_error(new_piece, ys, sigma)
      push!(candidate_pieces, new_piece)
      push!(candidate_errors, new_error)
    end

    # For an odd number of pieces, we directly include the last piece as a
    # candidate.
    if length(prev_pieces) % 2 == 1
      last_piece = prev_pieces[end]
      last_error = histogram_piece_merging_error(last_piece, ys, sigma)
      push!(candidate_pieces, last_piece)
      push!(candidate_errors, last_error)
    end

    # Select the num_holdout_pieces'th largest error (counting from the largest
    # error down) as threshold.
    error_threshold = partialsort(candidate_errors, max(0, length(candidate_pieces) - num_holdout_pieces + 1))
   
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
