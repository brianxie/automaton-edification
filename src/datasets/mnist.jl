#=
the IDX file format is a simple format for vectors and multidimensional matrices
of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for
vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in
most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension
changes the fastest.
=#

module MNIST

export read_file_from_index

using CodecZlib

TRAINING_SAMPLES_PATH = "datasets/mnist/train-images-idx3-ubyte.gz"
TRAINING_LABELS_PATH = "datasets/mnist/train-labels-idx1-ubyte.gz"

TEST_SAMPLES_PATH = "datasets/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_PATH = "datasets/mnist/t10k-labels-idx1-ubyte.gz"

"""
    read_file_from_index(filename; sample_index, num_samples)

Reads samples from `filename`.

- `sample_index`: index of the first sample to read, or the beginning of the
  file if not provided.
- `num_samples`: the number of samples to read, or the entire file if not
  provided.

Returns a vector of tensors, each of which represents a single sample.
"""
function read_file_from_index(filename::String;
                              sample_index=missing, num_samples=missing)
    open(GzipDecompressorStream, filename, "r") do stream
        tensor_dims, current_pos = read_header(stream)

        # tensor_dims[1] is assumed to represent the number of samples, with the
        # remaining fields (of which there may be none) representing the number
        # of dimensions and the size along each dimension.
        total_num_samples = tensor_dims[1]
        sample_dims = tensor_dims[2:end]

        if ismissing(sample_index) && ismissing(num_samples)
            # Read all samples from the beginning of the data.
            samples, current_pos = read_samples(stream, sample_dims, total_num_samples)

            # Make sure that we've actually read the whole file.
            seekend(stream)
            end_pos = position(stream)
            @assert current_pos == end_pos
        elseif !ismissing(sample_index) && !ismissing(num_samples)
            # Read num_samples from the offset, if valid.
            @assert 1 <=
                sample_index <=
                sample_index + num_samples - 1 <=
                total_num_samples
            
            target_offset = (sample_index - 1) * prod(sample_dims)
            skip(stream, target_offset)

            samples, current_pos = read_samples(stream, sample_dims, num_samples)
        else
            error("sample_index and num_samples must both be provided, or neither at all.")
        end

        return samples
    end
end

"""
    read_header(stream)

Reads the IDX header from a file stream.

Returns the tensor dimensions of the data, and the stream position of the
beginning of the data; also seeks the stream to that position.
"""
function read_header(stream::IO)
    seekstart(stream)

    magic_number = maybe_bswap(read(stream, UInt32))
    @assert (magic_number & 0xFFFF0000) == 0

    datatype = (magic_number >>> 8) & 0x000000FF
    tensor_order = magic_number & 0x000000FF

    tensor_dims = Vector{UInt32}(undef, tensor_order)
    for i in 1:tensor_order
        tensor_dims[i] = maybe_bswap(read(stream, UInt32))
    end

    return Tuple(tensor_dims), position(stream)
end

"""
    read_samples(stream, sample_dims, num_samples)

Reads `num_samples` samples from `stream`; does not perform bounds checking.

- `stream`: Stream from which to read, assumed to already be at the correct
  offset.
- `sample_dims`: Tuple containing the size of each sample along each axis;
  length(sample_dims) represents the number of axes.
- `num_samples`: Number of samples to read.

Returns a vector of samples, and the updated stream position after reading the
samples. Also seeks the stream to that position.
"""
function read_samples(stream::IO, sample_dims::Tuple, num_samples::Integer)
    # Each sample has tensor order length(sample_dims).
    samples = Vector{Array{UInt8, length(sample_dims)}}(undef, num_samples)

    for i in 1:num_samples
        samples[i] = read_as_sample(stream, sample_dims)
    end

    return samples, position(stream)
end

"""
    read_as_sample(stream, sample_dims)

Reads a single sample from `stream`; does not perform bounds checking.

- `stream`: Stream from which to read, assumed to already be at the correct
  offset; seeking on each individual sample would be expensive, since
  decompressed streams may not support arbitrary seeking.
- `sample_dims`: Tuple containing the size along each axis; length(sample_dims)
  represents the number of axes.
"""
function read_as_sample(stream::IO, sample_dims::Tuple)
    # Base.prod(...) nicely handles empty tuples by returning 1 (multiplicative
    # identity).
    return reshape(read(stream, prod(sample_dims)), # Vector
                   sample_dims) # Reshaped to a length(sample_dims)-order tensor
end

"""
    maybe_bswap(bytes)

Return `bytes` in reversed order if on a little-endian machine, otherwise return
`bytes`

Guarantees that the output is big-endian.
"""
function maybe_bswap(bytes)
    return ENDIAN_BOM == 0x04030201 ? bswap(bytes) : bytes
end


"""
    get_datatype(bytes)

Returns a string representing the datatype of the data.

Throws an error if the mapping is invalid.
"""
function get_datatype(bytes)
    datatypes = Dict(
        0x08=>"unsigned byte",
        0x09=>"signed byte",
        0x0B=>"short",
        0x0C=>"int",
        0x0D=>"float",
        0x0E=>"double"
    )
    return datatypes[bytes]
end

end # module
