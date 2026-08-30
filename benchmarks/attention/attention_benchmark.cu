// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/attention.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void causal_softmax_forward_cpu(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float maximum = scale * logits[row_offset];
                for (int key = 1; key <= query; ++key) {
                    maximum = std::max(maximum, scale * logits[row_offset + key]);
                }

                float denominator = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    const float exponential =
                        std::exp(scale * logits[row_offset + key] - maximum);
                    probabilities[row_offset + key] = exponential;
                    denominator += exponential;
                }

                for (int key = 0; key <= query; ++key) {
                    probabilities[row_offset + key] /= denominator;
                }
                for (int key = query + 1; key < sequence_length; ++key) {
                    probabilities[row_offset + key] = 0.0F;
                }
            }
        }
    }
}

void causal_softmax_backward_cpu(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float projection = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    projection += probabilities_gradient[row_offset + key] *
                                  probabilities[row_offset + key];
                }

                for (int key = 0; key <= query; ++key) {
                    const int index = row_offset + key;
                    logits_gradient[index] +=
                        scale * probabilities[index] *
                        (probabilities_gradient[index] - projection);
                }
            }
        }
    }
}


namespace {

int activation_index(
    int batch,
    int token,
    int head,
    int column,
    int sequence_length,
    int heads,
    int head_size) {
    return ((batch * sequence_length + token) * heads + head) * head_size +
           column;
}

int score_index(
    int batch,
    int head,
    int query,
    int key,
    int heads,
    int sequence_length) {
    return ((batch * heads + head) * sequence_length + query) *
               sequence_length +
           key;
}

}  // namespace

void dense_attention_forward_cpu(
    float* output,
    float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale) {
    const int score_elements =
        batch_size * heads * sequence_length * sequence_length;
    std::vector<float> scores(score_elements);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            for (int query_token = 0; query_token < sequence_length;
                 ++query_token) {
                for (int key_token = 0; key_token < sequence_length;
                     ++key_token) {
                    float dot = 0.0F;
                    for (int column = 0; column < head_size; ++column) {
                        dot += query[activation_index(
                                   batch,
                                   query_token,
                                   head,
                                   column,
                                   sequence_length,
                                   heads,
                                   head_size)] *
                               key[activation_index(
                                   batch,
                                   key_token,
                                   head,
                                   column,
                                   sequence_length,
                                   heads,
                                   head_size)];
                    }
                    scores[score_index(
                        batch,
                        head,
                        query_token,
                        key_token,
                        heads,
                        sequence_length)] = dot;
                }
            }
        }
    }

    causal_softmax_forward_cpu(
        probabilities,
        scores.data(),
        batch_size,
        heads,
        sequence_length,
        scale);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int token = 0; token < sequence_length; ++token) {
            for (int head = 0; head < heads; ++head) {
                for (int column = 0; column < head_size; ++column) {
                    float result = 0.0F;
                    for (int key_token = 0; key_token < sequence_length;
                         ++key_token) {
                        result += probabilities[score_index(
                                      batch,
                                      head,
                                      token,
                                      key_token,
                                      heads,
                                      sequence_length)] *
                                  value[activation_index(
                                      batch,
                                      key_token,
                                      head,
                                      column,
                                      sequence_length,
                                      heads,
                                      head_size)];
                    }
                    output[activation_index(
                        batch,
                        token,
                        head,
                        column,
                        sequence_length,
                        heads,
                        head_size)] = result;
                }
            }
        }
    }
}

void dense_attention_backward_cpu(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale) {
    const int score_elements =
        batch_size * heads * sequence_length * sequence_length;
    std::vector<float> probability_gradient(score_elements, 0.0F);
    std::vector<float> score_gradient(score_elements, 0.0F);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            for (int query_token = 0; query_token < sequence_length;
                 ++query_token) {
                for (int key_token = 0; key_token < sequence_length;
                     ++key_token) {
                    float dot = 0.0F;
                    for (int column = 0; column < head_size; ++column) {
                        dot += output_gradient[activation_index(
                                   batch,
                                   query_token,
                                   head,
                                   column,
                                   sequence_length,
                                   heads,
                                   head_size)] *
                               value[activation_index(
                                   batch,
                                   key_token,
                                   head,
                                   column,
                                   sequence_length,
                                   heads,
                                   head_size)];
                    }
                    probability_gradient[score_index(
                        batch,
                        head,
                        query_token,
                        key_token,
                        heads,
                        sequence_length)] = dot;
                }
            }
        }
    }

    causal_softmax_backward_cpu(
        score_gradient.data(),
        probability_gradient.data(),
        probabilities,
        batch_size,
        heads,
        sequence_length,
        scale);

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            for (int token = 0; token < sequence_length; ++token) {
                for (int column = 0; column < head_size; ++column) {
                    float query_update = 0.0F;
                    float key_update = 0.0F;
                    float value_update = 0.0F;
                    for (int other = 0; other < sequence_length; ++other) {
                        query_update += score_gradient[score_index(
                                            batch,
                                            head,
                                            token,
                                            other,
                                            heads,
                                            sequence_length)] *
                                        key[activation_index(
                                            batch,
                                            other,
                                            head,
                                            column,
                                            sequence_length,
                                            heads,
                                            head_size)];
                        key_update += score_gradient[score_index(
                                          batch,
                                          head,
                                          other,
                                          token,
                                          heads,
                                          sequence_length)] *
                                      query[activation_index(
                                          batch,
                                          other,
                                          head,
                                          column,
                                          sequence_length,
                                          heads,
                                          head_size)];
                        value_update += probabilities[score_index(
                                            batch,
                                            head,
                                            other,
                                            token,
                                            heads,
                                            sequence_length)] *
                                        output_gradient[activation_index(
                                            batch,
                                            other,
                                            head,
                                            column,
                                            sequence_length,
                                            heads,
                                            head_size)];
                    }

                    const int index = activation_index(
                        batch,
                        token,
                        head,
                        column,
                        sequence_length,
                        heads,
                        head_size);
                    query_gradient[index] += query_update;
                    key_gradient[index] += key_update;
                    value_gradient[index] += value_update;
                }
            }
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int attention_benchmark(std::size_t batch, std::size_t sequence, std::size_t heads, std::size_t hidden) {
    require_multiple(hidden, 4, "head size");
    const std::size_t size = batch * sequence * heads * hidden;
    const std::size_t scores = batch * heads * sequence * sequence;
    const float scale = 1.0F / std::sqrt(static_cast<float>(hidden));
    auto query = random_values(size), key = random_values(size, 43);
    auto value = random_values(size, 44), gradient = random_values(size, 45);
    std::vector<float> output(size), probabilities(scores);
    std::vector<float> dq(size, 0.25F), dk(size, 0.25F), dv(size, 0.25F);
    DeviceBuffer<float> q(size), k(size), v(size), dy(size), y(size), p(scores);
    DeviceBuffer<float> gq(size), gk(size), gv(size);
    DeviceBuffer<float> workspace(std::max(
        hpc::dense_attention_forward_workspace_elements(batch, sequence, heads, hidden),
        hpc::dense_attention_backward_workspace_elements(batch, sequence, heads, hidden)));
    q.copy_from_host(query.data()); k.copy_from_host(key.data()); v.copy_from_host(value.data());
    dy.copy_from_host(gradient.data()); gq.copy_from_host(dq.data());
    gk.copy_from_host(dk.data()); gv.copy_from_host(dv.data());
    reference::dense_attention_forward_cpu(output.data(), probabilities.data(), query.data(),
        key.data(), value.data(), batch, sequence, heads, hidden, scale);
    reference::dense_attention_backward_cpu(dq.data(), dk.data(), dv.data(), gradient.data(),
        probabilities.data(), query.data(), key.data(), value.data(), batch, sequence, heads, hidden, scale);
    hpc::dense_attention_forward(y.data(), p.data(), q.data(), k.data(), v.data(), workspace.data(),
        batch, sequence, heads, hidden, scale);
    hpc::dense_attention_backward(gq.data(), gk.data(), gv.data(), dy.data(), p.data(),
        q.data(), k.data(), v.data(), workspace.data(), batch, sequence, heads, hidden, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "attention output", 5.0e-5F);
    valid &= validate_result(p, probabilities, "attention probabilities");
    valid &= validate_result(gq, dq, "attention query gradient", 5.0e-5F);
    valid &= validate_result(gk, dk, "attention key gradient", 5.0e-5F);
    valid &= validate_result(gv, dv, "attention value gradient", 5.0e-5F);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
