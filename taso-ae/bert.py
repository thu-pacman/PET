import taso as ts
import sys

seq_length = 512
hidden_dims = 768
batch_size = int(sys.argv[1])


def attention(graph, input, heads):
    embed = input.dim(1)  # embedding len
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(embed, embed)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(batch_size, 512, 12, 64))
    k = graph.reshape(k, shape=(batch_size, 512, 12, 64))
    v = graph.reshape(v, shape=(batch_size, 512, 12, 64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(0, 2, 1, 3), shuffle=True)
    k = graph.transpose(k, perm=(0, 2, 3, 1), shuffle=True)
    v = graph.transpose(v, perm=(0, 2, 1, 3), shuffle=True)
    # perform matrix multiplications

    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output, perm=(0, 2, 1, 3), shuffle=True)
    output = graph.reshape(output, shape=(batch_size, 512, 768))

    # a final linear layer
    linear = graph.new_weight(dims=(batch_size, embed, embed))
    linear2 = graph.new_weight(dims=(batch_size, embed, embed))
    output = graph.matmul(output, linear)
    output = graph.relu(graph.reshape(output, shape=(batch_size * 512, 768)))

    output = graph.reshape(output, shape=(batch_size, 512, 768))
    output = graph.matmul(output, linear2)
    output = graph.relu(graph.reshape(output, shape=(batch_size * 512, 768)))

    output = graph.add(output, input)
    output = graph.reshape(output, shape=(batch_size * 512, 768))
    # output = graph.new_weight(dims=(seq_length, embed))
    return output


graph = ts.new_graph()
input = graph.new_input(dims=(batch_size * seq_length, hidden_dims))
input = graph.relu(input)
t = input
for i in range(12):
    t = attention(graph, t, 16)

new_graph = ts.optimize(graph, alpha=1.0, budget=100)

print(graph.run_time())
print(new_graph.run_time())