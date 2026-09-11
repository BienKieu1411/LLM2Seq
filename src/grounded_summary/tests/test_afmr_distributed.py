from grounded_summary.data.sampling import DistributedBatchSampler, DistributedEvalSampler


def test_distributed_batches_cover_each_example_once_with_last_placeholder():
    lengths = [5, 2, 7, 3, 9]
    samplers = [
        DistributedBatchSampler(lengths, batch_size=2, rank=rank, world_size=2, shuffle=False, bucket=False)
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert len(batches[0]) == len(batches[1]) == 2
    observed = [index for worker in batches for batch in worker for index in batch if index >= 0]
    assert sorted(observed) == list(range(len(lengths)))
    assert any(-1 in batch for worker in batches for batch in worker)


def test_distributed_eval_sampler_is_non_overlapping_and_ordered():
    samplers = [DistributedEvalSampler(7, rank, 3) for rank in range(3)]
    shards = [list(sampler) for sampler in samplers]
    assert shards == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(index for shard in shards for index in shard) == list(range(7))


def test_distributed_eval_sampler_balances_length_sorted_global_batches():
    lengths = [1, 100, 2, 101]
    shards = [list(DistributedEvalSampler(len(lengths), rank, 2, lengths=lengths, batch_size=2)) for rank in range(2)]
    assert shards == [[0, 1], [2, 3]]
