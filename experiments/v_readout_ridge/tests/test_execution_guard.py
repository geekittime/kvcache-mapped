import json
from guard_execution import is_resource_failure

def test_numerical_or_input_failure_is_not_auto_retried(tmp_path):
    p=tmp_path/'failure.json'
    for message in ['AssertionError: prompt IDs differ','RuntimeError: nonfinite mapper',
                    'RuntimeError: Worker failures: [failed]']:
        p.write_text(json.dumps(dict(traceback=message)))
        assert not is_resource_failure(p)

def test_explicit_cuda_oom_allows_resource_retry(tmp_path):
    p=tmp_path/'failure.json'
    p.write_text(json.dumps(dict(traceback='torch.OutOfMemoryError: CUDA out of memory.')))
    assert is_resource_failure(p)
