"""Authored qualification permit and proxy contracts (not endpoint results)."""
import json
import pytest


def test_qualification_permit_is_exact_fixed_fixture_and_hard_caps():
    from gluomem_bench.bench.native_backend_qualification import validate_permit
    permit = {'generation_admission_cap':192,'wall_seconds_cap':1200,'retries':0,'same_approved_endpoints_only':True,'all_spend_retained':True}
    assert validate_permit(permit) == (192,1200)
    for key,value in [('generation_admission_cap',193),('wall_seconds_cap',1201),('retries',1)]:
        with pytest.raises(ValueError):
            validate_permit(dict(permit,**{key:value}))


def test_encoder_proxy_retains_actual_role_text_batch_and_exact_replay(tmp_path):
    import httpx
    from gluomem_bench.bench.native_backend_qualification import EncoderProxy
    request = {'model':'authored','input':[' exact\n','second'],'input_type':'query'}
    response = {'model':'authored','data':[{'index':0,'embedding':[1.,0.]},{'index':1,'embedding':[0.,1.]}]}
    def upstream(req):
        assert json.loads(req.content) == request
        assert req.url.path == '/v1/embeddings'
        return httpx.Response(200,json=response)
    with EncoderProxy('http://approved.invalid/v1',receipt=tmp_path/'encoder.jsonl',
                      timeout=lambda:10,transport=httpx.MockTransport(upstream)) as proxy:
        result = httpx.post(proxy.base_url+'/embeddings',json=request,trust_env=False)
        assert result.json() == response
        proof = proxy.replay()
    assert proof['exact_same_shape_matches'] == 1
    rows = [json.loads(line) for line in (tmp_path/'encoder.jsonl').read_bytes().splitlines()]
    assert rows[0]['request'] == request
    assert rows[0]['response'] == response


def test_encoder_proxy_first_failure_latches_no_native_retry_dispatch(tmp_path):
    import httpx
    from gluomem_bench.bench.native_backend_qualification import EncoderProxy
    seen=[]
    def upstream(request):
        seen.append(request)
        return httpx.Response(400,content=b'PRIVATE_ENCODER_ERROR')
    with EncoderProxy('http://approved.invalid/v1',receipt=tmp_path/'encoder.jsonl',
                      timeout=lambda:10,transport=httpx.MockTransport(upstream)) as proxy:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                proxy.forward({'model':'authored','input':['test'],'input_type':'document'})
    assert len(seen)==1, 'native SDK retry must not repeat real failed encoder dispatch'
    assert b'PRIVATE_ENCODER_ERROR' not in (tmp_path/'encoder.jsonl').read_bytes()
