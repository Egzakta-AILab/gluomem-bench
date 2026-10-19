"""Actual driver failure serialization, with authored secrets and denied sockets."""
import io,json,urllib.error
from pathlib import Path
import pytest


def error():
    return urllib.error.HTTPError('https://fixture.invalid/private',400,'AUTHORED-REASON-SECRET',
        {'Authorization':'AUTHORED-HEADER-SECRET'},io.BytesIO(b'AUTHORED-BODY-SECRET'))


@pytest.mark.parametrize('boundary',['abort','publication','artifact-cleanup'])
def test_driver_failure_receipts_never_format_untrusted_exception(tmp_path,monkeypatch,boundary):
    from gluomem_bench.bench import comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget,CampaignReceiptError
    original=error(); root=tmp_path/'out';root.mkdir()
    (root/'execution-spec.json').write_bytes(b'{}')
    budget=CampaignBudget(max_calls=100,max_seconds=100,prior_calls=7,prior_elapsed_seconds=3.5)
    spec=dict(protocol=d.CHAINED_PROTOCOL,jobs=[],files=[],reuse_receipt={},cognee_unrun=[],cohorts={},prior_spend_uncertainty={})
    if boundary=='artifact-cleanup':
        spec.update(new_cohort='fresh',jobs=[dict(job_id='j',history_id='q',disposition='dispatch',output=str(root/'jobs/j'))])
        def fail_artifact(*a):raise error()
        monkeypatch.setattr(d,'_job_artifacts',fail_artifact)
        unlink=Path.unlink
        def fail_unlink(path,*a,**kw):
            if path==root/'root-status.json':raise error()
            return unlink(path,*a,**kw)
        monkeypatch.setattr(Path,'unlink',fail_unlink)
    with pytest.raises(CampaignReceiptError) as caught:
        if boundary=='publication':
            def fail_verify(*a,**kw):raise original
            monkeypatch.setattr(d,'verify_campaign',fail_verify)
            d._publish_continuation(root,dict(jobs=[],complete=False,execution_complete=False),budget)
        else:d._abort_continuation(root,spec,budget,original)
    assert caught.value.__cause__ is original
    material=json.dumps(caught.value.receipt)
    for path in root.glob('*.json'):
        if path.name!='execution-spec.json':material+=path.read_text()
    assert 'AUTHORED-' not in material
    assert caught.value.receipt['receipt_failure']=={'type':'HTTPError','message':'encoder_http_error'}
