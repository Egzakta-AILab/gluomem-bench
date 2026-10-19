"""Resource admission contracts use authored bytes only; no operational approval."""
import copy
import hashlib
import json
from pathlib import Path
import pytest


def bound(path, material):
    path.write_bytes(json.dumps(material, sort_keys=True).encode())
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def authored_allowance(spec, folder, calls=9000, seconds=20000):
    """Test-only parent role. Never call on real evidence or an operational root."""
    caps = {k:spec['campaign_policy'][k] for k in ('max_calls','max_seconds')}
    context = dict(output_root=spec['output_root'], predecessor_binding=spec['predecessor_binding'],
                   observed_prior_accounting=spec['prior_accounting'], original_caps=caps,
                   prior_spend_uncertainty=spec['prior_spend_uncertainty'],
                   conservative_prior_allowance=dict(calls=calls,elapsed_seconds=seconds))
    proof = bound(folder/'authored-proof.json',dict(context,protocol='longmemeval-prior-resource-proof@v1',
        no_bypass_established=True,uncertainty_settled=True))
    proposal = bound(folder/'authored-proposal.json',dict(context,protocol='longmemeval-prior-resource-proposal@v1',
        proof=proof, allowance_includes_observed=True,
        actual_additional_spend=dict(calls=None,elapsed_seconds=None,tokens=None)))
    reserve = bound(folder/'authored-reserve-approval.json',dict(context,
        protocol='longmemeval-parent-resource-reserve-approval@v1',purpose='prior-resource-reserve-only',
        approved=True, approver_kind='parent',operational_execution_authorized=False,
        proposal=proposal,proof=proof))
    return dict(protocol='longmemeval-prior-resource-binding@v1',proposal=proposal,proof=proof,parent_reserve_approval=reserve)


def small_spec(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    return dict(protocol=d.CHAINED_PROTOCOL,output_root=str(tmp_path/'out'),
        predecessor_binding=dict(root=str(tmp_path/'old'),spec_sha256='1'*64,root_sha256='2'*64),
        campaign_policy=dict(max_calls=57344000,max_seconds=75603600),
        prior_accounting=dict(calls=8225,elapsed_seconds=14937.567625900003),
        prior_spend_uncertainty=dict(extra_generation_attempts=None,extra_tokens=None))


def test_prior_allowance_is_total_not_added_to_observed(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    spec = small_spec(tmp_path)
    spec['prior_resource_allowance'] = authored_allowance(spec,tmp_path)
    debit = d._validate_prior_resource_allowance(spec, require_settled=True)
    assert debit == dict(calls=9000,elapsed_seconds=20000)
    budget = CampaignBudget(max_calls=57344000,max_seconds=75603600,
        prior_calls=debit['calls'], prior_elapsed_seconds=debit['elapsed_seconds'],clock=lambda:0)
    assert budget.snapshot()['calls'] == 9000
    assert budget.deadline == 75583600
    assert spec['prior_accounting'] == dict(calls=8225,elapsed_seconds=14937.567625900003)
    assert spec['prior_spend_uncertainty']['extra_generation_attempts'] is None


from test_comprehensive_driver_setup import approved_setup


@pytest.mark.parametrize('stage', ['seed', 'scheduler'])
def test_builder_approval_and_actual_setup_debit_reserved_total_once(approved_setup, tmp_path, monkeypatch, stage):
    d, previous_spec, spec_path, approval_path = approved_setup
    binding = authored_allowance(previous_spec,tmp_path)
    spec = d.build_chained_continuation_execution_spec(
        predecessor_binding=previous_spec['predecessor_binding'],output_root=previous_spec['output_root'],
        semantic_delta_manifest=previous_spec['semantic_delta_manifest'],
        repaired_cohort=previous_spec['cohorts'][previous_spec['new_cohort']],
        launch_metadata=previous_spec['launch_metadata'],prior_resource_allowance=binding)
    assert spec['prior_accounting'] == previous_spec['prior_accounting']
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = json.loads(approval_path.read_bytes())
    approval['spec_sha256'] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    approval_path.write_bytes(json.dumps(approval).encode())
    with pytest.raises(ValueError,match='resource-bound'):
        d.verify_execution_spec(spec_path,approval_path)
    approval.update(prior_resource_allowance_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        parent_reserve_approval_sha256=binding['parent_reserve_approval']['sha256'])
    approval_path.write_bytes(json.dumps(approval).encode())
    assert d.verify_execution_spec(spec_path,approval_path) == spec
    original = OSError('authored setup stops before endpoints')
    def fail(*a, **kw): raise original
    if stage == 'seed': monkeypatch.setattr(d,'_seed_chain_cache_assets',fail)
    else:
        from gluomem_bench.bench import parallel_campaign as pc
        monkeypatch.setattr(pc,'run_jobs',fail)
    from gluomem_bench.bench.parallel_campaign import CampaignReceiptError
    with pytest.raises(CampaignReceiptError) as caught: d.execute_longmemeval(spec_path,approval_path)
    assert caught.value.__cause__ is original
    result = d.verify_campaign(spec['output_root'])
    assert result['accounting']['calls'] == 9000
    assert result['accounting']['prior_elapsed_seconds'] == 20000
    assert result['resource_accounting']['prior_debit_kind'] == 'reserved'
    assert result['resource_accounting']['observed_prior_accounting'] == previous_spec['prior_accounting']
    assert result['resource_accounting']['actual_additional_spend'] == dict(calls=None,elapsed_seconds=None,tokens=None)


def test_unknown_prior_uncertainty_cannot_execute_without_bound_settlement(approved_setup):
    d, spec, spec_path, approval_path = approved_setup
    del spec['prior_resource_allowance']
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = json.loads(approval_path.read_bytes())
    approval['spec_sha256'] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    approval_path.write_bytes(json.dumps(approval).encode())
    with pytest.raises(ValueError,match='uncertainty'):
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    assert not Path(spec['output_root']).exists()


@pytest.mark.parametrize('forgery', ['calls-under-observed','seconds-under-observed','cap','nonfinite','boolean','root','predecessor','observed','uncertainty','proof-unsettled','proof-bypass','reserve-execution','reserve-not-parent','proposal-actual-zero','proposal-includes','tampered-proof','missing-execution-hash','reserve-as-execution'])
def test_resource_proof_and_approval_forgeries_refused(tmp_path, forgery):
    from gluomem_bench.bench import comprehensive_driver as d
    spec = small_spec(tmp_path)
    binding = authored_allowance(spec,tmp_path)
    spec['prior_resource_allowance'] = binding
    if forgery in ('missing-execution-hash','reserve-as-execution'):
        approval = {} if forgery == 'missing-execution-hash' else json.loads(Path(binding['parent_reserve_approval']['path']).read_bytes())
        with pytest.raises(ValueError): d._validate_prior_resource_allowance(spec,execution_approval=approval)
        return
    if forgery == 'tampered-proof':
        Path(binding['proof']['path']).write_bytes(b'{}')
    else:
        name = 'proof' if forgery.startswith('proof-') else ('parent_reserve_approval' if forgery.startswith('reserve-') else 'proposal')
        item = json.loads(Path(binding[name]['path']).read_bytes())
        if forgery == 'calls-under-observed': item['conservative_prior_allowance']['calls'] = 8224
        if forgery == 'seconds-under-observed': item['conservative_prior_allowance']['elapsed_seconds'] = 14937
        if forgery == 'cap': item['conservative_prior_allowance']['calls'] = 57344001
        if forgery == 'nonfinite': item['conservative_prior_allowance']['elapsed_seconds'] = float('nan')
        if forgery == 'boolean': item['conservative_prior_allowance']['calls'] = True
        if forgery == 'root': item['output_root'] += '-forged'
        if forgery == 'predecessor': item['predecessor_binding']['spec_sha256'] = '0'*64
        if forgery == 'observed': item['observed_prior_accounting']['calls'] = 0
        if forgery == 'uncertainty': item['prior_spend_uncertainty']['extra_generation_attempts'] = 0
        if forgery == 'proof-unsettled': item['uncertainty_settled'] = False
        if forgery == 'proof-bypass': item['no_bypass_established'] = False
        if forgery == 'reserve-execution': item['operational_execution_authorized'] = True
        if forgery == 'reserve-not-parent': item['approver_kind'] = 'agent'
        if forgery == 'proposal-actual-zero': item['actual_additional_spend']['calls'] = 0
        if forgery == 'proposal-includes': item['allowance_includes_observed'] = False
        binding[name] = bound(Path(binding[name]['path']),item)
    with pytest.raises(ValueError): d._validate_prior_resource_allowance(spec,require_settled=True)


def test_legacy_unbound_validation_keeps_observed_shape_and_v3_budget(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    spec = small_spec(tmp_path)
    assert d._validate_prior_resource_allowance(spec) == spec['prior_accounting']
    spec['protocol'] = d.CONTINUATION_PROTOCOL
    assert d._validate_prior_resource_allowance(spec) == spec['prior_accounting']
    assert d._resource_accounting(spec) == {}
