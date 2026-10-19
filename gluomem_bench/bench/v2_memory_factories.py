"""Concrete evidence-only native V2 factories. No native answers or raw fallback."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gluomem_bench.bench.v2_driver import (
    ResourceNotAdmitted, digest, document_events, file_hash, remaining_calls, write_json,
)


def verify_closed_native(out, *, method, documents, source_proof, profile):
    """Offline closed-store verification against trusted serialized source docs."""
    out = Path(out)
    if digest(documents) != source_proof['documents_sha256']:
        raise ValueError('trusted source digest differs')
    for document in documents:
        import hashlib
        if hashlib.sha256(document['text'].encode('utf-8')).hexdigest() != document['sha256']:
            raise ValueError('source document bytes differ')
    root = out / ('native-mem0' if method == 'mem0' else 'native-cognee')
    name = 'native-closed.json' if method == 'mem0' else 'native-ledger.json'
    ledger = json.loads((root / name).read_bytes())
    pins = ledger['files'] if method == 'mem0' else ledger['artifact_sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.name != name}
    if set(pins) != actual:
        raise ValueError('native closed artifact inventory differs')
    for relative, pin in pins.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != pin:
            raise ValueError('native closed artifact hash differs')
    if method == 'mem0':
        from gluomem_bench.bench.mab_campaign import _verify_mem0_coverage
        coverage = json.loads((root / 'coverage.json').read_bytes())
        history = SimpleNamespace(id=coverage['history_id'], context_sha256=source_proof['documents_sha256'])
        calls = list(map(json.loads, (out / 'calls.jsonl').read_bytes().splitlines()))
        _verify_mem0_coverage(root, history, document_events(documents), profile, calls)
    else:
        from gluomem_bench.bench.comparison_backends import Evidence
        from gluomem_bench.bench.cognee_campaign import verify_native_coverage, verify_generation_evidence
        expected = tuple(Evidence(d['id'], d['text'], {'offset': d['offset']}) for d in documents)
        if ledger['status'] != 'closed_verified' or not ledger['native_worker_closed'] or not ledger['proxy_closed']:
            raise ValueError('native worker not closed verified')
        verify_native_coverage(root / 'native', dataset=ledger['dataset'], expected=expected, quiescent=True)
        verify_generation_evidence(root / 'generation.jsonl', profile=profile,
                                   expected_calls=ledger['generation_calls'], starting_calls=0)
    return {'verified': True, 'method': method, 'documents_sha256': source_proof['documents_sha256']}


def _admit(documents, source_proof, budget, qa_reserve):
    # Mem0 has exactly one extraction per source string; Cognee at least one
    # processing request per document. This is a lower bound, NOT measured fit.
    if len(documents) + qa_reserve > remaining_calls(budget):
        raise ResourceNotAdmitted('native ingestion lower bound plus reader/judge reserve exceeds cap')
    if not source_proof.get('complete') or source_proof['documents_sha256'] != digest(documents):
        raise ValueError('complete trusted source proof required')
    budget.check()


def cognee_factory(*, python_executable, chunk_size=512, transport=None, **kwargs):
    _admit(kwargs['documents'], kwargs['source_proof'], kwargs['budget'], kwargs['qa_reserve'])
    return CogneeDocumentMemory(python_executable=python_executable,
                               chunk_size=chunk_size, transport=transport, **kwargs)


class CogneeDocumentMemory:
    def __init__(self, *, cfg, group, documents, source_proof, out, client, budget,
                 embedder, qa_reserve, python_executable, chunk_size, transport):
        from gluomem_bench.bench.comparison_backends import CogneeSettings, Evidence
        from gluomem_bench.bench.cognee_retriever import CogneeRetrieverFactory
        if embedder is None:
            raise ValueError('explicit qualified shared encoder required')
        self.documents, self.proof = documents, source_proof
        self.client, self.budget = client, budget
        self.top_k = cfg.benchmark.answer.context_k
        self.out = Path(out) / 'native-cognee'
        settings = CogneeSettings(
            python_executable=python_executable, root=self.out / 'native', dataset=group.id,
            llm_url=client.profile.base_url, llm_model=client.profile.model, llm_api_key='local',
            embedding_url=embedder.base_url,
            embedding_model=embedder.recipe.model, embedding_dimensions=embedder.dim,
            embedding_api_key='local', max_completion_tokens=client.profile.output_budget('cognify'),
            chunk_size=chunk_size, timeout_seconds=min(client.profile.timeout_seconds, budget.remaining()))
        self.factory = CogneeRetrieverFactory(
            settings=settings, profile=client.profile, outer_budget=budget, out=self.out,
            transport=transport,
            documents=tuple(Evidence(d['id'], d['text'], {'offset': d['offset']}) for d in documents))
        # NEVER call the LME answer adapter. Its native graph backend provides
        # evidence to the V2 reader, including the original independent query image.
        self.retriever = self.factory(cfg, None, None)
        self.closed = False

    def verify_source_coverage(self, *, documents, source_proof):
        from gluomem_bench.bench.cognee_retriever import verify_graph_receipt
        if documents != self.documents or source_proof != self.proof:
            raise ValueError('trusted source changed')
        r = self.retriever
        observed = verify_graph_receipt(self.factory.settings, r.documents, r.bridge, r.pins,
                                       quiescent=self.closed)
        return {**observed, 'complete': True,
                'ordered_document_hashes': [(d['id'], d['sha256']) for d in documents],
                'documents_sha256': source_proof['documents_sha256'],
                'shared_budget_identity': id(self.budget), 'shared_client_identity': id(self.client),
                'persistent_readback': True}

    def query(self, text, *, query_image=None):
        from dataclasses import asdict
        from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION
        r = self.retriever
        self.budget.check()
        r.bridge.check()
        r._bound_timeout()
        hits = r.runner.run(r.backend.search(text, top_k=self.top_k))
        for hit in hits:
            p = hit.provenance
            if (p.get('condition') != GRAPH_CONDITION or
                p.get('dataset_name') != self.factory.settings.dataset or
                not p.get('dataset_id') or p.get('completion') is not None or
                not any(ref.get('kind') == 'graph_edge' for ref in p['native_evidence'])):
                raise ValueError('native graph evidence lacks scoped lineage')
        self.verify_source_coverage(documents=self.documents, source_proof=self.proof)
        r.contexts.append({'query': text, 'hits': [asdict(hit) for hit in hits],
                           'native_final_completion_used': False})
        write_json(self.out / 'graph-contexts.json', r.contexts)
        self.budget.check()
        return [{'type': 'text', 'value': hit.text} for hit in hits]

    def close(self):
        if not self.closed:
            self.factory.close()
            self.closed = True


def mem0_factory(**kwargs):
    _admit(kwargs['documents'], kwargs['source_proof'], kwargs['budget'], kwargs['qa_reserve'])
    return Mem0DocumentMemory(**kwargs)


class Mem0DocumentMemory:
    def __init__(self, *, cfg, group, documents, source_proof, out, client, budget,
                 embedder, qa_reserve):
        from gluomem_bench.bench.mab_campaign import _ingest_mem0_documents
        self.documents, self.proof = documents, source_proof
        self.job_out = Path(out)
        self.out, self.client, self.budget = self.job_out / 'native-mem0', client, budget
        self.out.mkdir(exist_ok=False)
        self.history = SimpleNamespace(id=group.id, context_sha256=source_proof['documents_sha256'])
        self.events = document_events(documents)
        self.top_k = cfg.benchmark.answer.context_k
        (self.out / 'stores').mkdir(exist_ok=False)
        self.memory, self.native_coverage = _ingest_mem0_documents(
            cfg, history=self.history, events=self.events, out=self.out,
            client=client, budget=budget, embedder=embedder)
        write_json(self.out / 'coverage.json', self.native_coverage)
        self.contexts = []
        self.closed = False

    def verify_source_coverage(self, *, documents, source_proof):
        from gluomem_bench.bench.mab_campaign import _verify_mem0_coverage
        calls = [json.loads(line) for line in (self.job_out / 'calls.jsonl').read_bytes().splitlines()]
        records, snapshot = _verify_mem0_coverage(self.out, self.history, self.events, self.client.profile, calls)
        if documents != self.documents or source_proof != self.proof:
            raise ValueError('trusted source changed')
        return {**self.native_coverage, 'complete': True,
                'ordered_document_hashes': [(d['id'], d['sha256']) for d in documents],
                'documents_sha256': source_proof['documents_sha256'],
                'shared_budget_identity': id(self.budget), 'shared_client_identity': id(self.client),
                'persistent_readback': True}

    def query(self, text, *, query_image=None):
        self.budget.check()
        hits = self.memory.search(text, filters={'user_id': self.history.id}, top_k=self.top_k,
                                  threshold=0.1, rerank=False)['results']
        if any(hit.get('user_id') != self.history.id for hit in hits):
            raise ValueError('native Mem0 search escaped allowed source scope')
        self.memory._benchmark_diagnostics.check()
        self.contexts.append({'query': text, 'filters': {'user_id': self.history.id},
                              'top_k': self.top_k, 'threshold': 0.1, 'rerank': False, 'hits': hits})
        write_json(self.out / 'query-contexts.json', self.contexts)
        self.budget.check()
        return [{'type': 'text', 'value': hit['memory']} for hit in hits]

    def close(self):
        if self.closed:
            return
        from gluomem_bench.bench.mem0 import close_mem0_memory
        close_mem0_memory(self.memory)
        self.closed = True
        self.verify_source_coverage(documents=self.documents, source_proof=self.proof)
        write_json(self.out / 'native-closed.json', {
            'closed': True, 'files': {p.relative_to(self.out).as_posix(): file_hash(p)
            for p in sorted(self.out.rglob('*')) if p.is_file() and p.name != 'native-closed.json'}})
