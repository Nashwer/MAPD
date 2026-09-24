from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from mapd.config import MASConfig
from mapd.data.schema import QASample
from mapd.mas.orchestrator import Orchestrator
from mapd.mas.protocolizer import Protocolizer
from mapd.mas.provider import TeacherClient
from mapd.mas.repair import Repair
from mapd.mas.schema import ExplorationLog, SearchFinding, SearchRecord, SubTask, SynthesisArtifact
from mapd.mas.searcher import Searcher
from mapd.protocol.schema import TaskType
from mapd.protocol.validator import validate_protocol
from mapd.retrieval.base import Retriever
from mapd.reward.exact_match import exact_match


class MASPipeline:
    """Coordinates paper Section 3.2 while role logic stays in dedicated modules."""

    def __init__(self, teacher: TeacherClient, retriever: Retriever, config: MASConfig, top_k: int = 3):
        self.teacher = teacher
        self.config = config
        self.orchestrator = Orchestrator(teacher, config.max_subquestions)
        self.searcher = Searcher(
            teacher,
            retriever,
            max_queries=config.max_queries_per_searcher,
            top_k=top_k,
        )
        self.repair = Repair(teacher, config.max_subquestions)
        self.protocolizer = Protocolizer(teacher)

    def synthesize(self, sample: QASample) -> SynthesisArtifact:
        task_type = TaskType.OTHERS
        subtasks: list[SubTask] = []
        searches: list[SearchRecord] = []
        findings: list[SearchFinding] = []
        repairs = []
        candidate = None
        success = False
        search_rounds = 0

        for round_index in range(1, self.config.max_rounds + 1):
            proposed_type, proposed = self.orchestrator.plan(sample.question, round_index, findings)
            if proposed_type is not None:
                task_type = proposed_type
            if not proposed and round_index == 1:
                proposed = [SubTask(id="round-1-s1", objective=sample.question, round_index=1)]
            if not proposed:
                break
            new_searches, new_findings = self._execute_dependency_graph(proposed, findings, sample.answers)
            subtasks.extend(proposed)
            searches.extend(new_searches)
            findings.extend(new_findings)
            search_rounds += 1
            candidate = self._answer(sample.question, searches, findings)
            success = exact_match(candidate, sample.answers)
            if success:
                break

        repair_rounds = 0
        while not success and repair_rounds < self.config.max_repair_rounds:
            repair_rounds += 1
            repair_record, proposed = self.repair.diagnose(
                sample.question,
                sample.answers,
                candidate,
                findings,
                repair_rounds,
            )
            repairs.append(repair_record)
            if not proposed:
                break
            new_searches, new_findings = self._execute_dependency_graph(proposed, findings, sample.answers)
            subtasks.extend(proposed)
            searches.extend(new_searches)
            findings.extend(new_findings)
            candidate = self._answer(sample.question, searches, findings)
            success = exact_match(candidate, sample.answers)

        exploration = ExplorationLog(
            example_id=sample.id,
            question=sample.question,
            task_type=task_type,
            subtasks=subtasks,
            searches=searches,
            findings=findings,
            repairs=repairs,
            candidate_answer=candidate,
            success=success,
            search_rounds=search_rounds,
            repair_rounds=repair_rounds,
        )
        protocol, raw_protocol, schema_errors = self.protocolizer.generate(exploration)
        quality = validate_protocol(
            sample, exploration, protocol, schema_errors=schema_errors
        )
        return SynthesisArtifact(
            sample=sample,
            exploration=exploration,
            protocol=protocol,
            raw_protocol=raw_protocol,
            quality=quality,
            metadata={
                "teacher_model": self.teacher.model,
                "protocol_variant": "succeeded" if success else "evidence",
            },
        )

    def _execute_dependency_graph(
        self,
        subtasks: list[SubTask],
        prior_findings: list[SearchFinding],
        forbidden_answers: list[str],
    ) -> tuple[list[SearchRecord], list[SearchFinding]]:
        pending = {item.id: item for item in subtasks}
        completed = {item.subtask_id for item in prior_findings}
        searches, findings = [], []
        while pending:
            ready = [item for item in pending.values() if set(item.depends_on) <= completed]
            if not ready:
                raise ValueError(f"cyclic or unresolved subtask dependencies: {', '.join(sorted(pending))}")
            available = prior_findings + findings
            with ThreadPoolExecutor(max_workers=min(len(ready), self.config.max_subquestions)) as executor:
                results = list(
                    executor.map(
                        lambda task: self.searcher.run(task, available, forbidden_answers),
                        ready,
                    )
                )
            for task, (task_searches, finding) in zip(ready, results):
                searches.extend(task_searches)
                findings.append(finding)
                completed.add(task.id)
                pending.pop(task.id)
        return searches, findings

    def _answer(
        self, question: str, searches: list[SearchRecord], findings: list[SearchFinding]
    ) -> str | None:
        result = self.teacher.generate_json(
            "answerer",
            {
                "question": question,
                "findings": [item.model_dump(mode="json") for item in findings],
                "passages": [
                    passage.model_dump(mode="json")
                    for search in searches
                    for passage in search.passages
                ],
            },
        )
        answer = result.get("answer")
        return str(answer).strip() if answer is not None else None
