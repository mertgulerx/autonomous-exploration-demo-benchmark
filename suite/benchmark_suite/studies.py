# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Study and run-record reads for the Results page (RESULTS_OVERHAUL §8 and R4b; RESULTS_RESEARCH RD-37).

``runs_page`` answers ``results.query`` table ``runs``: one row per attempt with a current-version run record,
the record with its read-time fields (finality, replacement, report class, supersession). ``studies_page``
lists the studies (``studies.list``) and ``study`` describes one (``study.get``): its campaigns, counts, the
values the page offers as selectors and the warnings the export also records. ``runs_page``, ``studies_page``,
``attempt_record`` and ``provenance_document`` run on the engine thread, bounded by a page or one row;
``study`` reads every record of a study, so the engine runs it through ``read_study`` off its loop. Nothing here
reads attempt files.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import closing
import json
from pathlib import Path
import math
import sqlite3
from typing import Any, cast

from .contracts import UNASSIGNED, ContractError, canonical
from .provenance import REGISTRY_FILES
from .run_record import (
    FIELDS,
    INDEX_COLUMNS,
    MILESTONES,
    RUN_RECORD_VERSION,
    read_time_fields,
    report_class,
)
from .store import (
    ACTIVE_STATES,
    TUNING_CAMPAIGN,
    replaced_campaigns,
    restart_sources,
    study_campaign_rows,
)

# results.query table runs: filters on the index columns, finality and the record values study.get offers
# as selectors (debug_streams, a list, and run_record_version, always current here, are not filters).
RECORD_FILTERS = ("dataset_schema", "coverage_metric_version", "timing_rule")
RUN_FILTERS = frozenset({*INDEX_COLUMNS, *RECORD_FILTERS, "is_final"})
# Sortable: every scalar field of the record, and finality.
RUN_SORT_KEYS = frozenset({field.name for field in FIELDS if field.kind != "list"} | {"is_final"})
RUN_ORDER = "c.created_ns,c.id,e.position,r.attempt_number"
# Time-to-event fields and their censor times: a censored run sorts as later than every observed one
CENSORED_SORT = {
    "activation_to_movement_s": "activation_to_movement_censor_s",
    "completion_s": "completion_censor_s",
    **{f"t{m}_s": f"t{m}_censor_s" for m in MILESTONES},
}
# Values whose mixing among full final records the export refuses (``refused_mixes``).
REFUSED_MIXES = (
    "run_record_version",
    "dataset_schema",
    "coverage_metric_version",
    "mapping_condition",
)
# A full record with a stratum (comparison key) and no mapping condition ran before I10.
BEFORE_MAPPING_CONDITION = "none (before mapping-known-pose-v1)"
HOMOGENEOUS = ("timing_rule", "debug_streams", "diagnostics_enabled", "provenance_id")
EXPORTS_LISTED = 20


def _page(offset: Any, limit: Any) -> tuple[int, int]:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 250:
        raise ContractError("page", "Use a nonnegative offset and page size 1–250")
    return offset, limit


def _superseded(database: sqlite3.Connection) -> dict[str, str]:
    """Campaigns a restarted copy replaces, and that copy."""
    return {source: copy for copy, source in restart_sources(database).items()}


def held(database: sqlite3.Connection, alias: str = "r") -> tuple[str, list[str]]:
    """SQL condition (and its arguments) on ``alias.campaign_id`` for the campaigns an export holds: no tuning
    runs and no campaign a restarted copy replaces (``exports.snapshot`` applies the same rule)."""
    replaced = sorted(replaced_campaigns(database))
    return (
        f"{alias}.campaign_id NOT IN (SELECT id FROM campaigns WHERE {TUNING_CAMPAIGN}) "
        f"AND {alias}.campaign_id NOT IN ({','.join('?' * len(replaced))})",
        replaced,
    )


def last_repetition(database: sqlite3.Connection, study_id: str) -> int | None:
    """The highest repetition a study has used: its campaigns' experiments, or ``study_repetitions`` when
    campaigns were deleted (a new campaign continues after it; RD-30). None before the first campaign."""
    identities = [row[0] for row in study_campaign_rows(database, study_id)]
    used = database.execute(
        f"SELECT MAX(repetition) FROM experiments WHERE campaign_id IN ({','.join('?' * len(identities))})",
        identities,
    ).fetchone()[0]
    kept = database.execute(
        "SELECT last_repetition FROM study_repetitions WHERE study_id=?", (study_id,)
    ).fetchone()
    values = [value for value in (used, kept[0] if kept else None) if value is not None]
    return max(values) if values else None


def with_read_time(
    record: dict[str, Any],
    final_attempt_id: str | None,
    experiment_state: str | None,
    superseded: Mapping[str, str],
    last_attempt_number: int | None = None,
) -> dict[str, Any]:
    """A stored record with its read-time fields (``run_record.read_time_fields``)."""
    experiment = dict(
        final_attempt_id=final_attempt_id,
        state=experiment_state,
        last_attempt_number=last_attempt_number,
    )
    return dict(
        record, **read_time_fields(record, experiment, superseded.get(record["campaign_id"]))
    )


def _run_source(database: sqlite3.Connection, filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """The FROM … WHERE of a run read and its arguments, from validated ``filters``."""
    if set(filters) - RUN_FILTERS:
        raise ContractError("filters", "Unsupported table or filter")
    terms: list[str] = ["r.run_record_version=?"]
    arguments: list[Any] = [RUN_RECORD_VERSION]
    for key, value in filters.items():
        if key == "is_final":
            if type(value) is not bool:
                raise ContractError("filters.is_final", "Expected boolean")
            terms.append("(e.final_attempt_id IS r.attempt_id)=?")
            arguments.append(int(value))
            continue
        if value is not None and (
            type(value) not in (str, int, float)
            or type(value) is float
            and not math.isfinite(value)
        ):
            raise ContractError("filters." + key, "Use text, a finite number or null")
        column = f"r.{key}" if key in INDEX_COLUMNS else f"json_extract(r.record,'$.{key}')"
        if value is None:
            terms.append(column + " IS NULL")
        else:
            terms.append(column + "=?")
            arguments.append(value)
    # Tuning runs never appear in lists, and a study's runs are the ones its export holds (no campaign a
    # restarted copy replaces); a filter by one campaign still reaches that campaign's runs.
    if "campaign_id" not in filters and "study_id" in filters:
        condition, replaced = held(database)
        terms.append(condition)
        arguments.extend(replaced)
    elif "campaign_id" not in filters:
        terms.append(f"r.campaign_id NOT IN (SELECT id FROM campaigns WHERE {TUNING_CAMPAIGN})")
    source = (
        "run_records r JOIN experiments e ON e.id=r.experiment_id "
        "JOIN campaigns c ON c.id=r.campaign_id WHERE " + " AND ".join(terms)
    )
    return source, arguments


def runs_page(
    database: sqlite3.Connection,
    filters: dict[str, Any],
    offset: Any = 0,
    limit: Any = 100,
    sort_by: Any = None,
    sort_descending: Any = False,
) -> dict[str, Any]:
    """One page of run records (``results.query`` table ``runs``), in plan order unless sorted."""
    offset, limit = _page(offset, limit)
    if type(sort_descending) is not bool:
        raise ContractError("sort_descending", "Expected boolean")
    source, arguments = _run_source(database, filters)
    if sort_by is None:
        if sort_descending:
            raise ContractError("sort_by", "Choose a column before setting the sort direction")
        order = RUN_ORDER
    elif type(sort_by) is not str or sort_by not in RUN_SORT_KEYS:
        raise ContractError("sort_by", "Choose a run record field")
    else:
        column = (
            "(e.final_attempt_id IS r.attempt_id)"
            if sort_by == "is_final"
            else f"r.{sort_by}"
            if sort_by in INDEX_COLUMNS
            else f"json_extract(r.record,'$.{sort_by}')"
        )
        direction = "DESC" if sort_descending else "ASC"
        if sort_by in CENSORED_SORT:
            # observed times, then censored ones by their censor time (reversed when descending), then none
            censor = f"json_extract(r.record,'$.{CENSORED_SORT[sort_by]}')"
            first, second = (1, 0) if sort_descending else (0, 1)
            rank = (
                f"CASE WHEN {column} IS NOT NULL THEN {first} "
                f"WHEN {censor} IS NOT NULL THEN {second} ELSE 2 END"
            )
            order = f"{rank},{column} {direction},{censor} {direction},{RUN_ORDER}"
        else:
            order = f"{column} IS NULL,{column} {direction},{RUN_ORDER}"  # unavailable values last
    rows = database.execute(
        "SELECT r.record,e.final_attempt_id,e.state,"
        "(SELECT COUNT(*) FROM attempts n WHERE n.experiment_id=r.experiment_id),"
        "(SELECT MAX(n.number) FROM attempts n WHERE n.experiment_id=r.experiment_id) "
        f"FROM {source} ORDER BY {order} LIMIT ? OFFSET ?",
        (*arguments, limit, offset),
    ).fetchall()
    total = database.execute(f"SELECT COUNT(*) FROM {source}", arguments).fetchone()[0]
    superseded = _superseded(database)
    return dict(
        rows=[
            dict(
                with_read_time(json.loads(row[0]), row[1], row[2], superseded, row[4]),
                experiment_attempts=row[3],
            )
            for row in rows
        ],
        total=total,
        offset=offset,
        next_offset=offset + len(rows) if offset + len(rows) < total else None,
    )


def run_counts(database: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, int]:
    """Counts over every run a filter selects (``results.query`` runs with ``include: ["counts"]``): final
    runs by report class (``reported-outcomes-v2``, as each row's read-time field) and the attempts that are
    not final as ``discarded`` (a discard) or ``replaced`` (a timing rejection or an interruption, replaced or
    waiting for its replacement). The Results footer reads them."""
    source, arguments = _run_source(database, filters)
    counts: Counter[str] = Counter()
    for final, outcome, started, exited, state in database.execute(
        "SELECT (e.final_attempt_id IS r.attempt_id),r.outcome,"
        "json_extract(r.record,'$.insight_started'),json_extract(r.record,'$.insight_exited_before_stop'),"
        f"e.state FROM {source}",
        arguments,
    ):
        # json_extract gives SQLite integers for booleans; None stays None
        started = None if started is None else bool(started)
        exited = None if exited is None else bool(exited)
        kind = report_class(bool(final), outcome, started, exited, state)
        if kind == "not_final":
            kind = "discarded" if outcome == "manual_discard" else "replaced"
        counts[kind] += 1
    return dict(sorted(counts.items()))


def attempt_record(database: sqlite3.Connection, attempt_id: str) -> dict[str, Any] | None:
    """An attempt's current-version run record with its read-time fields; None until it is built."""
    row = database.execute(
        "SELECT r.record,e.final_attempt_id,e.state,"
        "(SELECT MAX(n.number) FROM attempts n WHERE n.experiment_id=r.experiment_id) FROM run_records r "
        "JOIN experiments e ON e.id=r.experiment_id WHERE r.attempt_id=? AND r.run_record_version=?",
        (attempt_id, RUN_RECORD_VERSION),
    ).fetchone()
    if row is None:
        return None
    return with_read_time(json.loads(row[0]), row[1], row[2], _superseded(database), row[3])


def provenance_document(
    database: sqlite3.Connection, identity: str | None
) -> dict[str, Any] | None:
    """A stored provenance document (``provenance.py``), or None."""
    if identity is None:
        return None
    row = database.execute("SELECT document FROM provenance WHERE id=?", (identity,)).fetchone()
    return cast(dict[str, Any], json.loads(row[0])) if row else None


def studies_page(database: sqlite3.Connection, offset: Any = 0, limit: Any = 100) -> dict[str, Any]:
    """The studies (``studies.list``), most recently active first; ``unassigned`` when it has campaigns."""
    offset, limit = _page(offset, limit)
    active = ",".join("?" * len(ACTIVE_STATES))
    studies: dict[str, dict[str, Any]] = {}
    for row in database.execute(
        f"SELECT COALESCE(json_extract(specification,'$.study.id'),?) sid,COUNT(*),"
        f"SUM(state IN ({active})),MIN(created_ns),"
        "MAX(MAX(created_ns,COALESCE(execution_finished_ns,0))),"
        "json_group_array(DISTINCT COALESCE(json_extract(specification,'$.study.evidence_level'),"
        f"'exploratory')) FROM campaigns WHERE NOT ({TUNING_CAMPAIGN}) GROUP BY sid",
        (UNASSIGNED, *ACTIVE_STATES),
    ):
        studies[row[0]] = dict(
            study_id=row[0],
            campaigns=row[1],
            active_campaigns=row[2],
            created_ns=row[3],
            last_activity_ns=row[4],
            evidence_levels=sorted(json.loads(row[5])),
            runs=0,
            final_runs=0,
        )
    # Runs of the campaigns an export holds: tuning runs and campaigns a restarted copy replaces are not.
    condition, replaced = held(database)
    for identity, runs, finals in database.execute(
        "SELECT r.study_id,COUNT(*),SUM(e.final_attempt_id IS r.attempt_id) FROM run_records r "
        f"JOIN experiments e ON e.id=r.experiment_id WHERE r.run_record_version=? AND {condition} "
        "GROUP BY r.study_id",
        (RUN_RECORD_VERSION, *replaced),
    ):
        if identity in studies:
            studies[identity].update(runs=runs, final_runs=finals)
    # The plan and evidence level of each study's newest campaign, and the last repetition a new campaign
    # continues after (kept in study_repetitions when campaigns are deleted; none for unassigned campaigns,
    # which do not pair).
    for identity, plan, digest, level in database.execute(
        "SELECT sid,plan,digest,level FROM (SELECT COALESCE(json_extract(specification,'$.study.id'),?) sid,"
        "json_extract(specification,'$.study.plan') plan,json_extract(specification,'$.study.plan_sha256') "
        "digest,COALESCE(json_extract(specification,'$.study.evidence_level'),'exploratory') level,"
        "ROW_NUMBER() OVER (PARTITION BY COALESCE(json_extract(specification,'$.study.id'),?) "
        f"ORDER BY created_ns DESC,id DESC) n FROM campaigns WHERE NOT ({TUNING_CAMPAIGN})) WHERE n=1",
        (UNASSIGNED, UNASSIGNED),
    ):
        studies[identity].update(
            plan=plan,
            plan_sha256=digest,
            evidence_level=level,
            last_repetition=None if identity == UNASSIGNED else last_repetition(database, identity),
        )
    rows = sorted(studies.values(), key=lambda s: (-s["last_activity_ns"], s["study_id"]))
    page = rows[offset : offset + limit]
    return dict(
        rows=page,
        total=len(rows),
        offset=offset,
        next_offset=offset + len(page) if offset + len(page) < len(rows) else None,
    )


def _mix_value(column: str, record: dict[str, Any]) -> Any:
    """The value of ``column`` that ``refused_mixes`` compares (None: no data of that kind)."""
    value = record[column]
    if column == "mapping_condition" and value is None and record["comparison_key"] is not None:
        return BEFORE_MAPPING_CONDITION
    return value


def refused_mixes(full: Iterable[dict[str, Any]]) -> list[str]:
    """The export's refusal (``exports._run_rows``) and ``study.get``'s warning: full final records whose
    record, dataset-schema or coverage-metric versions or mapping conditions differ. A run without a value (a
    startup failure before recording began has no coverage block) holds no data of that version and mixes
    nothing, except a null mapping condition of a record with a comparison key: that run mapped under the
    condition before I10 (``BEFORE_MAPPING_CONDITION``)."""
    full = list(full)
    mixes = []
    for column in REFUSED_MIXES:
        found = (_mix_value(column, record) for record in full)
        values = sorted({str(value) for value in found if value is not None})
        if len(values) > 1:
            mixes.append(f"Final runs mix {column} values {', '.join(values)}")
    return mixes


def homogeneity(
    specifications: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    documents: Mapping[str, Any],
) -> dict[str, Any]:
    """What should be equal across a study's campaigns and full final records. Differences are warnings
    only: the researcher judges them. Record, schema and metric version mixes are ``refused_mixes``."""
    specifications, full = list(specifications), list(records)
    values = {
        column: sorted({canonical(record[column]) for record in full}) for column in HOMOGENEOUS
    }
    studies = [specification.get("study") or {} for specification in specifications]
    for name in ("plan_sha256", "evidence_level"):
        values[f"study.{name}"] = sorted({canonical(study.get(name)) for study in studies})
    for name in ("default_parameter_hashes", "time_limit_rule"):
        values[name] = sorted(
            {canonical(specification.get(name)) for specification in specifications}
        )
    # The registry and robot catalog the engine read (provenance version 2; null in version 1 documents,
    # so a study mixing both warns too: the registry did not exist then).
    identities = {record["provenance_id"] for record in full if record["provenance_id"]}
    for name in REGISTRY_FILES:
        values[name] = sorted(
            {canonical((documents.get(identity) or {}).get(name)) for identity in identities}
        )
    cells: dict[str, set[str]] = defaultdict(set)
    for record in full:
        cells[f"{record['robot_id']}/{record['world_id']}/{record['spawn_id']}"].add(
            record["comparison_key"] or ""
        )
    warnings = [
        f"{name} differs within the study" for name, found in values.items() if len(found) > 1
    ]
    warnings += [
        f"{cell}: {len(found)} comparison keys"
        for cell, found in sorted(cells.items())
        if len(found) > 1
    ]
    for identity in sorted(identities):
        repositories = (documents.get(identity) or {}).get("repositories") or {}
        if any((state or {}).get("dirty") for state in repositories.values()):
            warnings.append(
                f"provenance {identity[:12]}: a repository was dirty (confirmatory check)"
            )
    return dict(
        values=values,
        comparison_keys={k: sorted(v) for k, v in sorted(cells.items())},
        warnings=warnings,
    )


RECORD_COLUMNS = (
    "run_record_version",
    "record_kind",
    "is_final",
    "robot_id",
    "world_id",
    "spawn_id",
    "profile_id",
    "parameter_hash",
    "comparison_key",
)
RECORD_VALUES = ("dataset_schema", "coverage_metric_version", "mapping_condition", *HOMOGENEOUS)
SELECTORS = (
    "robot_id",
    "world_id",
    "spawn_id",
    "profile_id",
    "parameter_hash",
    "comparison_key",
    "dataset_schema",
    "coverage_metric_version",
    "timing_rule",
    "debug_streams",
)


def _counted(rows: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Each value of ``name`` with its run and final-run counts, null first (values keyed by their
    canonical JSON, so lists count too)."""
    runs: Counter[str] = Counter()
    finals: Counter[str] = Counter()
    values: dict[str, Any] = {}
    for row in rows:
        key = canonical(row[name])
        values[key] = row[name]
        runs[key] += 1
        finals[key] += row["is_final"]
    return [
        dict(value=values[key], runs=runs[key], final_runs=finals[key])
        for key in sorted(runs, key=lambda key: (values[key] is not None, key))
    ]


def read_study(database_path: Path, study_id: str) -> dict[str, Any]:
    """``study`` on its own read-only connection, in one read transaction (the background thread)."""
    with closing(sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)) as database:
        database.execute("BEGIN")
        return study(database, study_id)


def study(database: sqlite3.Connection, study_id: str) -> dict[str, Any]:
    """One study (``study.get``): campaigns, counts, selector values, warnings and finished exports.

    Counts, selectors and warnings cover the campaigns an export holds: a campaign a restarted copy
    replaces is listed, marked, and left out of them. It reads every record of the study once, so the
    engine builds it off its loop (``files.BackgroundReads``) on a read-only connection.
    """
    rows = study_campaign_rows(database, study_id)
    if not rows:
        raise ContractError("study_id", "Study has no campaigns", "not_found")
    superseded = _superseded(database)
    campaigns, specifications = [], []
    for identity, name, state, specification_text, created, finished in rows:
        specification = json.loads(specification_text)
        block = specification.get("study") or {}
        campaigns.append(
            dict(
                id=identity,
                name=name,
                state=state,
                created_ns=created,
                execution_finished_ns=finished,
                evidence_level=block.get("evidence_level", "exploratory"),
                plan=block.get("plan"),
                plan_sha256=block.get("plan_sha256"),
                first_repetition=block.get("first_repetition", 0),
                repetitions=specification["repetitions"],
                debug_streams=bool(specification.get("debug_streams")),
                superseded_by_campaign_id=superseded.get(identity),
            )
        )
        if identity not in superseded:
            specifications.append(specification)
    kept = [c["id"] for c in campaigns if c["superseded_by_campaign_id"] is None]
    marks = ",".join("?" * len(kept))
    planned, done, unresolved = database.execute(
        f"SELECT COUNT(*),SUM(state='DONE'),SUM(state='UNRESOLVED') FROM experiments "
        f"WHERE campaign_id IN ({marks})",
        kept,
    ).fetchone()
    attempts, finished = database.execute(
        f"SELECT COUNT(*),SUM(phase='FINISHED') FROM attempts WHERE campaign_id IN ({marks})", kept
    ).fetchone()
    every = [
        dict(zip((*RECORD_COLUMNS, *RECORD_VALUES), row, strict=True))
        for row in database.execute(
            "SELECT r.run_record_version,r.record_kind,(e.final_attempt_id IS r.attempt_id),"
            "r.robot_id,r.world_id,r.spawn_id,r.profile_id,r.parameter_hash,r.comparison_key,"
            + ",".join(f"json_extract(r.record,'$.{name}')" for name in RECORD_VALUES)
            + " FROM run_records r JOIN experiments e ON e.id=r.experiment_id "
            f"WHERE r.campaign_id IN ({marks})",
            kept,
        )
    ]
    for record in every:  # json_extract returns booleans as integers and lists as JSON text
        if record["diagnostics_enabled"] is not None:
            record["diagnostics_enabled"] = bool(record["diagnostics_enabled"])
        if record["debug_streams"] is not None:
            record["debug_streams"] = json.loads(record["debug_streams"])
    current = [r for r in every if r["run_record_version"] == RUN_RECORD_VERSION]
    selectors = {name: _counted(current, name) for name in SELECTORS}
    selectors["run_record_version"] = _counted(every, "run_record_version")
    full = [r for r in current if r["is_final"] and r["record_kind"] == "full"]
    identities = sorted({record["provenance_id"] for record in full if record["provenance_id"]})
    documents = {
        row[0]: json.loads(row[1])
        for row in database.execute(
            f"SELECT id,document FROM provenance WHERE id IN ({','.join('?' * len(identities))})",
            identities,
        )
    }
    finished_exports = [
        dict(json.loads(row[0]), utc_ns=row[1])
        for row in database.execute(
            "SELECT payload,utc_ns FROM events WHERE kind='study.exported' "
            "AND json_extract(payload,'$.study_id')=? ORDER BY sequence DESC LIMIT ?",
            (study_id, EXPORTS_LISTED),
        )
    ]
    return dict(
        study_id=study_id,
        campaigns=campaigns,
        evidence_levels=sorted({c["evidence_level"] for c in campaigns}),
        counts=dict(
            planned=planned,
            done=done or 0,
            unresolved=unresolved or 0,
            attempts=attempts,
            finished_attempts=finished or 0,
            records=len(current),
            final_runs=sum(r["is_final"] for r in current),
        ),
        selectors=selectors,
        warnings=[f"{mix}: the export is refused" for mix in refused_mixes(full)]
        + homogeneity(specifications, full, documents)["warnings"],
        finished_exports=finished_exports,
    )
