"""Setu governance subnet adapter service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from aegean.core.models import CollaborationMode, GroupConsensusResult
from aegean.services.group_chat_service import GroupChatService
from aegean.setu_models import (
    EvaluateAcceptedResponse,
    GovernanceDecision,
    ProposalContent,
    SetuEvaluateRequest,
    SetuResultResponse,
    SetuTaskRecord,
    SetuTaskStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_GOVERNANCE_SUBNET_ID = (
    "0x0110000000000000000000000000000000000000000000000000000000000000"
)


class SetuService:
    """Adapter between Setu validator protocol and Aegean group consensus."""

    def __init__(
        self,
        group_service: GroupChatService,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.group_service = group_service
        self.config = config or {}
        self.tasks: Dict[str, SetuTaskRecord] = {}
        self.task_locks: Dict[str, asyncio.Lock] = {}
        self.bound_groups: Dict[str, str] = {}
        self.default_subnet_id = self.config.get(
            "default_subnet_id", DEFAULT_GOVERNANCE_SUBNET_ID
        )
        self.default_group_name = self.config.get(
            "default_group_name", "setu-governance-group"
        )
        self.default_group_description = self.config.get(
            "default_group_description",
            "Dedicated consensus group bound to Setu governance system subnet",
        )
        self.default_created_by = self.config.get("default_created_by", "setu-system")
        self.default_initial_members = self._resolve_default_members(
            self.config.get("initial_members")
        )
        self.default_quorum_threshold = float(
            self.config.get("quorum_threshold", 0.5)
        )
        self.default_stability_horizon = int(
            self.config.get("stability_horizon", 2)
        )
        self.default_max_rounds = int(self.config.get("max_rounds", 3))
        self.default_task_timeout_secs = int(self.config.get("task_timeout_secs", 300))
        self.callback_enabled = bool(self.config.get("callback_enabled", False))
        self.callback_timeout_secs = float(self.config.get("callback_timeout_secs", 5.0))

        self._ensure_bound_group(
            subnet_id=self.default_subnet_id,
            group_name=self.default_group_name,
            description=self.default_group_description,
            created_by=self.default_created_by,
            initial_members=self.default_initial_members,
        )

    def _resolve_default_members(self, initial_members: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if initial_members:
            return initial_members

        agents = self.group_service.agent_registry.get_all_agents()
        resolved = []
        for agent in agents:
            resolved.append(
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role or "setu_governance_validator",
                    "capability_weight": agent.capability_weight,
                    "specialization": agent.specialization,
                }
            )
        return resolved

    def _ensure_bound_group(
        self,
        subnet_id: str,
        group_name: str,
        description: str,
        created_by: str,
        initial_members: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        existing_group_id = self.bound_groups.get(subnet_id)
        if existing_group_id:
            return existing_group_id

        for group in self.group_service.list_groups(created_by=created_by):
            metadata = group.metadata or {}
            if (
                metadata.get("integration") == "setu"
                and metadata.get("subnet_id") == subnet_id
            ):
                self.bound_groups[subnet_id] = group.group_id
                self._ensure_group_members(group.group_id, initial_members or [])
                return group.group_id

        group = self.group_service.create_group(
            group_name=group_name,
            created_by=created_by,
            description=description,
            mode=CollaborationMode.CONSENSUS,
            metadata={
                "integration": "setu",
                "binding_type": "exclusive",
                "subnet_id": subnet_id,
                "adapter": "setu",
                "protocol": "system-subnet-governance-v1",
                "exclusive_use": True,
            },
            initial_members=initial_members or [],
        )
        self.bound_groups[subnet_id] = group.group_id
        logger.info(
            "Setu adapter bound subnet %s -> group %s",
            subnet_id,
            group.group_id,
        )
        return group.group_id

    def _ensure_group_members(self, group_id: str, initial_members: List[Dict[str, Any]]) -> None:
        existing_member_ids = {m.agent_id for m in self.group_service.get_members(group_id)}
        for member in initial_members:
            agent_id = member.get("agent_id")
            if not agent_id or agent_id in existing_member_ids:
                continue
            try:
                self.group_service.add_member(
                    group_id=group_id,
                    agent_id=agent_id,
                    role=member.get("role"),
                    capability_weight=member.get("capability_weight", 1.0),
                    specialization=member.get("specialization"),
                )
            except ValueError:
                continue

    def get_bound_group(self, subnet_id: Optional[str] = None) -> Dict[str, str]:
        effective_subnet_id = subnet_id or self.default_subnet_id
        group_id = self.bound_groups.get(effective_subnet_id)
        if not group_id:
            group_id = self._ensure_bound_group(
                subnet_id=effective_subnet_id,
                group_name=self.default_group_name,
                description=self.default_group_description,
                created_by=self.default_created_by,
                initial_members=self.default_initial_members,
            )
        group = self.group_service.get_group(group_id)
        if not group:
            raise ValueError(f"Bound Setu group {group_id} not found")
        return {
            "subnet_id": effective_subnet_id,
            "group_id": group.group_id,
            "group_name": group.group_name,
        }

    async def submit_evaluation(self, request: SetuEvaluateRequest) -> EvaluateAcceptedResponse:
        lock = self.task_locks.setdefault(request.task_id, asyncio.Lock())
        async with lock:
            existing = self.tasks.get(request.task_id)
            if existing:
                return EvaluateAcceptedResponse(
                    accepted=True,
                    task_id=existing.task_id,
                    status=existing.status,
                    bound_group_id=existing.group_id,
                    bound_group_name=existing.group_name,
                    subnet_id=existing.subnet_id,
                )

            bound = self.get_bound_group(request.system_context.subnet_id)
            record = SetuTaskRecord(
                task_id=request.task_id,
                subnet_id=bound["subnet_id"],
                group_id=bound["group_id"],
                group_name=bound["group_name"],
                callback_token=request.callback_token,
                callback_url=request.callback_url,
                proposal=request.proposal,
                system_context=request.system_context,
                metadata={
                    "source": "setu-validator",
                    "accepted_at": datetime.now(timezone.utc).isoformat(),
                    "timeout_secs": self.default_task_timeout_secs,
                },
            )
            self.tasks[request.task_id] = record

            logger.info(
                "Setu task accepted task_id=%s subnet_id=%s group_id=%s proposal_type=%s",
                record.task_id,
                record.subnet_id,
                record.group_id,
                record.proposal.proposal_type,
            )
            asyncio.create_task(self._run_task(record.task_id))

            return EvaluateAcceptedResponse(
                accepted=True,
                task_id=record.task_id,
                status=record.status,
                bound_group_id=record.group_id,
                bound_group_name=record.group_name,
                subnet_id=record.subnet_id,
            )

    def get_result(self, task_id: str) -> SetuResultResponse:
        record = self.tasks.get(task_id)
        if not record:
            return SetuResultResponse(status=SetuTaskStatus.NOT_FOUND)

        return SetuResultResponse(
            status=record.status,
            decision=record.decision,
            task_id=record.task_id,
            consensus_id=record.consensus_id,
            group_id=record.group_id,
            subnet_id=record.subnet_id,
            updated_at=record.updated_at,
            error=record.error,
        )

    async def _run_task(self, task_id: str) -> None:
        record = self.tasks.get(task_id)
        if not record:
            return

        try:
            task = self._proposal_to_task(record.proposal, record)
            result = await self.group_service.execute_consensus(
                group_id=record.group_id,
                task=task,
                message_id=record.task_id,
                quorum_threshold=self.default_quorum_threshold,
                stability_horizon=self.default_stability_horizon,
                max_rounds=self.default_max_rounds,
                risk_context=None,
            )
            decision = self._consensus_to_decision(result, record.proposal)
            record.consensus_id = result.consensus_id
            record.decision = decision
            record.status = SetuTaskStatus.COMPLETED
            record.metadata["consensus_reached"] = result.consensus_reached
            record.metadata["rounds_used"] = result.rounds_used
            record.metadata["participating_agents"] = result.participating_agents
            record.mark_updated()
            if self.callback_enabled:
                await self._post_callback(record)
            logger.info(
                "Setu task completed task_id=%s consensus_id=%s approved=%s",
                record.task_id,
                result.consensus_id,
                decision.approved,
            )
        except Exception as exc:
            logger.error("Setu task failed task_id=%s error=%s", task_id, exc, exc_info=True)
            record.status = SetuTaskStatus.FAILED
            record.error = str(exc)
            record.mark_updated()

    def _proposal_to_task(self, proposal: ProposalContent, record: SetuTaskRecord) -> str:
        action_json = json.dumps(proposal.action, ensure_ascii=False, sort_keys=True)
        system_context_json = json.dumps(
            record.system_context.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        guidance = self._build_proposal_guidance(proposal)
        return (
            "You are the governance decision committee for the Setu system subnet.\n"
            "Evaluate the following governance proposal and decide whether it should be approved.\n\n"
            f"Task ID: {record.task_id}\n"
            f"Subnet ID: {record.subnet_id}\n"
            f"Validator ID: {record.system_context.validator_id or 'unknown'}\n"
            f"Proposal Type: {proposal.proposal_type}\n"
            f"Proposer: {proposal.proposer}\n"
            f"Title: {proposal.title}\n"
            f"Description: {proposal.description}\n"
            f"Action: {action_json}\n"
            f"System Context: {system_context_json}\n\n"
            f"Proposal-specific review guidance:\n{guidance}\n\n"
            "Return a strict JSON object only, with this exact schema:\n"
            '{"approved": true, "reasoning": "short explanation", "conditions": ["optional condition"]}\n\n'
            "Rules:\n"
            "1. approved must be true or false.\n"
            "2. reasoning must be concise, concrete, and governance-focused.\n"
            "3. conditions must be an array of strings and can be empty.\n"
            "4. Do not include markdown fences.\n"
            "5. If information is insufficient or risky, prefer reject with a clear reason."
        )

    def _build_proposal_guidance(self, proposal: ProposalContent) -> str:
        proposal_type = (proposal.proposal_type or "").strip().lower()
        if proposal_type == "parameterchange":
            return (
                "Review whether the parameter change is reversible, proportionate, operationally safe, "
                "and clearly justified by the description and action payload."
            )
        if proposal_type == "validatorslash":
            return (
                "Review whether there is clear validator fault evidence, whether the slash amount is proportionate, "
                "and whether due process appears to be respected."
            )
        if proposal_type == "disputeresolution":
            return (
                "Review whether the dispute resolution is internally consistent, supported by the proposal description, "
                "and minimizes governance ambiguity."
            )
        if proposal_type == "subnetpolicy":
            return (
                "Review whether the subnet policy change is enforceable, clearly scoped, and aligned with network governance norms."
            )
        return "Review whether the proposal is clearly justified, safe to execute, and consistent with governance intent."

    def _consensus_to_decision(
        self,
        result: GroupConsensusResult,
        proposal: ProposalContent,
    ) -> GovernanceDecision:
        parsed = self._extract_decision_payload(result)
        if parsed:
            return GovernanceDecision(
                approved=bool(parsed.get("approved", False)),
                reasoning=str(parsed.get("reasoning") or "No reasoning provided."),
                conditions=self._normalize_conditions(parsed.get("conditions")),
            )

        fallback_reasoning = (
            result.final_solution.answer.strip()
            if result.final_solution and result.final_solution.answer.strip()
            else f"Proposal {proposal.title} was evaluated but no structured decision could be parsed."
        )
        approved = self._infer_approved_from_text(fallback_reasoning)
        return GovernanceDecision(
            approved=approved,
            reasoning=fallback_reasoning[:2000],
            conditions=[],
        )

    def _extract_decision_payload(
        self,
        result: GroupConsensusResult,
    ) -> Optional[Dict[str, Any]]:
        candidates: List[str] = []
        if result.final_solution and result.final_solution.answer:
            candidates.append(result.final_solution.answer)
        for solution in result.agent_responses:
            if solution.answer:
                candidates.append(solution.answer)

        for candidate in candidates:
            parsed = self._parse_json_object(candidate)
            if parsed and isinstance(parsed.get("approved"), bool):
                return parsed
        return None

    def _parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        fragment = cleaned[start : end + 1]
        try:
            parsed = json.loads(fragment)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_conditions(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _infer_approved_from_text(self, text: str) -> bool:
        normalized = text.lower()
        negative_markers = ["reject", "rejected", "deny", "denied", "disapprove", "false"]
        positive_markers = ["approve", "approved", "accept", "accepted", "true"]

        if any(marker in normalized for marker in negative_markers):
            return False
        if any(marker in normalized for marker in positive_markers):
            return True
        return False

    async def _post_callback(self, record: SetuTaskRecord) -> None:
        if not record.callback_url or not record.decision:
            return

        payload = {
            "task_id": record.task_id,
            "proposal_id": record.task_id,
            "callback_token": record.callback_token,
            "decision": record.decision.model_dump(mode="json"),
            "subnet_id": record.subnet_id,
            "group_id": record.group_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.callback_timeout_secs) as client:
                response = await client.post(record.callback_url, json=payload)
                response.raise_for_status()
            record.metadata["callback_delivered"] = True
            record.metadata["callback_delivered_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                "Setu callback delivered task_id=%s callback_url=%s",
                record.task_id,
                record.callback_url,
            )
        except Exception as exc:
            record.metadata["callback_delivered"] = False
            record.metadata["callback_error"] = str(exc)
            logger.warning(
                "Setu callback failed task_id=%s callback_url=%s error=%s",
                record.task_id,
                record.callback_url,
                exc,
            )


def build_setu_config_from_env() -> Dict[str, Any]:
    """Build Setu adapter config from environment variables."""

    initial_members_raw = os.getenv("SETU_GROUP_INITIAL_MEMBERS", "").strip()
    initial_members: Optional[List[Dict[str, Any]]] = None
    if initial_members_raw:
        try:
            parsed = json.loads(initial_members_raw)
            if isinstance(parsed, list):
                initial_members = [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            logger.warning("Invalid SETU_GROUP_INITIAL_MEMBERS JSON, ignoring")

    return {
        "default_subnet_id": os.getenv(
            "SETU_GOVERNANCE_SUBNET_ID", DEFAULT_GOVERNANCE_SUBNET_ID
        ),
        "default_group_name": os.getenv(
            "SETU_GROUP_NAME", "setu-governance-group"
        ),
        "default_group_description": os.getenv(
            "SETU_GROUP_DESCRIPTION",
            "Dedicated consensus group bound to Setu governance system subnet",
        ),
        "default_created_by": os.getenv("SETU_GROUP_CREATED_BY", "setu-system"),
        "quorum_threshold": float(os.getenv("SETU_QUORUM_THRESHOLD", "0.5")),
        "stability_horizon": int(os.getenv("SETU_STABILITY_HORIZON", "2")),
        "max_rounds": int(os.getenv("SETU_MAX_ROUNDS", "3")),
        "task_timeout_secs": int(os.getenv("SETU_TASK_TIMEOUT_SECS", "300")),
        "callback_enabled": os.getenv("SETU_CALLBACK_ENABLED", "false").lower() == "true",
        "callback_timeout_secs": float(os.getenv("SETU_CALLBACK_TIMEOUT_SECS", "5.0")),
        "initial_members": initial_members,
    }

