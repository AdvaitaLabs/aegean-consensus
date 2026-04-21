from __future__ import annotations

from aegean.core.agent import AgentRegistry
from aegean.core.models import CollaborationMode
from aegean.memory.global_memory import GlobalMemorySystem
from aegean.memory.knowledge_base import KnowledgeBase
from aegean.investment.models import AssetType, InvestmentMode
from aegean.investment.service import InvestmentAnalysisService
from aegean.services.group_chat_service import GroupChatService


def test_group_shared_knowledge_initializes_with_four_categories() -> None:
    service = GroupChatService(agent_registry=AgentRegistry())
    group = service.create_group(
        "Research Team",
        created_by="user_1",
        mode=CollaborationMode.CONSENSUS,
    )

    shared = service.get_group_shared_knowledge(group.group_id)

    assert shared.static_documents == []
    assert shared.historical_case_ids == []
    assert shared.skills == []
    assert shared.knowledge_graph_ids == []
    assert shared.metadata["document_count"] == 0
    assert shared.metadata["case_count"] == 0
    assert shared.metadata["skill_count"] == 0
    assert shared.metadata["graph_count"] == 0


def test_group_shared_knowledge_accepts_documents_skills_and_cases() -> None:
    service = GroupChatService(agent_registry=AgentRegistry())
    group = service.create_group(
        "Research Team",
        created_by="user_1",
    )

    service.add_group_document(
        group.group_id,
        doc_id="doc-1",
        category="investment_methodology",
        title="DCF Playbook",
        summary="Valuation reference",
        metadata={"group_id": group.group_id},
    )
    service.add_group_skill(
        group.group_id,
        skill_id="fundamental_analysis",
        name="Fundamental Analysis",
        description="Analyze business quality.",
        applicable_task_types=["equity_analysis"],
    )
    service.add_group_case_reference(group.group_id, "consensus-123")

    shared = service.get_group_shared_knowledge(group.group_id)

    assert [doc.doc_id for doc in shared.static_documents] == ["doc-1"]
    assert shared.historical_case_ids == ["consensus-123"]
    assert [skill.skill_id for skill in shared.skills] == ["fundamental_analysis"]


def test_group_documents_sync_to_knowledge_base() -> None:
    kb = KnowledgeBase()
    memory = GlobalMemorySystem(knowledge_base=kb)
    service = GroupChatService(agent_registry=AgentRegistry(), memory_system=memory)
    group = service.create_group(
        "Research Team",
        created_by="user_1",
    )

    service.add_group_document(
        group.group_id,
        doc_id="doc-kb-1",
        category="investment_methodology",
        title="Margin of Safety",
        summary="Use conservative downside cases.",
        content="Use conservative downside cases and demand a margin of safety.",
    )

    import asyncio

    async def _run():
        return await kb.get_document("doc-kb-1")

    document = asyncio.run(_run())

    assert document is not None
    assert document.doc_id == "doc-kb-1"
    assert document.metadata["group_id"] == group.group_id
    assert document.metadata["title"] == "Margin of Safety"


def test_group_shared_knowledge_enriches_memory_context() -> None:
    kb = KnowledgeBase()
    memory = GlobalMemorySystem(knowledge_base=kb)

    import asyncio

    async def _run():
        await kb.add_document(
            content="Equity valuation checklist for downside protection.",
            category="investment_methodology",
            metadata={"group_id": "group-1", "title": "Valuation Playbook"},
        )
        return await memory.retrieve_context(
            query="How to evaluate downside protection?",
            group_id="group-1",
            categories=["investment_methodology"],
            metadata_filters={"group_id": "group-1"},
            group_context={
                "skills": ["fundamental_analysis", "equity_valuation"],
                "knowledge_graph_ids": ["graph-1"],
            },
        )

    context = asyncio.run(_run())
    rendered = context.format_for_prompt()

    assert context.group_id == "group-1"
    assert "fundamental_analysis" in context.group_skills
    assert "graph-1" in context.group_graph_ids
    assert "Group ID: group-1" in rendered
    assert "Group Skills:" in rendered


def test_investment_service_resolves_skill_profiles() -> None:
    service = InvestmentAnalysisService(agent_registry=AgentRegistry())

    skill_ids = service._selected_skills_for_asset(AssetType.EQUITY)
    profiles = service._resolve_skill_profiles(skill_ids)

    assert skill_ids == ["fundamental_analysis", "equity_valuation"]
    assert profiles[0]["skill_id"] == "fundamental_analysis"
    assert profiles[0]["name"] == "Fundamental Analysis"
    assert "company_fundamentals" in profiles[0]["required_data_sources"]


def test_investment_service_formats_role_specific_normalized_data() -> None:
    service = InvestmentAnalysisService(agent_registry=AgentRegistry())
    normalized = {
        "market": {"price": 100, "change_pct": 0.03},
        "fundamentals": {"pe_ttm": 18.5, "revenue_growth": 0.12},
        "news": ["New product cycle expected next quarter."],
    }

    fundamental_view = service._format_normalized_data_for_prompt(
        normalized,
        "fundamental_specialist",
    )
    macro_view = service._format_normalized_data_for_prompt(
        normalized,
        "macro_specialist",
    )

    assert "Fundamental data:" in fundamental_view
    assert "Market data:" in fundamental_view
    assert "News:" not in fundamental_view
    assert "Market data:" in macro_view
    assert "News:" in macro_view
    assert "Fundamental data:" not in macro_view


def test_investment_service_builds_equity_multi_role_panel() -> None:
    service = InvestmentAnalysisService(agent_registry=AgentRegistry())

    panel_roles = service._panel_roles_for_task("equity_analysis")
    focused_skills = [service._skills_for_role(role, ["fundamental_analysis", "equity_valuation"]) for role in panel_roles]

    assert panel_roles == [
        "fundamental_specialist",
        "valuation_specialist",
        "macro_specialist",
        "risk_specialist",
    ]
    assert focused_skills[0] == ["fundamental_analysis"]
    assert focused_skills[1] == ["equity_valuation"]
    assert focused_skills[2] == ["fundamental_analysis", "equity_valuation"]
    assert focused_skills[3] == ["fundamental_analysis", "equity_valuation"]


def test_investment_service_builds_panel_metadata_and_committee_trace() -> None:
    service = InvestmentAnalysisService(agent_registry=AgentRegistry())

    framework = service._build_analysis_framework(
        mode=InvestmentMode.ROUNDTABLE,
        task_type="equity_analysis",
        selected_skills=["fundamental_analysis", "equity_valuation"],
        data_sources=["public_market_data", "company_fundamentals"],
        resolved_skill_profiles=[],
        group_injection=None,
        panel_roles=["fundamental_specialist", "valuation_specialist", "macro_specialist", "risk_specialist"],
        panel_role_skills={
            "fundamental_specialist": ["fundamental_analysis"],
            "valuation_specialist": ["equity_valuation"],
            "macro_specialist": ["fundamental_analysis", "equity_valuation"],
            "risk_specialist": ["fundamental_analysis", "equity_valuation"],
        },
        panel_role_data_focus={
            "fundamental_specialist": ["fundamentals", "market"],
            "valuation_specialist": ["fundamentals", "market"],
            "macro_specialist": ["market", "news"],
            "risk_specialist": ["news", "market"],
        },
    )

    assert framework.panel_roles[0] == "fundamental_specialist"
    assert framework.panel_role_data_focus["macro_specialist"] == ["market", "news"]
    assert "Investment committee panel active" in framework.why_selected[-1]

    trace = service._build_consensus_trace(
        agent_outputs=[
            service._solution_to_output(
                type("S", (), {"agent_id": "a1", "answer": "BUY because fundamentals improved", "confidence": 0.8})(),
                "equity_analysis",
                role="fundamental_specialist",
            ),
            service._solution_to_output(
                type("S", (), {"agent_id": "a2", "answer": "HOLD due to valuation risk", "confidence": 0.7})(),
                "equity_analysis",
                role="valuation_specialist",
            ),
        ],
        final_action="hold",
        final_confidence=0.82,
    )

    assert [round_info.stage for round_info in trace.rounds] == [
        "opening_statements",
        "cross_challenge",
        "chair_synthesis",
    ]
    assert "opening statements, cross-challenge, and chair synthesis" in trace.final_summary

