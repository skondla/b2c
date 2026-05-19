"""Integration tests for the LangGraph mortgage workflow."""

import pytest
from unittest.mock import patch, MagicMock
from src.graphs.states import LoanStage, default_state


class TestMortgageState:
    def test_default_state_initialized(self):
        state = default_state(
            application_id="app-001",
            borrower_id="borrower-001",
            borrower_name="Alex Johnson",
            borrower_email="alex@example.com",
        )
        assert state["stage"] == LoanStage.DISCOVERY.value
        assert state["application_id"] == "app-001"
        assert state["messages"] == []
        assert state["documents_uploaded"] == []
        assert state["e_consent_given"] is False
        assert state["rate_locked"] is False

    def test_stage_values_are_strings(self):
        for stage in LoanStage:
            assert isinstance(stage.value, str)

    def test_stage_order_is_complete(self):
        from src.graphs.states import STAGE_ORDER
        assert LoanStage.DISCOVERY in STAGE_ORDER
        assert LoanStage.COMPLETED in STAGE_ORDER
        assert len(STAGE_ORDER) == 10  # 9 journey stages + COMPLETED


class TestGraphRouting:
    def test_route_by_stage_discovery(self):
        from src.graphs.mortgage_graph import route_by_stage
        state = {"stage": LoanStage.DISCOVERY.value}
        assert route_by_stage(state) == "discovery"

    def test_route_by_stage_prequalification(self):
        from src.graphs.mortgage_graph import route_by_stage
        state = {"stage": LoanStage.PREQUALIFICATION.value}
        assert route_by_stage(state) == "prequalification"

    def test_route_by_stage_completed(self):
        from src.graphs.mortgage_graph import route_by_stage
        from langgraph.graph import END
        state = {"stage": LoanStage.COMPLETED.value}
        assert route_by_stage(state) == END

    def test_route_by_stage_declined(self):
        from src.graphs.mortgage_graph import route_by_stage
        state = {"stage": LoanStage.DECLINED.value}
        assert route_by_stage(state) == "decline"

    def test_should_use_tools_no_tool_calls(self):
        from src.graphs.mortgage_graph import should_use_tools
        from langchain_core.messages import AIMessage
        state = {"messages": [AIMessage(content="Here is your pre-qualification result.")]}
        assert should_use_tools(state) == "supervisor"

    def test_should_use_tools_empty_messages(self):
        from src.graphs.mortgage_graph import should_use_tools
        state = {"messages": []}
        assert should_use_tools(state) == "supervisor"


class TestDiscoveryNode:
    def test_discovery_node_sets_stage(self):
        from src.graphs.mortgage_graph import discovery_node
        state = default_state(
            application_id="app-001",
            borrower_id="b-001",
            borrower_name="Test User",
            borrower_email="test@example.com",
        )
        result = discovery_node(state)
        assert result["stage"] == LoanStage.PREQUALIFICATION.value
        assert len(result["messages"]) == 1
        assert "Welcome" in result["messages"][0].content

    def test_discovery_node_personalized_message(self):
        from src.graphs.mortgage_graph import discovery_node
        state = default_state("app-002", "b-002", "Jordan Smith", "jordan@example.com")
        result = discovery_node(state)
        assert "Jordan Smith" in result["messages"][0].content


class TestDeclineNode:
    def test_decline_node_sets_declined_stage(self):
        from src.graphs.mortgage_graph import decline_node
        state = default_state("app-003", "b-003", "Test", "test@example.com")
        result = decline_node(state)
        assert result["stage"] == LoanStage.DECLINED.value
        assert "adverse_action_notice_required" in result["compliance_flags"]
