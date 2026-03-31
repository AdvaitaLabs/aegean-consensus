#!/usr/bin/env python3
"""
Test script to verify group chat fixes.
Tests: initial members, is_active flag, and collaboration modes.
"""

import asyncio
import requests
import json

BASE_URL = "http://173.249.5.203:8000"

def test_list_agents():
    """Test listing available agents."""
    print("\n=== Test 1: List Available Agents ===")
    resp = requests.get(f"{BASE_URL}/api/v1/groups/agents")
    print(f"Status: {resp.status_code}")
    agents = resp.json()
    print(f"Available agents: {len(agents)}")
    for agent in agents:
        print(f"  - {agent['agent_id']}: weight={agent['capability_weight']}")
    return agents

def test_create_group_with_members(agents):
    """Test creating group with initial members."""
    print("\n=== Test 2: Create Group with Initial Members ===")
    
    payload = {
        "group_name": "Test Risk Team",
        "description": "Testing initial members feature",
        "mode": "consensus",
        "created_by": "test_user",
        "initial_members": [
            {
                "agent_id": agents[0]["agent_id"],
                "role": "analyst",
                "capability_weight": 1.0
            },
            {
                "agent_id": agents[1]["agent_id"],
                "role": "reviewer",
                "capability_weight": 0.9
            }
        ]
    }
    
    resp = requests.post(
        f"{BASE_URL}/api/v1/groups/",
        json=payload
    )
    print(f"Status: {resp.status_code}")
    group = resp.json()
    print(f"Created group: {group['group_id']}")
    return group

def test_get_members(group_id):
    """Test getting group members."""
    print(f"\n=== Test 3: Get Group Members ({group_id}) ===")
    resp = requests.get(f"{BASE_URL}/api/v1/groups/{group_id}/members")
    print(f"Status: {resp.status_code}")
    members = resp.json()
    print(f"Members count: {len(members)}")
    for member in members:
        print(f"  - {member['agent_id']}: role={member['role']}, is_active={member['is_active']}")
    return members

def test_execute_consensus(group_id):
    """Test executing consensus."""
    print(f"\n=== Test 4: Execute Consensus ({group_id}) ===")
    
    payload = {
        "task": "What is the credit rating for a customer with 750 score and 5 years history?",
        "quorum_threshold": 0.5,
        "max_rounds": 2
    }
    
    resp = requests.post(
        f"{BASE_URL}/api/v1/groups/{group_id}/consensus",
        json=payload
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 201:
        result = resp.json()
        print(f"Consensus ID: {result['consensus_id']}")
        print(f"Success: {result['success']}")
        print(f"Consensus reached: {result['consensus_reached']}")
        print(f"Participating agents: {result['participating_agents']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        if result.get('final_solution'):
            print(f"Final answer: {result['final_solution']['answer']}")
    else:
        print(f"Error: {resp.text}")

def test_collaboration_mode(agents):
    """Test collaboration mode (no voting)."""
    print("\n=== Test 5: Create Group with Collaboration Mode ===")
    
    payload = {
        "group_name": "Collaboration Team",
        "description": "Testing collaboration mode",
        "mode": "collaboration",
        "created_by": "test_user",
        "initial_members": [
            {"agent_id": agents[0]["agent_id"], "role": "task_a"},
            {"agent_id": agents[1]["agent_id"], "role": "task_b"}
        ]
    }
    
    resp = requests.post(
        f"{BASE_URL}/api/v1/groups/",
        json=payload
    )
    print(f"Status: {resp.status_code}")
    group = resp.json()
    print(f"Created group: {group['group_id']} (mode={group['mode']})")
    
    # Execute collaboration
    print("\nExecuting collaboration task...")
    payload = {
        "task": "Analyze the transaction from different perspectives",
        "max_rounds": 1
    }
    
    resp = requests.post(
        f"{BASE_URL}/api/v1/groups/{group['group_id']}/consensus",
        json=payload
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 201:
        result = resp.json()
        print(f"Mode: {result['mode']}")
        print(f"Agent responses: {len(result['agent_responses'])}")
        print(f"Consensus reached: {result['consensus_reached']} (should be False for collaboration)")
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    print("Testing Group Chat Fixes")
    print("=" * 50)
    
    try:
        # Test 1: List agents
        agents = test_list_agents()
        
        if len(agents) < 2:
            print("ERROR: Need at least 2 agents to test")
            exit(1)
        
        # Test 2: Create group with initial members
        group = test_create_group_with_members(agents)
        
        # Test 3: Get members
        members = test_get_members(group["group_id"])
        
        # Test 4: Execute consensus
        test_execute_consensus(group["group_id"])
        
        # Test 5: Test collaboration mode
        test_collaboration_mode(agents)
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

