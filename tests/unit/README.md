# Unit Tests

Unit tests for individual components.

## Running Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest tests/unit/ --cov=aegean.core

# Run with verbose output
pytest tests/unit/ -v
```

## Test Files

- `test_models.py` - Data model tests
- `test_agent.py` - Agent interface and registry tests
- `test_decision_engine.py` - Decision engine tests
- `test_coordinator.py` - Coordinator tests (TODO)
- `test_autogen_adapter.py` - AutoGen adapter tests (TODO)

## Test Coverage

Current coverage:
- ✅ Models: 100%
- ✅ Agent: 100%
- ✅ DecisionEngine: 100%
- ⏳ Coordinator: TODO
- ⏳ AutoGen Adapter: TODO

