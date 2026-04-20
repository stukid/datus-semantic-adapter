from unittest.mock import patch

import pytest
from datus_semantic_core import semantic_adapter_registry
from datus_semantic_core.exceptions import SemanticCoreException

from datus_semantic_metricflow import MetricFlowAdapter, MetricFlowConfig, register


@pytest.fixture(autouse=True)
def reset_registry_state():
    adapters = semantic_adapter_registry._adapters.copy()
    factories = semantic_adapter_registry._factories.copy()
    metadata = semantic_adapter_registry._metadata.copy()
    initialized = semantic_adapter_registry._initialized

    semantic_adapter_registry._adapters.clear()
    semantic_adapter_registry._factories.clear()
    semantic_adapter_registry._metadata.clear()
    semantic_adapter_registry._initialized = False

    yield

    semantic_adapter_registry._adapters.clear()
    semantic_adapter_registry._adapters.update(adapters)
    semantic_adapter_registry._factories.clear()
    semantic_adapter_registry._factories.update(factories)
    semantic_adapter_registry._metadata.clear()
    semantic_adapter_registry._metadata.update(metadata)
    semantic_adapter_registry._initialized = initialized


class TestSemanticAdapterRegistry:
    def test_register_adds_metricflow_metadata(self):
        register()

        assert semantic_adapter_registry.is_registered("metricflow")
        assert semantic_adapter_registry.list_adapters()["metricflow"] is MetricFlowAdapter
        metadata = semantic_adapter_registry.get_metadata("metricflow")
        assert metadata is not None
        assert metadata.adapter_class is MetricFlowAdapter
        assert metadata.config_class is MetricFlowConfig
        assert metadata.display_name == "MetricFlow"

    def test_create_adapter_uses_registered_metricflow_adapter(self):
        register()
        config = MetricFlowConfig(namespace="test", semantic_models_path="/tmp/models")

        with patch.object(MetricFlowAdapter, "__init__", return_value=None) as mock_init:
            adapter = semantic_adapter_registry.create_adapter("metricflow", config)

        assert isinstance(adapter, MetricFlowAdapter)
        mock_init.assert_called_once_with(config)

    def test_create_adapter_unknown_type_raises_semantic_core_exception(self):
        with pytest.raises(SemanticCoreException, match="not found"):
            semantic_adapter_registry.create_adapter(
                "unknown_adapter", MetricFlowConfig(namespace="test", semantic_models_path="/tmp/models")
            )

    def test_get_metadata_unknown_returns_none(self):
        assert semantic_adapter_registry.get_metadata("unknown") is None
