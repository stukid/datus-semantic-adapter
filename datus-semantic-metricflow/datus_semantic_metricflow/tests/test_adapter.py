from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus_semantic_metricflow import MetricFlowAdapter, MetricFlowConfig
from datus_semantic_metricflow.models import DimensionInfo, MetricDefinition, QueryResult, ValidationIssue, ValidationResult


class _FakeColumns(list):
    def tolist(self):
        return list(self)


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows
        self.empty = not rows
        self.columns = _FakeColumns(list(rows[0].keys()) if rows else [])

    def to_dict(self, orient="records"):
        assert orient == "records"
        return list(self._rows)


def _validation_results(*, errors=None, warnings=None, has_blocking_issues=False):
    return SimpleNamespace(
        errors=list(errors or []),
        warnings=list(warnings or []),
        has_blocking_issues=has_blocking_issues,
    )


@pytest.fixture
def config():
    return MetricFlowConfig(namespace="test", timeout=300, semantic_models_path="/tmp/models")


@pytest.fixture
def adapter():
    instance = MetricFlowAdapter.__new__(MetricFlowAdapter)
    instance.service_type = "metricflow"
    instance.namespace = "test"
    instance.timeout = 300
    instance.client = MagicMock()
    instance._config_handler = MagicMock()
    return instance


class TestMetricFlowAdapter:
    def test_resolve_model_path_uses_semantic_models_path(self):
        config = MetricFlowConfig(namespace="analytics", semantic_models_path="/tmp/project/subject/semantic_models")

        result = MetricFlowAdapter._resolve_model_path(config)

        assert result.endswith("/tmp/project/subject/semantic_models")

    def test_init_uses_dict_config_handler_when_db_config_present(self):
        mock_handler = MagicMock()
        mock_handler.get_value.return_value = "datus_system"
        mock_sql_client = MagicMock()
        mock_user_model = MagicMock()
        mock_client = MagicMock()

        with (
            patch("datus_semantic_metricflow.adapter.build_config_dict_from_db_params", return_value={"k": "v"}),
            patch("datus_semantic_metricflow.adapter.DictConfigHandler", return_value=mock_handler) as mock_dict_handler,
            patch("datus_semantic_metricflow.adapter.MetricFlowClient", return_value=mock_client) as mock_client_cls,
            patch("metricflow.sql_clients.sql_utils.make_sql_client_from_config", return_value=mock_sql_client),
            patch("metricflow.engine.utils.build_user_configured_model_from_config", return_value=mock_user_model),
            patch("metricflow.configuration.constants.CONFIG_DWH_SCHEMA", "datus_system"),
            patch.object(MetricFlowAdapter, "_resolve_model_path", return_value="/tmp/models"),
        ):
            adapter = MetricFlowAdapter(
                MetricFlowConfig(
                    namespace="test",
                    db_config={"type": "duckdb", "database": "demo"},
                    semantic_models_path="/tmp/project/subject/semantic_models",
                )
            )

        mock_dict_handler.assert_called_once_with({"k": "v"})
        mock_client_cls.assert_called_once_with(
            sql_client=mock_sql_client,
            user_configured_model=mock_user_model,
            system_schema="datus_system",
        )
        assert adapter.client is mock_client

    def test_init_uses_file_config_handler_when_db_config_missing(self):
        mock_handler = MagicMock()
        mock_client = MagicMock()

        with (
            patch("datus_semantic_metricflow.adapter.DatusConfigHandler", return_value=mock_handler) as mock_handler_cls,
            patch("datus_semantic_metricflow.adapter.MetricFlowClient", return_value=mock_client),
            patch("metricflow.sql_clients.sql_utils.make_sql_client_from_config", return_value=MagicMock()),
            patch("metricflow.engine.utils.build_user_configured_model_from_config", return_value=MagicMock()),
            patch("metricflow.configuration.constants.CONFIG_DWH_SCHEMA", "datus_system"),
        ):
            adapter = MetricFlowAdapter(
                MetricFlowConfig(namespace="test", config_path="/tmp/agent.yml", semantic_models_path="/tmp/models")
            )

        mock_handler_cls.assert_called_once_with(namespace="test", config_path="/tmp/agent.yml")
        assert adapter.client is mock_client

    @pytest.mark.asyncio
    async def test_list_metrics_returns_metric_definitions(self, adapter):
        metric1 = SimpleNamespace(
            name="revenue",
            description="Total revenue",
            type="simple",
            input_measures=[SimpleNamespace(name="revenue_measure")],
        )
        metric2 = SimpleNamespace(
            name="orders",
            description="Order count",
            type="simple",
            input_measures=[SimpleNamespace(name="orders_measure")],
        )
        metric_semantics = MagicMock()
        metric_semantics.metric_references = ["revenue", "orders"]
        metric_semantics.get_metrics.return_value = [metric1, metric2]
        adapter.client.semantic_model.metric_semantics = metric_semantics
        adapter.client.engine.simple_dimensions_for_metrics.side_effect = [
            [SimpleNamespace(name="date"), SimpleNamespace(name="region")],
            [SimpleNamespace(name="date")],
        ]

        metrics = await adapter.list_metrics(limit=1, offset=1)

        assert metrics == [
            MetricDefinition(
                name="orders",
                description="Order count",
                type="simple",
                dimensions=["date"],
                measures=["orders_measure"],
                metadata={},
            )
        ]

    @pytest.mark.asyncio
    async def test_get_dimensions_returns_dimension_info(self, adapter):
        adapter.client.list_dimensions.return_value = [
            SimpleNamespace(name="date", description="Calendar date"),
            SimpleNamespace(name="region", description="Sales region"),
        ]

        dimensions = await adapter.get_dimensions("revenue")

        assert dimensions == [
            DimensionInfo(name="date", description="Calendar date"),
            DimensionInfo(name="region", description="Sales region"),
        ]
        adapter.client.list_dimensions.assert_called_once_with(metric_names=["revenue"])

    @pytest.mark.asyncio
    async def test_query_metrics_returns_rows_as_dicts(self, adapter):
        adapter.client.query.return_value = SimpleNamespace(
            result_df=_FakeDataFrame(
                [
                    {"date": "2024-01-01", "revenue": 1000},
                    {"date": "2024-01-02", "revenue": 1200},
                ]
            ),
            dataflow_plan="mock-plan",
        )

        result = await adapter.query_metrics(metrics=["revenue"], dimensions=["date"], limit=10)

        assert result == QueryResult(
            columns=["date", "revenue"],
            data=[
                {"date": "2024-01-01", "revenue": 1000},
                {"date": "2024-01-02", "revenue": 1200},
            ],
            metadata={"dataflow_plan": "mock-plan"},
        )
        adapter.client.query.assert_called_once_with(
            metrics=["revenue"],
            dimensions=["date"],
            start_time=None,
            end_time=None,
            where=None,
            limit=10,
            order=None,
        )

    @pytest.mark.asyncio
    async def test_query_metrics_adds_metric_time_dimension_for_granularity(self, adapter):
        adapter.client.query.return_value = SimpleNamespace(result_df=_FakeDataFrame([]), dataflow_plan=None)

        await adapter.query_metrics(
            metrics=["revenue"],
            dimensions=["region"],
            time_granularity="month",
            order_by=["-revenue", "null"],
        )

        adapter.client.query.assert_called_once_with(
            metrics=["revenue"],
            dimensions=["region", "metric_time__month"],
            start_time=None,
            end_time=None,
            where=None,
            limit=None,
            order=["-revenue"],
        )

    @pytest.mark.asyncio
    async def test_query_metrics_dry_run_returns_sql(self, adapter):
        adapter.client.explain.return_value = SimpleNamespace(
            rendered_sql_without_descriptions=SimpleNamespace(sql_query="SELECT 1")
        )

        result = await adapter.query_metrics(metrics=["revenue"], dimensions=["date"], dry_run=True)

        assert result == QueryResult(
            columns=["sql"],
            data=[{"sql": "SELECT 1"}],
            metadata={"explain": True, "sql": "SELECT 1"},
        )
        adapter.client.explain.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_semantic_returns_valid_result(self, adapter):
        adapter.client = SimpleNamespace(sql_client=MagicMock(), system_schema="datus_system")
        lint_results = _validation_results()
        parsing_issues = _validation_results()
        semantic_issues = _validation_results()

        with (
            patch("metricflow.engine.utils.path_to_models", return_value="/tmp/models"),
            patch("metricflow.model.parsing.config_linter.ConfigLinter") as mock_linter_cls,
            patch(
                "metricflow.engine.utils.model_build_result_from_config",
                return_value=SimpleNamespace(issues=parsing_issues, model="user-model"),
            ),
            patch("metricflow.model.model_validator.ModelValidator") as mock_validator_cls,
            patch("metricflow.model.data_warehouse_model_validator.DataWarehouseModelValidator"),
            patch.object(adapter, "_run_dw_validations", return_value=_validation_results()),
        ):
            mock_linter_cls.return_value.lint_dir.return_value = lint_results
            mock_validator_cls.return_value.validate_model.return_value = SimpleNamespace(issues=semantic_issues)

            result = await adapter.validate_semantic()

        assert result == ValidationResult(valid=True, issues=[])

    @pytest.mark.asyncio
    async def test_validate_semantic_returns_errors_from_lint_stage(self, adapter):
        lint_results = _validation_results(errors=["bad lint"], has_blocking_issues=True)

        with (
            patch("metricflow.engine.utils.path_to_models", return_value="/tmp/models"),
            patch("metricflow.model.parsing.config_linter.ConfigLinter") as mock_linter_cls,
        ):
            mock_linter_cls.return_value.lint_dir.return_value = lint_results

            result = await adapter.validate_semantic()

        assert result.valid is False
        assert result.issues == [ValidationIssue(severity="error", message="bad lint")]

    def test_convert_validation_results_maps_errors_and_warnings(self, adapter):
        results = _validation_results(errors=["bad metric"], warnings=["deprecated field"])

        converted = adapter._convert_validation_results(results)

        assert converted == [
            ValidationIssue(severity="error", message="bad metric"),
            ValidationIssue(severity="warning", message="deprecated field"),
        ]

    def test_run_dw_validations_uses_adapter_timeout(self, adapter):
        adapter.timeout = 42
        dw_validator = MagicMock()

        with patch("metricflow.model.validations.validator_helpers.ModelValidationResults.merge", return_value="merged"):
            merged = adapter._run_dw_validations(dw_validator, model="user-model")

        assert merged == "merged"
        dw_validator.validate_data_sources.assert_called_once_with("user-model", 42)
        dw_validator.validate_dimensions.assert_called_once_with("user-model", 42)
        dw_validator.validate_identifiers.assert_called_once_with("user-model", 42)
        dw_validator.validate_measures.assert_called_once_with("user-model", 42)
        dw_validator.validate_metrics.assert_called_once_with("user-model", 42)


class TestConfiguration:
    def test_config_defaults(self):
        config = MetricFlowConfig(namespace="test", semantic_models_path="/tmp/models")

        assert config.namespace == "test"
        assert config.service_type == "metricflow"
        assert config.config_path is None
        assert config.timeout == 300
        assert config.db_config is None
        assert config.semantic_models_path == "/tmp/models"

    def test_config_requires_semantic_models_path(self):
        with pytest.raises(Exception):
            MetricFlowConfig(namespace="test")

    def test_config_custom_values(self):
        config = MetricFlowConfig(
            namespace="prod",
            config_path="/tmp/agent.yml",
            timeout=600,
            db_config={"type": "postgres", "database": "analytics"},
            semantic_models_path="/tmp/project/subject/semantic_models",
        )

        assert config.namespace == "prod"
        assert config.config_path == "/tmp/agent.yml"
        assert config.timeout == 600
        assert config.db_config == {"type": "postgres", "database": "analytics"}
        assert config.semantic_models_path == "/tmp/project/subject/semantic_models"
