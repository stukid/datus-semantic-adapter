from typing import Any, Dict, List, Optional

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.utils.loggings import get_logger
from datus_semantic_metricflow.config import MetricFlowConfig
from datus_semantic_metricflow.models import (
    DimensionInfo,
    MetricDefinition,
    MetricType,
    QueryResult,
    ValidationIssue,
    ValidationResult,
)

# Import MetricFlow API
from metricflow.api.metricflow_client import MetricFlowClient
from metricflow.configuration.datus_config_handler import DatusConfigHandler

logger = get_logger(__name__)


class MetricFlowAdapter(BaseSemanticAdapter):
    """
    MetricFlow semantic layer adapter.

    Integrates with MetricFlow CLI to provide metric querying capabilities.
    """

    def __init__(self, config: MetricFlowConfig):
        super().__init__(config, service_type="metricflow")
        self.namespace = config.namespace
        self.timeout = config.timeout
        self._client: Optional[MetricFlowClient] = None

        logger.info(f"Initializing MetricFlowAdapter for namespace: {self.namespace}")

        # Only initialize config handler, defer client creation to lazy load
        config_path = getattr(config, 'config_path', None)
        self._config_handler = DatusConfigHandler(namespace=self.namespace, config_path=config_path)
        logger.info("MetricFlowAdapter initialized (client will be created on first use)")

    @property
    def client(self) -> MetricFlowClient:
        """Lazy load MetricFlowClient when first accessed."""
        if self._client is None:
            try:
                from metricflow.configuration.constants import CONFIG_DWH_SCHEMA
                from metricflow.engine.utils import build_user_configured_model_from_config
                from metricflow.sql_clients.sql_utils import make_sql_client_from_config

                # Build client components using the config handler
                sql_client = make_sql_client_from_config(self._config_handler)
                user_configured_model = build_user_configured_model_from_config(self._config_handler)
                schema = self._config_handler.get_value(CONFIG_DWH_SCHEMA)

                # Construct MetricFlowClient directly
                self._client = MetricFlowClient(
                    sql_client=sql_client,
                    user_configured_model=user_configured_model,
                    system_schema=schema,
                )
                logger.info("MetricFlowClient initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize MetricFlowClient: {e}", exc_info=True)
                raise

        return self._client

    # Semantic Model Interface

    def get_semantic_model(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> Optional[Dict[str, Any]]:
        """MetricFlow doesn't directly expose semantic models."""
        return None

    def list_semantic_models(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[str]:
        """MetricFlow uses semantic models internally."""
        return []

    # Metrics Interface

    async def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MetricDefinition]:
        """
        List available metrics using MetricFlow client.

        Args:
            path: Optional subject area filter
            limit: Maximum metrics to return
            offset: Number to skip

        Returns:
            List of metric definitions
        """
        # Get full metric objects directly from semantic_model
        metric_semantics = self.client.semantic_model.metric_semantics
        metric_refs = metric_semantics.metric_references
        full_metrics = metric_semantics.get_metrics(metric_refs)

        # Convert to MetricDefinition list
        metrics = []
        for metric in full_metrics:
            # Get dimensions for this metric
            dimensions = self.client.engine.simple_dimensions_for_metrics([metric.name])
            metrics.append(MetricDefinition(
                name=metric.name,
                description=metric.description,
                type=metric.type,
                dimensions=[d.name for d in dimensions],
                measures=[m.name for m in metric.input_measures],
                metadata={},
            ))

        if path:
            metrics = [m for m in metrics if m.path and m.path[: len(path)] == path]

        return metrics[offset : offset + limit]

    async def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> List[DimensionInfo]:
        """
        Get dimensions for a metric using MetricFlow client.

        Args:
            metric_name: Name of the metric
            path: Optional subject area filter

        Returns:
            List of DimensionInfo objects containing name and description
        """
        # Get dimensions from client (returns List[Dimension])
        dimensions = self.client.list_dimensions(metric_names=[metric_name])

        # Convert to DimensionInfo objects
        return [DimensionInfo(name=d.name, description=d.description) for d in dimensions]

    async def query_metrics(
        self,
        metrics: List[str],
        dimensions: List[str] = [],
        path: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_granularity: Optional[str] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> QueryResult:
        """
        Query metrics using MetricFlow client.

        Args:
            metrics: List of metric names
            dimensions: List of dimensions to group by
            path: Optional subject area filter
            time_start: Start time (ISO format like '2024-01-01' or relative like '-7d')
            time_end: End time (ISO format like '2024-01-31' or relative like 'now')
            time_granularity: Time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: Optional WHERE clause
            limit: Result limit
            order_by: Columns to order by
            dry_run: If True, explain query instead of executing

        Returns:
            Query result
        """
        # Prepare query parameters
        start_time = time_start
        end_time = time_end

        if dry_run:
            # Use explain to get SQL without executing
            result = self.client.explain(
                metrics=metrics,
                dimensions=dimensions,
                start_time=start_time,
                end_time=end_time,
                where=where,
                limit=limit,
                order=order_by,
            )
            # Return SQL as result
            sql = result.rendered_sql_without_descriptions.sql_query
            return QueryResult(
                columns=["sql"],
                data=[{"sql": sql}],
                metadata={"explain": True, "sql": sql},
            )
        else:
            # Execute the query
            result = self.client.query(
                metrics=metrics,
                dimensions=dimensions,
                start_time=start_time,
                end_time=end_time,
                where=where,
                limit=limit,
                order=order_by,
            )

            # Convert DataFrame to QueryResult
            if result.result_df is not None and not result.result_df.empty:
                columns = result.result_df.columns.tolist()
                # Convert to list of dicts (QueryResult.data expects List[Dict[str, Any]])
                data = result.result_df.to_dict(orient="records")
                return QueryResult(
                    columns=columns,
                    data=data,
                    metadata={"dataflow_plan": result.dataflow_plan},
                )
            else:
                return QueryResult(columns=[], data=[])

    async def validate_semantic(self) -> ValidationResult:
        """
        Validate MetricFlow configuration using full validation pipeline.

        This performs the same validations as 'mf validate-configs':
        1. Lint validation (YAML format)
        2. Parsing validation (model building)
        3. Semantic validation (model semantics)
        4. Data warehouse validation

        Returns:
            Validation result
        """
        from metricflow.engine.utils import path_to_models, model_build_result_from_config
        from metricflow.model.model_validator import ModelValidator
        from metricflow.model.parsing.config_linter import ConfigLinter
        from metricflow.model.data_warehouse_model_validator import DataWarehouseModelValidator

        all_issues: List[ValidationIssue] = []

        # Step 1: Lint Validation
        try:
            model_path = path_to_models(handler=self._config_handler)
            lint_results = ConfigLinter().lint_dir(model_path)
            all_issues.extend(self._convert_validation_results(lint_results))
            if lint_results.has_blocking_issues:
                return ValidationResult(valid=False, issues=all_issues)
        except Exception as e:
            logger.error(f"Lint validation failed: {e}")
            all_issues.append(ValidationIssue(severity="error", message=f"Lint validation failed: {e}"))
            return ValidationResult(valid=False, issues=all_issues)

        # Step 2: Parsing Validation
        try:
            parsing_result = model_build_result_from_config(
                handler=self._config_handler, raise_issues_as_exceptions=False
            )
            all_issues.extend(self._convert_validation_results(parsing_result.issues))
            if parsing_result.issues.has_blocking_issues:
                return ValidationResult(valid=False, issues=all_issues)
            user_model = parsing_result.model
        except Exception as e:
            logger.error(f"Parsing validation failed: {e}")
            all_issues.append(ValidationIssue(severity="error", message=f"Parsing validation failed: {e}"))
            return ValidationResult(valid=False, issues=all_issues)

        # Step 3: Semantic Validation
        try:
            semantic_result = ModelValidator().validate_model(user_model)
            all_issues.extend(self._convert_validation_results(semantic_result.issues))
            if semantic_result.issues.has_blocking_issues:
                return ValidationResult(valid=False, issues=all_issues)
        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            all_issues.append(ValidationIssue(severity="error", message=f"Semantic validation failed: {e}"))
            return ValidationResult(valid=False, issues=all_issues)

        # Step 4: Data Warehouse Validation
        try:
            dw_validator = DataWarehouseModelValidator(
                sql_client=self.client.sql_client,
                system_schema=self.client.system_schema,
            )
            dw_results = self._run_dw_validations(dw_validator, user_model)
            all_issues.extend(self._convert_validation_results(dw_results))
        except Exception as e:
            logger.error(f"Data warehouse validation failed: {e}")
            all_issues.append(ValidationIssue(severity="error", message=f"Data warehouse validation failed: {e}"))

        has_errors = any(issue.severity == "error" for issue in all_issues)
        return ValidationResult(valid=not has_errors, issues=all_issues)

    def _run_dw_validations(self, dw_validator, model):
        """Run all data warehouse validations and merge results."""
        from metricflow.model.validations.validator_helpers import ModelValidationResults

        timeout = self.timeout
        data_source_results = dw_validator.validate_data_sources(model, timeout)
        dimension_results = dw_validator.validate_dimensions(model, timeout)
        identifier_results = dw_validator.validate_identifiers(model, timeout)
        measure_results = dw_validator.validate_measures(model, timeout)
        metric_results = dw_validator.validate_metrics(model, timeout)

        return ModelValidationResults.merge([
            data_source_results,
            dimension_results,
            identifier_results,
            measure_results,
            metric_results,
        ])

    def _convert_validation_results(self, results) -> List[ValidationIssue]:
        """Convert ModelValidationResults to list of ValidationIssue."""
        issues = []
        for error in results.errors:
            issues.append(ValidationIssue(
                severity="error",
                message=str(error),
            ))
        for warning in results.warnings:
            issues.append(ValidationIssue(
                severity="warning",
                message=str(warning),
            ))
        return issues

