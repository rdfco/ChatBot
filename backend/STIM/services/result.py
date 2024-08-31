import logging
from datetime import datetime
from uuid import UUID

from fastapi import Depends
from STIM.errors import ValidationError
from STIM.models.llm_flow.schema import (
    ChartGenerationResultContent,
    SQLQueryStringResultContent,
)
from STIM.models.result.schema import ChartRefreshOut, ResultUpdate
from STIM.repositories.base import AsyncSession, NotFoundError
from STIM.repositories.result import ResultRepository
from STIM.services.llm_flow.llm_calls.chart_generator import ChartType
from STIM.services.llm_flow.toolkit import (
    RunException,
    execute_sql_query,
    query_run_result_to_chart_json,
)
from STIM.services.llm_flow.utils import STIMSQLDatabase as SQLDatabase

logger = logging.getLogger(__name__)


class ResultService:
    result_repo: ResultRepository

    def __init__(self, result_repo: ResultRepository = Depends(ResultRepository)) -> None:
        self.result_repo = result_repo

    async def update_sql_query_result_content(
        self, session: AsyncSession, result_id: UUID, sql: str, for_chart: bool
    ) -> ChartRefreshOut | None:
        # Need to validate the SQL run output to ensure it's compatible with the linked chart
        if for_chart:
            try:
                chart_id = await self._validate_chart_sql(session, result_id, sql)
                await self._update_sql(session, result_id, sql)
                return await self.refresh_chart_result_data(session, chart_id)
            except NotFoundError:
                # TODO: Deal with faulty chart generation in a better way
                # Chart generation probably failed, let's let it slide
                pass
        else:
            # Just update sql, no chart involved
            await self._update_sql(session, result_id, sql)

    async def _validate_chart_sql(
        self,
        session: AsyncSession,
        sql_query_string_id: UUID,
        sql: str,
    ) -> UUID:
        linked_chart = await self.result_repo.get_chart_from_sql_query(session, sql_query_string_id)
        chart_content = ChartGenerationResultContent.model_validate_json(linked_chart.content)
        await self.validate_sql_query_result_for_chart(
            session, sql_query_string_id, sql, ChartType[chart_content.chart_type]
        )
        return linked_chart.id

    async def _update_sql(self, session, result_id: UUID, sql: str):
        query_string_result = await self.result_repo.get_by_uuid(session, result_id)

        # Parse json and update content sql (Do not want to ever update for_chart)
        new_content = SQLQueryStringResultContent.model_validate_json(query_string_result.content)
        new_content.sql = sql

        # Dump json and update stored model
        content_dumps = new_content.model_dump_json()
        await self.result_repo.update_by_uuid(session, result_id, ResultUpdate(content=content_dumps))

    async def refresh_chart_result_data(self, session: AsyncSession, chart_id: UUID) -> ChartRefreshOut:
        chart_result = await self.result_repo.get_by_uuid(session, chart_id)
        chart_content = ChartGenerationResultContent.model_validate_json(chart_result.content)
        chart_type = ChartType[chart_content.chart_type]

        # Get linked SQL string result
        if not chart_result.linked_id:
            raise ValueError("Attempting to refresh a chart result without a linked_id")
        sql_query_string_result = await self.result_repo.get_by_uuid(session, chart_result.linked_id)
        sql_string = SQLQueryStringResultContent.model_validate_json(sql_query_string_result.content).sql

        # Get DSN from linked connection
        dsn = await self.result_repo.get_dsn_from_result(session, chart_id)
        db = SQLDatabase.from_uri(dsn)

        # Refresh chart data
        query_run_data = execute_sql_query(db, sql_string, for_chart=True, chart_type=chart_type)
        updated_chartjs_json = query_run_result_to_chart_json(chart_content.chartjs_json, chart_type, query_run_data)

        # Store updated chart result
        updated_content = ChartGenerationResultContent(
            chartjs_json=updated_chartjs_json, chart_type=chart_content.chart_type
        )
        updated_date = datetime.now()
        await self.result_repo.update_by_uuid(
            session, chart_id, ResultUpdate(created_at=updated_date, content=updated_content.model_dump_json())
        )

        return ChartRefreshOut(chartjs_json=updated_chartjs_json, created_at=updated_date)

    async def validate_sql_query_result_for_chart(
        self, session: AsyncSession, result_id: UUID, sql: str, chart_type: ChartType
    ) -> None:
        # Get DSN from linked connection
        dsn = await self.result_repo.get_dsn_from_result(session, result_id)
        db = SQLDatabase.from_uri(dsn)

        # Run query to ensure it's compatible with the linked chart
        try:
            execute_sql_query(db, sql, for_chart=True, chart_type=chart_type)
        except RunException:
            # TODO: Modify this based on chart type
            raise ValidationError(
                "New SQL query is not compatible with chart! "
                "Make sure to specify 2 columns, first for labels and second for values."
            )
