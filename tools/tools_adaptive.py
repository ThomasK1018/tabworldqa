from __future__ import annotations
from enum import Enum
from typing import Literal, Dict, Optional, List, Union, Sequence
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, validator
import pandas as pd
import re
import json


# ===== EXISTING TOOLS =====

class SelectColumnInput(BaseModel):
    """Schema for selecting subset of columns."""
    
    column_names: Union[str, List[str]] = Field(..., description="List of column names to select")
    reasoning: str = Field(..., description="Your reasoning")

    
class SelectColumnTool(BaseTool):
    """Tool to select a subset of columns relevant to the question."""
    
    name: str = "f_select_column"
    description: str = "Select a subset of columns."
    response_format: Literal["text"] = "text"
    args_schema: type[SelectColumnInput] = SelectColumnInput
    
    def _run(self, column_names: Union[str, List[str]], reasoning: str, **kwargs) -> str:
        return f"""
_cols = str_or_list("{column_names}")
_missing = [c for c in _cols if c not in df.columns]
if _missing:
    print(f"❌ Columns not found: {{_missing}}. Available: {{df.columns.tolist()}}")
else:
    df = f_select_column(df, _cols, "{reasoning}")
"""
    
    async def _arun(self, column_names: Union[str, List[str]], reasoning: str, **kwargs) -> str:
        return self._run(column_names, reasoning, **kwargs)


class NewPrintTableInput(BaseModel):
    """Schema for printing the current table state."""
    max_rows: int = Field(10, description="Maximum number of rows to display")

class NewPrintTableTool(BaseTool):
    """Tool to print all rows of the current table state."""
    
    name: str = "print_table"
    description: str = "Print the table."
    response_format: Literal["text"] = "text"
    args_schema: type[NewPrintTableInput] = NewPrintTableInput
    
    def _run(self, max_rows: int = 20, **kwargs) -> str:
        return f"""
f_print_table(df, {max_rows})
"""
    
    async def _arun(self, max_rows: int = 20, **kwargs) -> str:
        return self._run(max_rows, **kwargs)


# ===== NEW TOOLS - EASY TO ADD! =====

class CalculateAverageInput(BaseModel):
    """Schema for calculating average of a column by groups."""
    column_name: str = Field(..., description="Column name to calculate average for")
    group_by_column: str = Field(..., description="Column to group by before calculating average")
    filter_column: Optional[str] = Field(None, description="Optional column to filter on before aggregation")
    filter_value: Optional[Union[str, int, float]] = Field(None, description="Value to compare in filter_column")
    filter_op: str = Field("==", description="Comparison operator for filter_column (==, !=, >, <, >=, <=)")
        
class CalculateAverageTool(BaseTool):
    """Tool to calculate average of a column grouped by another column."""
    name: str = "f_calculate_average"
    description: str = "Calculate average of a column in groupby."
    response_format: Literal["text"] = "text"
    args_schema: type[CalculateAverageInput] = CalculateAverageInput
    
    def _run(self, column_name: str, group_by_column: str, filter_column: Optional[str] = None, filter_value: Optional[Union[str, int, float]] = None, filter_op: str = "==", **kwargs) -> str:
        filter_col_literal = f'"{filter_column}"' if filter_column else "None"
        filter_val_literal = json.dumps(filter_value) if filter_value is not None else "None"
        return f"""
df = f_calculate_average(df, "{column_name}", "{group_by_column}", filter_column={filter_col_literal}, filter_value={filter_val_literal}, filter_op="{filter_op}")
"""

    async def _arun(self, column_name: str, group_by_column: str, filter_column: Optional[str] = None, filter_value: Optional[Union[str, int, float]] = None, filter_op: str = "==", **kwargs) -> str:
        return self._run(column_name, group_by_column, filter_column, filter_value, filter_op, **kwargs)


class NewGetDataInfoInput(BaseModel):
    """Schema for getting basic information about the dataset."""
    show_missing: bool = Field(True, description="Whether to show missing value counts")
    

class NewGetDataInfoTool(BaseTool):
    """Tool to get basic information about the dataset."""
    
    name: str = "f_get_data_info"
    description: str = "Dataset info, i.e. shape, columns, dtypes, and missing values."
    response_format: Literal["text"] = "text"
    args_schema: type[NewGetDataInfoInput] = NewGetDataInfoInput
    
    def _run(self, show_missing: bool = True, **kwargs) -> str:
        return f"""
f_get_data_info(df, {show_missing})    
"""
    
    async def _arun(self, show_missing: bool = True, **kwargs) -> str:
        return self._run(show_missing, **kwargs)

# ===== ADDITIONAL TOOLS FROM ENHANCED_COT_TOOLS =====

class SelectRowInput(BaseModel):
    """Schema for selecting rows by indices or conditions."""
    indices: list[int] | None = Field(None, description="Explicit row indices to keep (0-based)")
    condition: Optional[Union[str, List[str]]] = Field(None, description="Row selection conditions")
    lookup: bool = Field(True, description="Whether to only lookup info without transformation (True) or truly filter dataframe (False)")
    reasoning: str = Field(..., description="Explanation of why this row selection is needed")


class SelectRowTool(BaseTool):
    """Tool to select rows by indices or conditions."""
    name: str = "f_select_row"
    description: str = "Select specific rows of table."
    response_format: Literal["text"] = "text"
    args_schema: type[SelectRowInput] = SelectRowInput
    
    def _run(self, indices: Optional[List[int]] = None,
            condition: Optional[Union[str, List[str]]] = None,
            lookup: bool = False,
            reasoning: str = "",
              **kwargs) -> str:
        cond_literal = json.dumps(condition) if condition is not None else "None"
        reasoning_literal = json.dumps(reasoning)
        return f"""
cond_val = {cond_literal}
if {lookup}:
    f_select_row(df, {indices}, str_or_list(cond_val), {lookup}, {reasoning_literal})
else:
    df = f_select_row(df, {indices}, str_or_list(cond_val), {lookup}, {reasoning_literal})
"""
    
    async def _arun(self, indices: Optional[List[int]] = None,
                    condition: Optional[Union[str, List[str]]] = None,
                    lookup: bool = False,
                    reasoning: str = "", **kwargs) -> str:
        return self._run(indices, condition, lookup, reasoning, **kwargs)


class SortByInput(BaseModel):
    """Schema for sorting table by columns."""
    
    columns: Union[str, List[str]] = Field(..., description="Column names to sort by (in order of priority)")
    order: str = Field("asc", description="Sort order: 'asc' for ascending, 'desc' for descending")

class SortByTool(BaseTool):
    """Tool to sort table lexicographically by columns."""
    
    name: str = "f_sort_by"
    description: str = "Sort table by column(s)."
    response_format: Literal["text"] = "text"
    args_schema: type[SortByInput] = SortByInput
    
    def _run(self, columns: Union[str, List[str]], order: str = "asc", **kwargs) -> str:
        return f"""
df = f_sort_by(df,  str_or_list("{columns}"), "{order}")
"""
    
    async def _arun(self, columns: Union[str, List[str]], order: str = "asc", **kwargs) -> str:
        return self._run(columns, order, **kwargs)


class AggregateInput_v0(BaseModel):
    """Schema for computing aggregations."""
    column: str = Field(..., description="Column to aggregate")
    op: str = Field(..., description="Aggregation operation: count, sum, avg, min, max")
    show_stats_only: bool = Field(default=..., description="Show statistics only without changing the DF")
    group_by: Optional[Union[str, List[str]]] = Field(None, description="Optional list of columns to group by before aggregating")
    
            
class AggregateTool_v0(BaseTool):
    """Tool to compute aggregations over columns."""
    name: str = "f_aggregate"
    description: str = "Aggregation (count, sum, avg, min, max) of a column."
    response_format: Literal["text"] = "text"
    args_schema: type[AggregateInput_v0] = AggregateInput_v0

    def _run(self, column: str, op: str, show_stats_only: bool, group_by: Optional[Union[str, List[str]]] = None, **kwargs) -> str:
        return f"""
if {show_stats_only} == True:
    f_aggregate(df, "{column}", "{op}", {show_stats_only}, str_or_list("{group_by}"))
else:
    df = f_aggregate(df, "{column}","{op}", {show_stats_only}, str_or_list("{group_by}"))
"""

    async def _arun(self, column: str, op: str, show_stats_only: bool, group_by: Optional[Union[str, List[str]]] = None, **kwargs) -> str:
        return self._run(column, op, show_stats_only, group_by, **kwargs)

class AggregateInput(BaseModel):
    """Schema for computing aggregations."""
    # Accept either single col + op OR dict/list
    column: Optional[str] = Field(None, description="Single column for aggregation (legacy mode)")
    op: Optional[str] = Field(None, description="Operation for single column (legacy mode)")
    columns_ops: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = Field(
        None, description="Mapping of {column: operation} or list of mappings"
    )
    show_stats_only: bool = Field(default=False, description="Show statistics only without changing the DF")
    group_by: Optional[Union[str, List[str]]] = Field(None, description="Group by columns")

class AggregateTool(BaseTool):
    name: str = "f_aggregate"
    description: str = "Aggregation (count, sum, avg, min, max, diff) of columns."
    response_format: Literal["text"] = "text"
    args_schema: type[AggregateInput] = AggregateInput

    def _run(self,
             column: Optional[str] = None,
             op: Optional[str] = None,
             columns_ops: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
             show_stats_only: bool = False,
             group_by: Optional[Union[str, List[str]]] = None,
             **kwargs) -> str:

        if column and op:  # legacy single-column mode
            return f"""
df = f_aggregate(df, "{column}", op="{op}", show_stats_only={show_stats_only}, group_by=str_or_list("{group_by}"))
"""
        else:  # dict/list mode
            return f"""
df = f_aggregate(df, {columns_ops}, show_stats_only={show_stats_only}, group_by=str_or_list("{group_by}"))
"""

class ComputeColumnInput_v0(BaseModel):
    """Schema for computing new columns."""
    
    new_col: str = Field(..., description="Name of the new column to create")
    col_A: str = Field(..., description="Left operand column name")
    op: str = Field(..., description="Operation: add, sub, mul, div, ratio")
    col_B: str | int | float = Field(..., description="Right operand (column name or scalar value)")
    element_wise: bool = Field(..., description="Whether to apply operation element-wise over two columns (True) or on a single value (False)")



class ComputeColumnTool_v0(BaseTool):
    """Tool to create new columns with binary operations."""
    
    name: str = "f_compute_column"
    description: str = "Create new column by applying binary operations onto an existing column. If used to calculate time difference, will return day difference. "
    response_format: Literal["text"] = "text"
    args_schema: type[ComputeColumnInput_v0] = ComputeColumnInput_v0
    
    def _run(self, new_col: str, col_A: str, op: str, col_B: str | int | float, element_wise: bool, **kwargs) -> str:
        
        
        return f"""
df = f_compute_column(df, "{new_col}", "{col_A}", "{op}", {repr(col_B)}, {element_wise})
"""
    
    async def _arun(self, new_col: str, col_A: str, op: str, col_B: str | int | float, element_wise: bool, **kwargs) -> str:
        return self._run(new_col, col_A, op, col_B, element_wise, **kwargs)


class ComputeColumnInput(BaseModel):
    """Schema for computing new columns."""
    new_col: str = Field(..., description="Name of the new column to create")
    col_A: str = Field(..., description="Left operand column name")
    op: str = Field(..., description="Operation: add, sub, mul, div, ratio, or custom")
    col_B: str | int | float = Field(..., description="Right operand (column name or scalar value)")
    element_wise: bool = Field(..., description="Apply operation element-wise (True) or aggregate (False)")
    special_op: Optional[str] = Field(
        default=None,
        description="Special operation like 'time_diff' for group-based diffs"
    )

class ComputeColumnTool(BaseTool):
    name: str = "f_compute_column"
    description: str = "Create new columns with binary operations or special ops (e.g., time difference)."
    response_format: Literal["text"] = "text"
    args_schema: type[ComputeColumnInput] = ComputeColumnInput
    
    def _run(self, new_col: str, col_A: str, op: str,
             col_B: str | int | float, element_wise: bool,
             special_op: Optional[str] = None, **kwargs) -> str:
        return f"""
df = f_compute_column(df, "{new_col}", "{col_A}", "{op}", {repr(col_B)}, {element_wise}, special_op="{special_op}")
"""

class DistinctCountInput(BaseModel):
    """Schema for counting distinct values."""
    
    column: str = Field(..., description="Column name to count distinct values for")


class DistinctCountTool(BaseTool):
    """Tool to count distinct values in a column."""
    
    name: str = "f_distinct_count"
    description: str = "Number of unique values in a column."
    response_format: Literal["text"] = "text"
    args_schema: type[DistinctCountInput] = DistinctCountInput
    
    def _run(self, column: str, **kwargs) -> str:
        return f"""
f_distinct_count(df, "{column}")
"""
    
    async def _arun(self, column: str, **kwargs) -> str:
        return self._run(column, **kwargs)
    
class DistinctCountInput_v2(BaseModel):
    """Schema for counting distinct values in one or more columns."""
    columns: Union[str, List[str]] = Field(..., description="Column name(s) to count distinct values for")


class DistinctCountTool_v2(BaseTool):
    """Tool to count distinct values in one or more columns."""

    name: str = "f_distinct_count"
    description: str = "Number of unique values in one or more columns."
    response_format: Literal["text"] = "text"
    args_schema: type[DistinctCountInput_v2] = DistinctCountInput_v2

    def _run(self, columns: Union[str, List[str]], **kwargs) -> str:
        return f"""
f_distinct_count_v2(df, {columns!r})
"""

    async def _arun(self, columns: Union[str, List[str]], **kwargs) -> str:
        return self._run(columns, **kwargs)


class FinalAnswerInput(BaseModel):
    """Schema for submitting the final answer."""
    answer: str = Field(..., description="The final answer to the user question")


class FinalAnswerTool(BaseTool):
    """Tool to deliver the final answer and end the analysis."""
    name: str = "f_final_answer"
    description: str = "Submit the final answer."
    response_format: Literal["text"] = "text"
    args_schema: type[FinalAnswerInput] = FinalAnswerInput

    def _run(self, answer: str, **kwargs) -> str:
        return f"FINAL ANSWER: {answer}"

    async def _arun(self, answer: str, **kwargs) -> str:
        return self._run(answer, **kwargs)

    

    
class StringOperationInput(BaseModel):
    """Schema for string operations on a DataFrame column."""
    columns: Union[str, list[str]] = Field(
        ...,
        description="Name of the string column to operate on.",
        example="customer_name"
    )
    
    operation: Literal[
        "lowercase",
        "uppercase",
        "titlecase",
        "substring",
        "concat",
        "replace",
        "type_change"
    ] = Field(
        ...,
        description="Type of string operation to apply.",
        example="substring"
    )

    start: Optional[int] = Field(
        default=None,
        description="Start index for substring extraction.",
        example=0
    )
    end: Optional[int] = Field(
        default=None,
        description="End index for substring extraction.",
        example=5
    )

    other_columns: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Second column name for concatenation.",
        example="last_name"
    )
    separator: str = Field(
        default=" ",
        description="Separator to use when concatenating columns.",
        example="_"
    )

    find: Optional[str] = Field(
        default=None,
        description="Substring or regex pattern to find for replacement.",
        example="Ltd"
    )
    replace_with: Optional[str] = Field(
        default=None,
        description="Replacement string for matched substrings.",
        example="Limited"
    )

    new_columns: Union[str, List[str]] = Field(
        default=None,
        description="Optional name for the new column. If not provided, '_mod' is appended.",
        example="name_clean"
    )

    reasoning: str = Field(
        ...,
        description="Explanation of why this string operation is needed.",
        example="Standardize company names to lowercase."
    )


class StringOperationTool(BaseTool):
    name: str = "f_string_operation"
    description: str = """Handle strings column in dataframe."""
    args_schema: type[StringOperationInput] = StringOperationInput

    def _run(
        self,
        columns: Union[str, List[str]],
        operation: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        other_columns: Optional[Union[str, List[str]]] = None,
        separator: str = " ",
        find: str = None,
        replace_with: str = None,
        new_columns: Union[str, List[str]] = None,
        replace_type: Optional[str] = "float",
        **kwargs
    ):
        return f"""
df = f_string_operation(df, str_or_list("{columns}"), "{operation}", {start}, {end}, str_or_list("{other_columns}"), "{separator}", "{find}", "{replace_with}", str_or_list("{new_columns}"), {replace_type})
"""
    

class NewProcessDatetimeInput(BaseModel):
    """Schema for processing one or more datetime columns."""
    columns: Union[str, list[str]] = Field(
        ...,
        description="List of column names with datetime strings or objects.",
    )
    
        
    operation: Literal[
        "parse",
        "convert_time_taken_to_seconds",
        "extract_year",
        "extract_month",
        "extract_day",
        "extract_weekday",
        "to_timezone",
        "format",
    ] = Field(
        ...,
        description="Type of datetime operation to perform.",
        example="extract_year"
    )
    
    parse_format: str = Field(
        default="%d%m%y",
        description="Format of the date to be parsed (required for parse).",
        example="%y%m%d"
    )

    timezone: Optional[str] = Field(
        default=None,
        description="Timezone name (e.g., 'UTC', 'America/New_York'). Used for timezone conversion.",
        example="UTC"
    )
    
    date_format: Optional[str] = Field(
        default=None,
        description="Format string for output if formatting dates. E.g., '%Y-%m-%d'.",
        example="%Y-%m-%d"
    )

    new_columns: Optional[list[str]] = Field(
        default=None,
        description="Optional list of new column names. If not provided, suffix will be auto-added.",
        example=["order_date_year", "ship_date_year"]
    )

    reasoning: str = Field(
        ...,
        description="Why this datetime operation is needed.",
        example="Extract year for time-series grouping."
    )


class NewProcessDatetimeTool(BaseTool):
    name: str = "f_process_datetime"
    description: str = """Process or transform datetime columns (supports multiple columns)."""
    args_schema: type[NewProcessDatetimeInput] = NewProcessDatetimeInput

    def _run(
        self,
        columns: Union[str, list[str]],
        operation: str,
        parse_format: str = "%d%m%y",
        timezone: Optional[str] = None,
        date_format: Optional[str] = None,
        new_columns: Optional[list[str]] = None,
        **kwargs
    ):
        return f"""
df = f_process_datetime(df,
    str_or_list("{columns}"),
    "{operation}",
    "{parse_format}",
    {timezone},
    {date_format},
    {new_columns})
"""



class LookupInput(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="Substring(s) to search for across all columns. Use a partial name if you are unsure of the exact spelling, e.g. a keyword rather than a full exact value.")
    case_sensitive: bool = Field(False, description="Should the lookup be case-sensitive (default False)")
    reasoning: Optional[str] = Field(
        "",
        description="Why do we need to search this term in the dataframe."
    )


class RenameColumnInput(BaseModel):
    """Schema for renaming dataframe columns."""
    rename_map: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping old column names to new names. Format: {'old_name': 'new_name'}",
        example={"old_name": "new_name"}
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why these renames are being made",
        example="Standardizing column names for database ingestion"
    )
            
class RenameColumnTool(BaseTool):
    name: str = "f_rename_column"
    description: str = "Renames columns in the DataFrame"
    args_schema: type[RenameColumnInput] = RenameColumnInput  # Uses our schema
    
    def _run(self, rename_map: Dict[str, str], reasoning: str):
        return f"""
df = f_rename_column(df, {rename_map}, "{reasoning}")
"""


class LookupTool(BaseTool):
    name: str = "f_lookup"
    description: str = "Substring search across ALL columns for one or more text values. Use this when you are unsure of the exact cell value or spelling (e.g. search a partial company name to find rows where any column contains that word). Returns matching rows and their positions. Prefer this over f_select_row when the exact value is unknown."
    response_format: Literal["text"] = "text"
    args_schema: type[LookupInput] = LookupInput
    
    def _run(self, texts: Union[str, List[str]], case_sensitive: bool = False, **kwargs) -> str:
        if isinstance(texts, list):
            texts_repr = repr(texts)
        else:
            texts_repr = f'"{texts}"'
        return f"""
f_lookup(df, str_or_list({texts_repr}), case_sensitive={case_sensitive})
"""
    
    async def _arun(self, texts: Union[str, List[str]], case_sensitive: bool = False, **kwargs) -> str:
        return self._run(texts, case_sensitive, **kwargs)


class RetrieveOriginalDFInput(BaseModel):
    dummy: bool = Field(default= True, description="Name for the saved dataframe copy")

class RetrieveOriginalDFTool(BaseTool):
    name: str = "f_retrieve_original_df"
    description: str = "Retrieve original dataframe"
    response_format: Literal["text"] = "text"
    args_schema: type[RetrieveOriginalDFInput] = RetrieveOriginalDFInput
    
    def _run(self, dummy: bool = False, **kwargs) -> str:
        return f"""
df = df_og.copy()
"""
    async def _arun(self, dummy: bool = False, **kwargs) -> str:
        return self._run(dummy, **kwargs)
    
class SaveIntermediateAnswerDictInput(BaseModel):
    name: str = Field(..., description="Key for saving the intermediate result")
    content: str = Field(..., description="The intermediate result to store (stringified if needed)")

class SaveIntermediateAnswerDictTool(BaseTool):
    name: str = "f_save_intermediate_answer"
    description: str = "Save intermediate answer into a global dictionary"
    response_format: Literal["text"] = "text"
    args_schema: type[SaveIntermediateAnswerDictInput] = SaveIntermediateAnswerDictInput

    def _run(self, name: str, content: str, **kwargs) -> str:
        return f"""
f_save_intermediate_answer("{name}", "{content}") 
"""

    async def _arun(self, name: str, content: str, **kwargs) -> str:
        return self._run(name, content, **kwargs)

class ViewIntermediateAnswerInput(BaseModel):
    name: Optional[str] = Field(None, description="Key to view specific intermediate result. Leave blank to view all.")

class ViewIntermediateAnswerTool(BaseTool):
    name: str = "f_view_intermediate_answer"
    description: str = "View saved intermediate answers"
    response_format: Literal["text"] = "text"
    args_schema: type[ViewIntermediateAnswerInput] = ViewIntermediateAnswerInput

    def _run(self, name: Optional[str] = None, **kwargs) -> str:
        return f"""
result = f_view_intermediate_answer("{name}")
print(str(result) if result is not None else "No results")
"""

    async def _arun(self, name: Optional[str] = None, **kwargs) -> str:
        return self._run(name, **kwargs)
    
# ===== MODALITY META-ACTION =====
# This tool does not mutate df. Its _run() returns a sentinel string that is
# intercepted by run_reasoning_loop before _capture_exec, so it never reaches
# the Python exec environment. It lets the agent choose its observation mode.

MODALITY_SENTINEL_PREFIX = "__SWITCH_MODALITY__:"


class SwitchModalityInput(BaseModel):
    """Schema for switching the table observation modality."""
    mode: Literal["text", "image", "multimodal"] = Field(
        ...,
        description=(
            "Observation mode for the next step. "
            "'text': serialized rows — exact values, good for precise filters/lookups. "
            "'image': rendered table image — good for overall layout and structure. "
            "'multimodal': image + column list — balanced; use when both structure and values matter."
        ),
    )
    reason: str = Field(..., description="Why you are switching modality.")


class SwitchModalityTool(BaseTool):
    """Meta-action: change how the table is observed in the next step."""
    name: str = "f_switch_modality"
    description: str = (
        "Switch the table observation modality for subsequent steps. "
        "Use 'text' when you need exact cell values (e.g. precise filter conditions or numeric lookups). "
        "Use 'image' when you need to understand overall table layout or scan many rows at once. "
        "Use 'multimodal' as a balanced default. "
        "This does NOT modify the table — it only changes how you observe it."
    )
    response_format: Literal["text"] = "text"
    args_schema: type[SwitchModalityInput] = SwitchModalityInput

    def _run(self, mode: str, reason: str, **kwargs) -> str:
        return f"{MODALITY_SENTINEL_PREFIX}{mode}"

    async def _arun(self, mode: str, reason: str, **kwargs) -> str:
        return self._run(mode, reason, **kwargs)


# ===== TOOL REGISTRY =====
# This is where the magic happens - just add tools to this list!

ENHANCED_MODULAR_TOOLS = [
    # Core tools
    SelectColumnTool(),
    NewPrintTableTool(),
    # Enhanced tools - easy to add!
    CalculateAverageTool(),
    NewGetDataInfoTool(),
    # Additional tools from enhanced_CoT_tools.py
    SelectRowTool(),
    SortByTool(),
    AggregateTool_v0(),
    ComputeColumnTool(),
    DistinctCountTool_v2(),
    FinalAnswerTool(),
    RenameColumnTool(), 
    NewProcessDatetimeTool(),
    StringOperationTool(),
    LookupTool(),
    RetrieveOriginalDFTool(),
    SwitchModalityTool(),
]

# Want to add more tools? Just create them and add to the list above!
# LangChain's bind_tools() will automatically:
# 1. Extract tool schemas from Pydantic models
# 2. Include them in the LLM prompt
# 3. Validate tool calls
# 4. Route to the correct tool execution

# Example of how easy it is to add a new tool:
"""
class NewToolInput(BaseModel):
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(default=10, description="Description of param2")

class NewTool(BaseTool):
    name: str = "f_new_tool"
    description: str = "Description of what this tool does"
    response_format: Literal["text"] = "text"
    args_schema: type[NewToolInput] = NewToolInput
    
    def _run(self, param1: str, param2: int = 10, **kwargs) -> str:
        return f"# Generated Python code here"
    
    async def _arun(self, param1: str, param2: int = 10, **kwargs) -> str:
        return self._run(param1, param2, **kwargs)

# Then just add: NewTool() to ENHANCED_MODULAR_TOOLS list!
""" 
