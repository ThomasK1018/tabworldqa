def function_definition() -> str:
    return f"""
import ast
import re
from typing import Optional, Union, List, Dict, Literal, Sequence, Callable, Any
from pydantic import BaseModel, Field

def f_print_table(df, max_rows: int = 20) -> str:
    if len(df) <= max_rows:
        print("📄 All rows:")
        print(df.to_json())
    else:
        print(f"📄 First {{max_rows}} rows:")
        print(df.head(max_rows).to_json())
        if len(df) > max_rows:
            print(f"... and {{len(df) - max_rows}} more rows")  
            
            
def f_select_column(df: pd.DataFrame, column_names: Union[str, List[str]], reasoning: str) -> pd.DataFrame:
    print(f"🔧 Selecting columns: {{column_names}}")
    print(f"💭 Reasoning: {{reasoning}}")

    available_columns = df.columns.tolist()
    missing_columns = [col for col in column_names if col not in available_columns]

    if missing_columns:
        print(f"❌ Error: Columns not found: {{missing_columns}}")
        print(f"Available columns: {{available_columns}}")
    else:
        df = df[column_names].copy()
        print(f"✅ Successfully selected columns")
        print(f"📊 New table shape: {{df.shape}}")
        print(f"🔍 Selected sample info:")
        print(df.sample(20).to_json() if len(df) > 20 else df.to_json())  
        f_aggregate(df, df.columns, show_stats_only = True)
    return df 
        
def f_lookup(
    df: pd.DataFrame,
    texts: Union[str, List[str]],
    case_sensitive: bool = False
) -> Dict[str, Dict[str, Union[bool, pd.DataFrame, List[Dict]]]]:

    if isinstance(texts, str):
        texts = [texts]

    results: Dict[str, Dict[str, Union[bool, pd.DataFrame, List[Dict]]]] = {{}}
    df_str = df.astype(str)

    for text in texts:
        print(text)

        match_mask = df_str.apply(
            lambda col: col.str.contains(text, case=case_sensitive, na=False, regex=False)
        )

        matched_positions = []
        for col_idx, col in enumerate(match_mask.columns):
            rows = match_mask.index[match_mask[col]]
            for row in rows:
                matched_positions.append({{
                    "row_index": int(row),
                    "column": col,
                    "column_index": int(col_idx),
                    "value": df.iloc[row, col_idx]
                }})

        matching_rows = df.loc[match_mask.any(axis=1)].copy()
        found = not matching_rows.empty

        print(f"{{text}} is{{'' if found else ' not'}} in the dataframe")

        if found:
            print("🔍 Matches (preview):")
            for m in matched_positions[:20]:
                print(m)

        results[text] = {{
            "found": found,
            "rows": matching_rows,
            "matches": matched_positions
        }}

    return results
 
       
import pandas as pd
from typing import Union, List, Dict

def f_search_name_in_df_v0(
    df: pd.DataFrame,
    texts: Union[str, List[str]],
    case_sensitive: bool = False
) -> Dict[str, Dict[str, Union[bool, pd.DataFrame]]]:
    df_copy = df.copy()
    
    if isinstance(texts, str):
        texts = [texts]

    results = {{}}

    for text in texts:
        text = str(text).strip()

        # Detect numeric query
        is_number = False
        try:
            num_value = float(text)
            is_number = True
        except ValueError:
            pass

        # ---- Numeric search ----
        if is_number:
            row_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
            for col in df.columns:
                col_num = pd.to_numeric(df[col], errors="coerce")
                row_mask[col] = col_num == num_value
            row_mask = row_mask.any(axis=1)

        # ---- Text search ----
        else:
            row_mask = df.apply(
                lambda col: col.astype(str).str.contains(text, case=case_sensitive, na=False)
            ).any(axis=1)

        matching_rows = df[row_mask].copy()
        found = not matching_rows.empty

        print(f"{{text}} is{{'' if found else ' not'}} in the dataframe")
        if found:
            preview = matching_rows.head(20)
            print("🔍 Matching rows (preview):")
            print(preview.to_json())

        results[text] = {{
            "found": found,
            "rows": matching_rows
        }}

    return results
 

def f_select_row(
    df: pd.DataFrame,
    indices: Optional[List[int]] = None,
    condition: Optional[Union[str, List[str]]] = None,
    lookup: bool = True,
    reasoning: str = "",
    custom_funcs: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame | None:
    if condition == "None":
        condition = None
    # Normalize cases where the model passed placeholders like [None] or ["None"]
    if isinstance(condition, list):
        cleaned = []
        for c in condition:
            if c is None:
                continue
            if isinstance(c, str) and c.strip().lower() == "none":
                continue
            cleaned.append(c)
        condition = cleaned if cleaned else None

    def show_sample(subdf: pd.DataFrame, label: str):
        if subdf.empty:
            print("⚠️ No rows selected.")
        elif len(subdf) > 20:
            print(f"🔍 {{label}}:")
            print(subdf.sample(20).to_json())
        else:
            print(f"🔍 {{label}}:")
            print(subdf.to_json())

    # Ensure exactly one selector
    provided = sum(x is not None for x in [indices, condition])
    if provided != 1:
        print("❌ Error: Provide either indices OR condition(s), not both.")
        print(f"   indices: {{indices}}")
        print(f"   condition: {{condition}}")
        #return None

    print(f"💭 Reasoning: {{reasoning}}")

    # Case 1: indices
    if indices is not None:
        print(f"🔧 Selecting rows by indices: {{indices}}")
        max_index = len(df) - 1
        invalid = [i for i in indices if i < 0 or i > max_index]
        if invalid:
            print(f"❌ Invalid indices {{invalid}} (valid range: 0–{{max_index}})")
            #return None
        selected = df.iloc[indices].reset_index(drop=True)
        print("✅ Successfully selected rows")
        print(f"📊 New table shape: {{selected.shape}}")
        print("🔍 Selected sample info:")
        show_sample(selected, "Selected sample")

    # Case 2: condition(s)
    else:
        def quote_special_cols(expr: str) -> str:
            updated = expr
            for col in df.columns:
                col_str = str(col)
                # Wrap columns that aren't valid identifiers so df.query can parse them (e.g., 'Res.')
                if not col_str.isidentifier():
                    pattern = rf"(?<!`){{re.escape(col_str)}}(?!`)"
                    updated = re.sub(pattern, f"`{{col_str}}`", updated)
            return updated

        def normalize_contains(expr: str) -> str:
            pattern = r'(?P<col>[^!=<>]+?)\\s+contains\\s+[\\\'"](?P<val>.+?)[\\\'"]'
            match = re.search(pattern, expr, flags=re.IGNORECASE)
            if match:
                col = match.group("col").strip()
                val = match.group("val").replace('"', r"\\\"")
                return f'{{col}}.str.contains("{{val}}", case=False, na=False)'
            return expr

        if isinstance(condition, str):
            conditions_list = [condition]
        else:
            conditions_list = condition  # already a list

        # Normalize friendly "contains" syntax before quoting columns
        conditions_list = [normalize_contains(c) for c in conditions_list]

        sanitized_conditions = [quote_special_cols(c) for c in conditions_list]
        combined = " & ".join(f"({{c}})" for c in sanitized_conditions)
        print(f"🔧 Filtering rows with condition(s): {{conditions_list}}")
        if sanitized_conditions != conditions_list:
            print(f"✨ Sanitized condition(s): {{sanitized_conditions}}")
        print(f"Combined condition: {{combined}}")

        # Allow lightweight custom functions inside query (e.g., lowercase(language) != 'english')
        default_funcs = {{
            "lowercase": lambda s: s.astype(str).str.lower(),
            "uppercase": lambda s: s.astype(str).str.upper(),
            "strip": lambda s: s.astype(str).str.strip(),
        }}
        func_ctx = {{**default_funcs, **(custom_funcs or {{}})}}
        print(f"🧰 Available helper functions in query: {{list(func_ctx.keys())}}")

        try:
            selected = df.query(combined, local_dict=func_ctx, engine="python").reset_index(drop=True)
            print("✅ Successfully selected rows")
            print(f"📊 New table shape: {{selected.shape}}")
            print("🔍 Selected sample info:")
            show_sample(selected, "Selected sample")
        except Exception as e:
            print(f"❌ Error applying condition(s): {{e}}")
            print(f"Available columns: {{df.columns.tolist()}}")
            #return None

    if lookup:
        print("🔍 Your selected info:")
        show_sample(selected, "Selected sample")
        print(f"📊 Current table shape is unchanged: {{df.shape}}")
        #return None
    else:
        print(f"Table is transformed to new shape {{selected.shape}}")
        return selected.copy() 
        
def f_get_data_info(df: pd.DataFrame, show_missing: bool = True) -> None:
    print(f"Shape:")
    print(df.shape)
    print(f"Columns:")
    print(df.columns.tolist())
    print(f"DataFrame:")
    print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
    print(f"🔍 Data Types:")
    print(df.dtypes)
    print(f"❓ Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])      
    
def f_calculate_average(
    df: pd.DataFrame,
    column_name: str,
    group_by_column: str,
    filter_column: Optional[str] = None,
    filter_value: Any = None,
    filter_op: str = "=="
) -> pd.DataFrame:
    print(f"🔧 Calculating average of {{column_name}} by {{group_by_column}}")
    working_df = df

    # Special-case: language != English => report both splits side by side
    if (
        filter_column
        and str(filter_column).lower() == "language"
        and str(filter_op) == "!="
        and isinstance(filter_value, str)
        and str(filter_value).lower() == "english"
        and "language" in df.columns
    ):
        print("🧭 Special handling: computing averages for language == English and language != English")
        english_df = df[df["language"] == filter_value]
        non_english_df = df[df["language"] != filter_value]

        def _print_avg(split_df: pd.DataFrame, label: str):
            if split_df.empty:
                print(f"⚠️ No rows for {{label}}")
                return
            if column_name not in split_df.columns or group_by_column not in split_df.columns:
                print(f"⚠️ Missing columns in {{label}} split. Available: {{split_df.columns.tolist()}}")
                return
            try:
                avg_by_group = split_df.groupby(group_by_column)[column_name].mean().round(2)
                print(f"✅ {{label}} - Average {{column_name}} by {{group_by_column}}:")
                print(avg_by_group)
            except Exception as e:
                print(f"❌ Error computing average for {{label}}: {{e}}")

        _print_avg(english_df, "language == English")
        _print_avg(non_english_df, "language != English")
        return df

    # Optional pre-filter (general case)
    if filter_column:
        if filter_column not in df.columns:
            print(f"❌ Error: Filter column {{filter_column}} not found. Available: {{df.columns.tolist()}}")
            return df
        allowed_ops = {"==", "!=", ">", "<", ">=", "<="}
        if filter_op not in allowed_ops:
            print(f"❌ Unsupported filter_op '{{filter_op}}'. Allowed: {{sorted(allowed_ops)}}")
            return df
        try:
            if filter_op == "==":
                mask = df[filter_column] == filter_value
            elif filter_op == "!=":
                mask = df[filter_column] != filter_value
            elif filter_op == ">":
                mask = df[filter_column] > filter_value
            elif filter_op == "<":
                mask = df[filter_column] < filter_value
            elif filter_op == ">=":
                mask = df[filter_column] >= filter_value
            else:  # "<="
                mask = df[filter_column] <= filter_value
            working_df = df[mask]
            print(f"📌 Applied filter: {{filter_column}} {{filter_op}} {{filter_value}}")
            print(f"📊 Rows after filter: {{len(working_df)}} / {{len(df)}}")
        except Exception as e:
            print(f"❌ Error applying filter {{filter_column}} {{filter_op}} {{filter_value}}: {{e}}")
            return df

    if column_name not in working_df.columns:
        print(f"❌ Error: Column {{column_name}} not found")
        print(f"Available columns: {{working_df.columns.tolist()}}")
    elif group_by_column not in working_df.columns:
        print(f"❌ Error: Column {{group_by_column}} not found")
        print(f"Available columns: {{working_df.columns.tolist()}}")
    else:
        try:
            avg_by_group = working_df.groupby(group_by_column)[column_name].mean().round(2)
            print(f"✅ Average {{column_name}} by {{group_by_column}}:")
            print(avg_by_group)
            highest_avg_group = avg_by_group.idxmax()
            highest_avg_value = avg_by_group.max()
            print(f"\\n🏆 ANSWER: The {{group_by_column}} with highest average {{column_name}} is:")
            print(f"{{group_by_column}}: {{highest_avg_group}}")
            print(f"Average {{column_name}}: {{highest_avg_value}}")
        except Exception as e:
            print(f"❌ Error computing average: {{e}}")
    return df 

def f_sort_by(df: pd.DataFrame, columns: list[str], order: str = "asc") -> pd.DataFrame:
    print(f"🔧 Sorting by columns: {{columns}}")
    print(f"📊 Sort order: {{order}}")
    available_columns = df.columns.tolist()
    missing_columns = [col for col in columns if col not in available_columns]
    if missing_columns:
        print(f"❌ Error: Columns not found: {{missing_columns}}")
        print(f"Available columns: {{available_columns}}")
    elif (order not in ["asc", "desc"]):
        print(order not in ["asc", "desc"])
        print(f"❌ Error: Order must be 'asc' or 'desc', got order")
    else:
        ascending = order == "asc"
        df = df.sort_values(by = columns, ascending = ascending).reset_index(drop = True)
        
        print(f"✅ Successfully sorted by {{columns}}")
        print(f"📊 Table shape: {{df.shape}}")
        print(f"🔍 Sorted sample data:")
        print(df.head(20).to_json() if len(df) > 20 else df.to_json())  
                
    return df
        
def f_aggregate_v0(df: pd.DataFrame, column: str, op: str, show_stats_only: bool, group_by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
    valid_ops = ["count", "sum", "avg", "min", "max"]

    if group_by is None or len(group_by) == 0:
        if show_stats_only:
            print("🔍 Group counts:")
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                print("📈 Group statistics for numeric columns:")
                try:
                    group_stats = df.groupby(column)[numeric_columns].agg(['count', 'mean', 'std']).round(2)
                except Exception:
                    group_stats = df[numeric_columns].agg(['count', 'mean', 'std']).round(2)
                print(group_stats)
            else:
                group_stats = df.groupby(column).size().to_frame('count').round(2)
                print(group_stats)
            try:
                OPS = {{"count": "count","sum": "sum", "avg": "mean", "min": "min", "max": "max",}}
                if op == "diff":
                    # Special case: diff is row-wise, not an aggregation
                    print(f"🔧 Computing diff of column {{column}}")
                    df[f"{{column}}_diff"] = df[column].diff()
                    print(f"✅ Created new column {{column}}_diff")
                    print("🔍 Sample:")
                    print(df[[column, f"{{column}}_diff"]].head(10).to_json())
                    return df

                elif op not in OPS:
                    raise ValueError(f"❌ Unsupported operation: {{op}}. Valid operations: {{list(OPS.keys()) + ['diff']}}")
                else:
                    result = df[column].agg(OPS[op])
                    print(f"✅ {{op.title()}} of {{column}}: {{result}}")
                    df = pd.DataFrame({{op: [result]}})
                    print("🔍 New Result:")
                    print(df)
            except Exception as e:
                print(f"❌ Error computing {{op}}: {{e}}")
                print(f"Column dtype: {{df[column].dtype}}")
        else:
            print(f"🔧 Computing {{op.title()}} of column {{column}}")
            if column not in df.columns:
                print(f"❌ Error: Column {{column}} not found")
                print(f"Available columns: {{df.columns.tolist()}}")
            elif op not in valid_ops:
                print(f"❌ Error: Invalid operation {{op}}. Valid operations: {{valid_ops}}")
            else:
                try:
                    OPS = {{"count": "count","sum": "sum", "avg": "mean", "min": "min", "max": "max",}}
                    if op == "diff":
                        # Special case: diff is row-wise, not an aggregation
                        print(f"🔧 Computing diff of column {{column}}")
                        df[f"{{column}}_diff"] = df[column].diff()
                        print(f"✅ Created new column {{column}}_diff")
                        print("🔍 Sample:")
                        print(df[[column, f"{{column}}_diff"]].head(10).to_json())
                        return df

                    elif op not in OPS:
                        raise ValueError(f"❌ Unsupported operation: {{op}}. Valid operations: {{list(OPS.keys()) + ['diff']}}")
                    else:
                        result = df[column].agg(OPS[op])
                        print(f"✅ {{op.title()}} of {{column}}: {{result}}")
                        df = pd.DataFrame({{op: [result]}})
                        print("🔍 New Result:")
                        print(df)
                        return df
                except Exception as e:
                    print(f"❌ Error computing {{op}}: {{e}}")
                    print(f"Column dtype: {{df[column].dtype}}")
    else:
        if isinstance(group_by, str):
            missing_cols = [c for c in [column] + [group_by] if c not in df.columns]
        else:
            missing_cols = [c for c in [column] + group_by if c not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns: {{missing_cols}}")
            print(f"Available columns: {{df.columns.tolist()}}")
        else:
            if show_stats_only:
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_columns:
                    print("📈 Group statistics for numeric columns:")
                    print(df.groupby(group_by)[numeric_columns].agg(['count', 'mean', 'std']).round(2))
                else:
                    group_stats = df.groupby(group_by).size().to_frame('count').round(2)
                    print(group_stats)
                try:
                    OPS = {{"count": "count","sum": "sum", "avg": "mean", "min": "min", "max": "max",}}

                    if op not in OPS:
                        raise ValueError(f"Unsupported operation: {{op}}")
                    result = df[column].agg(OPS[op])
                    print(f"✅ {{op.title()}} of {{column}}: {{result}}")
                    df = pd.DataFrame({{op: [result]}})
                    print("🔍 New Result:")
                    print(df)
                except Exception as e:
                    print(f"❌ Error computing {{op}}: {{e}}")
                    print(f"Column dtype: {{df[column].dtype}}")
            else:
                OPS = {{"count": "count", "sum": "sum", "avg": "mean", "min": "min", "max": "max"}}

                if op == "diff":
                    print(f"🔧 Computing diff of column {{column}} within groups {{group_by}}")
                    try:
                        result = df.groupby(group_by)[column].diff()
                        new_col = f"{{column}}_diff"
                        df[new_col] = result
                        print(f"✅ Created new column {{new_col}}")
                        print("🔍 Sample:")
                        print(df[[*([group_by] if isinstance(group_by, str) else group_by), column, new_col]].head(10))
                        return df
                    except Exception as e:
                        print(f"❌ Error computing diff: {{e}}")
                        return df

                elif op not in OPS:
                    raise ValueError(f"Unsupported operation: {{op}}")
                else:
                    print(f"🔧 Computing {{op.title()}} of column {{column}} grouped by {{group_by}}")
                    try:
                        result = df.groupby(group_by)[column].agg(OPS[op])
                        df = result.reset_index().rename(columns={{column: f"{{op}}_result"}})
                        print("🔍 Result as table:")
                        print(df.sample(20) if len(df) > 20 else df)
                        return df
                    except Exception as e:
                        print(f"❌ Error computing grouped {{op}}: {{e}}")
                        cols = [c for c in set([column] + group_by)]
                        print("Column dtypes:")
                        print(df[cols].dtypes)
     
def f_aggregate(
    df: pd.DataFrame,
    columns_ops: Union[str, Dict[str, str], List[Dict[str, str]]] = None,
    op: Optional[str] = None,
    show_stats_only: bool = False,
    group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    OPS = {{"count": "count", "sum": "sum", "avg": "mean", "min": "min", "max": "max"}}

    # 🔹 Normalize input
    if isinstance(columns_ops, str) and op:  # Legacy mode
        agg_map = {{columns_ops: OPS.get(op, op)}}
        single_col, single_op = columns_ops, op
    elif isinstance(columns_ops, dict):
        agg_map = {{col: OPS.get(op, op) for col, op in columns_ops.items()}}
        single_col, single_op = None, None
    elif isinstance(columns_ops, list):
        # Accept list of dicts or list of column names
        if all(isinstance(x, dict) for x in columns_ops):
            agg_map = {{}}
            for mapping in columns_ops:
                for col, op in mapping.items():
                    agg_map[col] = OPS.get(op, op)
            single_col, single_op = None, None
        elif all(isinstance(x, str) for x in columns_ops):
            fallback_op = OPS.get(op, op) if op else "count"
            agg_map = {{c: fallback_op for c in columns_ops}}
            single_col, single_op = None, None
        else:
            raise ValueError("Invalid columns_ops list. Provide list of dicts or list of column names.")
    else:
        raise ValueError("Invalid columns_ops or op. Provide str+op, dict, or list of dicts.")

    # 🔹 Special case: diff (only works in legacy single-col mode)
    if single_op == "diff":
        if group_by:
            print(f"🔧 Computing diff of column {{single_col}} within groups {{group_by}}")
            try:
                df[f"{{single_col}}_diff"] = df.groupby(group_by)[single_col].diff()
            except Exception as e:
                print(f"❌ Error computing diff: {{e}}")
            return df
        else:
            print(f"🔧 Computing diff of column {{single_col}}")
            try:
                df[f"{{single_col}}_diff"] = df[single_col].diff()
            except Exception as e:
                print(f"❌ Error computing diff: {{e}}")
            return df

    # 🔹 Stats-only mode
    if show_stats_only:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if group_by:
            if numeric_columns:
                print("📈 Group statistics for numeric columns:")
                print(df.groupby(group_by)[numeric_columns].agg(["count", "mean", "std", "sum"]).round(2))
            else:
                print(df.groupby(group_by).size().to_frame("count").round(2))
        else:
            if numeric_columns:
                print("📈 Overall statistics for numeric columns:")
                print(df[numeric_columns].agg(["count", "mean", "std", "sum"]).round(2))
            else:
                print(df[column_ops].agg(["count"]))
        #return None

    # 🔹 Normal aggregations
    try:
        if group_by:
            grouped = df.groupby(group_by).agg(agg_map).reset_index()
            print("📊 Grouped aggregation preview:")
            print(grouped.head(10).to_json())
            return grouped
        else:
            result = df.agg(agg_map)
            print("📊 Aggregation result:")
            print(result.to_dict())
            return result.to_frame().T
    except Exception as e:
        print(f"❌ Error computing aggregation: {{e}}")
        return df

      

from pandas.api.types import is_timedelta64_dtype
import operator as ope
def f_compute_column_v0(df: pd.DataFrame, new_col: str, col_A: str, op: str, col_B: str | int | float, element_wise: bool) -> pd.DataFrame:
    valid_ops = ["add", "sub", "mul", "div", "ratio"]

    OPS = {{
            "add": ope.add,
            "sub": ope.sub,
            "mul": ope.mul,
            "div": ope.truediv}}

    if element_wise:
        print(f"🔧 Creating column {{new_col}} = {{col_A}} {{op}} {{col_B}}")
        if op not in valid_ops:
            print(f"❌ Error: Invalid operation {{op}}. Valid operations: valid_ops")
        elif col_A not in df.columns:
            print(f"❌ Error: Column {{col_A}} not found")
            print(f"Available columns: {{df.columns.tolist()}}")
        else:
            try:
                if isinstance(col_B, str) and col_B not in df.columns:
                    print(f"❌ Error: Column {{col_B}} not found")
                    print(f"Available columns: {{df.columns.tolist()}}")
                else:
                    if op not in OPS:
                        raise ValueError(f"Unsupported operation: {{op}}")

                    df[new_col] = OPS[op](df[col_A], df[col_B])
                
                    print(f"📊 New table shape: {{df.shape}}")
                    print(f"🔍 New column sample:")
                    print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
                    
            except Exception as e:
                print(f"❌ Error creating column: {{e}}")
                print(f"Check data types and values for division by zero")
    else:
        value = float(col_B)
        print(f"🔧 Creating column {{new_col}} = {{col_A}} {{op}} {{value}}")
        if op not in valid_ops:
            print(f"❌ Error: Invalid operation {{op}}. Valid operations: valid_ops")
        elif col_A not in df.columns:
            print(f"❌ Error: Column {{col_A}} not found")
            print(f"Available columns: {{df.columns.tolist()}}")
        else:
            try:
                if op not in OPS:
                    raise ValueError(f"Unsupported operation: {{op}}")

                df[new_col] = OPS[op](df[col_A], value)
                print(f"✅ Successfully created column {{new_col}}")
                print(f"📊 New table shape: {{df.shape}}")
                print(f"🔍 New column sample:")
                print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
                    
            except Exception as e:
                print(f"❌ Error creating column: {{e}}")
                print(f"Check data types and values for division by zero")
                
    if is_timedelta64_dtype(df[new_col]):
        df[new_col] = df[new_col] / pd.Timedelta(days=1)  
    return df

def f_compute_column(
    df: pd.DataFrame,
    new_col: str,
    col_A: str,
    op: str,
    col_B: str | int | float,
    element_wise: bool,
    special_op: Optional[str] = None
) -> pd.DataFrame:
    import operator as ope
    OPS = {{
        "add": ope.add,
        "sub": ope.sub,
        "mul": ope.mul,
        "div": ope.truediv,
        "ratio": lambda a, b: a / b
    }}

    if special_op == "time_diff":
        # Expect col_A = datetime column, col_B = groupby column
        if col_A not in df.columns or col_B not in df.columns:
            raise ValueError(f"Columns {{col_A}} or {{col_B}} not found")
        df[new_col] = df.groupby(col_B)[col_A].diff().dt.total_seconds() / 3600
        print(f"✅ Created {{new_col}} as time difference (hours) within groups of {{col_B}}")
        return df

    if element_wise:
        if op not in OPS:
            raise ValueError(f"Unsupported op: {{op}}")
        df[new_col] = OPS[op](df[col_A], df[col_B] if isinstance(col_B, str) else col_B)
    else:
        if op not in OPS:
            raise ValueError(f"Unsupported op: {{op}}")
        df[new_col] = OPS[op](df[col_A], float(col_B))

    print(f"✅ Created column {{new_col}}")
    return df
        
def f_distinct_count(df: pd.DataFrame, column: str) -> None:
    print(f"🔧 Counting distinct values in column {{column}}")
    if column not in df.columns:
        print(f"❌ Error: Column {{column}} not found")
        print(f"Available columns: {{df.columns.tolist()}}")
    else:
        distinct_count = df[column].nunique()
        
        print(f"✅ Number of distinct values in {{column}}: {{distinct_count}}")
        
        if distinct_count <= 20:
            print(f"🔍 Distinct values:")
            print(df[column].unique())
        else:
            print(f"🔍 First 20 distinct values:")
            print(df[column].unique()[:20])
            print(f"... and {{distinct_count - 20}} more")
        
        result_df = pd.DataFrame({{"distinct_count": [distinct_count]}})
        print(f"📊 Result:")
        print(result_df)    
       
def f_distinct_count_v2(df: pd.DataFrame, columns: Union[str, List[str]]) -> None:
    if isinstance(columns, str):
        columns = [columns]
        
    for column in columns:
        print(f"🔧 Counting distinct values in column {{column}}")
        if column not in df.columns:
            print(f"❌ Error: Column {{column}} not found")
            print(f"Available columns: {{df.columns.tolist()}}")
            continue

        distinct_count = df[column].nunique()
        print(f"✅ Number of distinct values in {{column}}: {{distinct_count}}")

        if distinct_count <= 20:
            print(f"🔍 Distinct values:")
            print(df[column].unique())
        else:
            print(f"🔍 First 20 distinct values:")
            print(df[column].unique()[:20])
            print(f"... and {{distinct_count - 20}} more")

        result_df = pd.DataFrame({{f"{{column}}_distinct_count": [distinct_count]}})
        print(f"📊 Result for {{column}}:")
        print(result_df)
        
 
def f_rename_column(df: pd.DataFrame, rename_map: Dict[str, str], reasoning: str) -> pd.DataFrame:
    if df is None:
        raise ValueError("No DataFrame provided")    
    try:
        df.rename(columns=rename_map, inplace=True)  
        return df
    except Exception as e:
        print(e)
        
def f_handle_missing_value_v0(df: pd.DataFrame, strategy: str, columns: Optional[list[str]] = None, fill_value: Optional[float | str] = None):
    if df is None:
        raise ValueError("No DataFrame provided") 
    target_cols = columns if columns else df.columns
    try:
        if strategy == "drop":
            df.dropna(subset=target_cols, inplace=True)
        elif strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value required for 'fill' strategy")
            df[target_cols] = df[target_cols].fillna(fill_value)
        elif strategy == "interpolate":
            df[target_cols] = df[target_cols].interpolate()
        else:
            raise ValueError(f"Unknown strategy: {{strategy}}")
        return df            
    except Exception as e:
        print(e) 
 
def f_handle_missing_value(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[list[str]] = None,
    fill_value: Optional[float | str] = None,
    na_symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    if df is None:
        raise ValueError("No DataFrame provided") 
    target_cols = columns if columns else df.columns

    # Replace custom NA symbols first
    if na_symbols:
        for col in target_cols:
            if df[col].dtype == "object":
                df[col] = df[col].replace(na_symbols, pd.NA)

    try:
        if strategy == "drop":
            df.dropna(subset=target_cols, inplace=True)
        elif strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value required for 'fill' strategy")
            df[target_cols] = df[target_cols].fillna(fill_value)
        elif strategy == "interpolate":
            df[target_cols] = df[target_cols].interpolate()
        else:
            raise ValueError(f"Unknown strategy: {{strategy}}")
        return df            
    except Exception as e:
        print(e)

        
def f_drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first", reset_index: bool = True, reasoning: str = "") -> pd.DataFrame:
    if df is None:
        raise ValueError("No DataFrame provided")
    try:
        df.drop_duplicates(
            subset=subset,
            keep=keep,
            inplace=True
        )       
        if reset_index:
            df = df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        raise ValueError(e)   
        
def f_pivot_table(df: pd.DataFrame, operation: str, reasoning: str, index: Union[str, List[str]] = None, columns: Union[str, List[str]] = None, values: Union[str, List[str]] = None, aggfunc: Union[str, List[str]] = "mean", var_name: str = "variable", value_name: str = "value") -> pd.DataFrame:
    if df is None:
        raise ValueError("No DataFrame provided")
    try:
        if operation == "pivot":
            pivoted = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
            if isinstance(pivoted.columns, pd.MultiIndex):
                pivoted.columns = ['_'.join(map(str, col)) for col in pivoted.columns.values]
            df = pivoted.reset_index()
                    
        elif operation == "melt":
            df = df.melt(id_vars=index,value_vars=values,var_name=var_name,value_name=value_name)
        else:
            raise ValueError(f"Unknown operation: {{operation}}")
        print("Sample results:")
        print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
        return df

    except Exception as e:
        raise ValueError(e)  
        
def f_join(left_df, right_df, join_type, left_on=None, right_on=None, cross_join=False, suffix="_right", drop_duplicates=True):
    if left_df is None or right_df is None:
            raise ValueError("Both left and right DataFrames must be provided")
    try:
        if cross_join:
            merged = left_df.assign(key=1).merge(
                    right_df.assign(key=1),
                    on="key"
                ).drop("key", axis=1)
        else:
            merged = pd.merge(
                    left=left_df,
                    right=right_df,
                    how=join_type,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=("", suffix)
                )
        if drop_duplicates:
                merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged
    except Exception as e:
        raise ValueError(e)
        
def f_sample(df, n: Optional[int] = None, frac: Optional[float] = None, random_state: Optional[int] = None, replace: bool = False):
    if df is None:
        raise ValueError("A DataFrame must be provided for sampling.")

    if n is None and frac is None:
        raise ValueError("Either `n` or `frac` must be specified.")
    if n is not None and frac is not None:
        raise ValueError("Specify only one of `n` or `frac`.")

    try:
        sampled_df = df.sample(
                n=n,
                frac=frac,
                replace=replace,
                random_state=random_state
            )
        return sampled_df
    except Exception as e:
        raise ValueError(e)
        
def f_bin_values(df, column: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None,
        include_lowest: bool = True,
        right: bool = True,
        new_column: Optional[str] = None):
    if df is None:
        raise ValueError("A DataFrame must be provided.")
    if column not in df.columns:
        raise ValueError(f"Column {{column}} not found in DataFrame.")

    new_column = f"{{column}}_binned" or new_column
    binned = pd.cut(df[column],bins=bins,labels=labels,include_lowest=include_lowest,right=right)

    df[new_column] = binned
    return df            
    

def f_string_operation(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    operation: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    other_columns: Optional[Union[str, List[str]]] = None,
    separator: str = " ",
    find: str = None,
    replace_with: str = None,
    new_columns: Union[str, List[str]] = None,
    replace_type: str = "float",
) -> pd.DataFrame:
    if find == 'None':
        find = None
    if replace_with == 'None':
        replace_with = None
    if df is None:
        raise ValueError("A DataFrame must be provided.")

    # Normalize to list
    if isinstance(columns, str):
        columns = [columns]

    # Validate inputs
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {{missing}}")

    if new_columns is not None and len(new_columns) != len(columns):
        raise ValueError("`new_columns` must have the same length as `columns`.")

    def _pair_other(j: int) -> str:
        if other_columns is None:
            raise ValueError("`other_columns` must be provided for 'concat'.")
        oc = other_columns[j] if isinstance(other_columns, list) else other_columns
        if oc not in df.columns:
            raise ValueError(f"`other_columns` {{oc}} not found in DataFrame.")
        return oc

    string_ops = {{
        "lowercase": lambda s: s.str.lower(),
        "uppercase": lambda s: s.str.upper(),
        "titlecase": lambda s: s.str.title(),
    }}

    try:
        for i, col in enumerate(columns):
            out_col = new_columns[i] if new_columns else col
            base = df[col].astype(str)

            if operation in string_ops:
                df[out_col] = string_ops[operation](base)

            elif operation == "substring":
                if start is None:
                    raise ValueError("`start` must be specified for 'substring'.")
                df[out_col] = base.str[start:end]

            elif operation == "concat":
                oc = _pair_other(i)
                df[out_col] = base + separator + df[oc].astype(str)

            elif operation == "replace":
                if find is None:
                    raise ValueError("`find` must be specified for 'replace'.")
                replacement_value = pd.NA if replace_with == "pd.NA" else replace_with
                df[out_col] = base.replace(find, replacement_value, regex=True)

            elif operation == "type_change":
                bad_rows, results = [], []
                for idx, val in df[col].items():
                    try:
                        if pd.isna(val):
                            results.append(pd.NA)
                        elif str(replace_type).startswith("datetime"):
                            results.append(pd.to_datetime(val))
                        else:
                            results.append(pd.Series([val]).astype(replace_type).iloc[0])
                    except Exception as e:
                        bad_rows.append((idx, val, str(e)))
                        results.append(pd.NA)

                if bad_rows:
                    err_preview = " ".join(
                        f"Row {{i}}: {{repr(v)}}" for i, v, _ in bad_rows[:10]
                    )
                    raise ValueError(
                        f"Failed converting {{col}} to dtype {{replace_type!r}} "
                        f"due to {{len(bad_rows)}} failed cases, e.g. {{err_preview}}"
                    )
                df[out_col] = pd.Series(results, index=df.index, name=out_col)

            else:
                raise ValueError(f"Unsupported operation: {{operation}}")
        print("Sample rows:")
        print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
        return df

    except Exception as e:
        raise ValueError(e)
        

        
def f_process_datetime(
    df: pd.DataFrame,
    columns: Union[str, list[str]],
    operation: str,
    parse_format: str = "%d%m%y",
    timezone: Optional[str] = None,
    date_format: Optional[str] = None,
    new_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    
    # Normalize "None" strings
    timezone = None if timezone == "None" else timezone
    date_format = None if date_format == "None" else date_format

    if df is None:
        raise ValueError("A DataFrame must be provided.")

    # Normalize columns to a list
    if isinstance(columns, str):
        columns = [columns]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {{missing}}. Dataframe only contains the following columns: {{df.columns.tolist()}}")

    # Validate/prepare `new_columns`
    if new_columns is not None and len(new_columns) != len(columns):
        raise ValueError("`new_columns` must have the same length as `columns`.")
    def _out_name(i: int, col: str, operation: str) -> str:
        if new_columns:
            return new_columns[i]
        return col if operation == "parse" else f"{{col}}_{{operation}}"

    # Allow either a single parse_format or a list aligned with `columns`
    if isinstance(parse_format, (list, tuple)):
        if len(parse_format) != len(columns):
            raise ValueError("When passing multiple parse formats, match `columns` length.")
        parse_formats = list(parse_format)
    else:
        parse_formats = [parse_format] * len(columns)

    def _to_seconds(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        parts = s.split(":")
        try:
            if len(parts) == 2:        # mm:ss
                m, sec = parts
                return int(m) * 60 + float(sec)
            elif len(parts) == 3:      # hh:mm:ss
                h, m, sec = parts
                return int(h) * 3600 + int(m) * 60 + float(sec)
            else:
                # Fallback: let pandas try
                return pd.to_timedelta(s).total_seconds()
        except Exception:
            return np.nan

    for i, col in enumerate(columns):
        out_col = _out_name(i, col, operation)
        ser = df[col]

        try:
            if operation == "parse":
                fmt = parse_formats[i]
                df[out_col] = pd.to_datetime(ser, format=fmt, errors="coerce")

            elif operation == "convert_time_taken_to_seconds":
                df[out_col] = ser.apply(_to_seconds)

            else:
                dt = pd.to_datetime(ser, errors="coerce")

                if operation == "extract_year":
                    df[out_col] = dt.dt.year
                elif operation == "extract_month":
                    df[out_col] = dt.dt.month
                elif operation == "extract_day":
                    df[out_col] = dt.dt.day
                elif operation == "extract_weekday":
                    df[out_col] = dt.dt.day_name()

                elif operation == "to_timezone":
                    if timezone is None:
                        raise ValueError("`timezone` must be specified for 'to_timezone'.")
                    # localize naive as UTC then convert
                    try:
                        localized = dt.dt.tz_localize("UTC")
                    except TypeError:
                        # already tz-aware
                        localized = dt
                    df[out_col] = localized.dt.tz_convert(timezone)

                elif operation == "format":
                    if date_format is None:
                        raise ValueError("`date_format` must be specified for 'format'.")
                    df[out_col] = dt.dt.strftime(date_format)

                else:
                    raise ValueError(f"Unsupported operation: {{operation}}")

        except Exception as e:
            raise ValueError(f"Error processing column {{col}}: {{e}}")

    # ✅ return once after processing all columns
    print("Success!")
    print("Current sample rows:")
    print(df.sample(20).to_json() if len(df) > 20 else df.to_json())
    return df
            
def f_column_twoway_splitter(df, column: str, sep: str, new_columns: list[str]):
    df[new_columns[0]] = df[column].apply(lambda x: str(x).split(sep)[0] if pd.notnull(x) else None)
    df[new_columns[1]] = df[column].apply(lambda x: str(x).split(sep)[1] if pd.notnull(x) and sep in str(x) else None)
    return df
    

def f_check_hidden_duplicates(df: pd.DataFrame, column_name: str):
    def has_hidden_repeats(
        data: Union[pd.Series, pd.DataFrame], column: Optional[str] = None, sep: str = ",") -> bool:
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("When passing a DataFrame, you must supply `column`.")
            s = data[column]
        elif isinstance(data, pd.Series):
            s = data
        else:
            raise TypeError("`data` must be a pandas Series or DataFrame.")

        s = s.astype("string").dropna()

        seen_single = set()
        seen_composite = set()

        for entry in s:
            parts = [p.strip() for p in str(entry).split(sep) if p is not None]
            if len(parts) > 1:
                seen_composite.update(parts)
            else:
                seen_single.update(parts)

        hidden = bool(seen_single & seen_composite)
        print(
            f"There are{{'' if hidden else ' no'}} hidden repeats. {{'Use ExplodeConcurrentNamesInSameRowTool() to extract all values!' if hidden else 'We are good to go!'}}"
        )
    if column_name in df.columns:
        has_hidden_repeats(df[column_name])
    else: 
        print(f"Column {{column_name}} not found in DataFrame.")
        print(f"Available columns: {{df.columns.tolist()}}")
    
def f_explode_concurrent_names_in_row(df: pd.DataFrame, column_name: str, symbol: str):
    df[column_name] = df[column_name].str.split(symbol)
    df = df.explode(column_name)
    df[column_name] = df[column_name].str.strip()
    return df
    
def f_create_new_bool(df: pd.DataFrame, col: str, new_col: str, condition: str):
    df[new_col] = df[col].apply(lambda x: eval(condition, {{}}, {{"x": x}})) 
    return df
    
def f_find_overlapping(df: pd.DataFrame, col1: str, col2: str):
    intersection = set(df[col1]) & set(df[col2])
    print(intersection)
    
INTERMEDIATE_ANSWERS = {{}}

def f_save_intermediate_answer(name: str, content) -> None:
    global INTERMEDIATE_ANSWERS
    INTERMEDIATE_ANSWERS[name] = content
    print(f"💾 Saved intermediate answer under key '{{name}}'")
    print(f"🔍 Current keys: {{list(INTERMEDIATE_ANSWERS.keys())}}")
    
def f_view_intermediate_answer(name: Optional[str] = None):
    global INTERMEDIATE_ANSWERS
    if not INTERMEDIATE_ANSWERS:
        print("⚠️ No intermediate answers saved yet.")
        return None
    
    if name:
        if name in INTERMEDIATE_ANSWERS:
            print(f"📂 Intermediate answer [{{name}}]:")
            print(INTERMEDIATE_ANSWERS[name])
            return INTERMEDIATE_ANSWERS[name]
        else:
            print(f"❌ No entry found under key '{{name}}'")
            print(f"Available keys: {{list(INTERMEDIATE_ANSWERS.keys())}}")
            return None
    else:
        print("📂 All intermediate answers:")
        for k, v in INTERMEDIATE_ANSWERS.items():
            print(f" - {{k}}: {{type(v)}}")
        return INTERMEDIATE_ANSWERS

def f_retrieve_original_df(df: pd.DataFrame, df_og: pd.DataFrame) -> pd.DataFrame:
    "Reset working DataFrame to the original copy."
    print("♻️ Refreshing df from df_og")
    df = df_og.copy()
    print(f"📊 Refreshed shape: {{df.shape}}")
    return df
    
    
def is_numeric_string(x: str) -> bool:
    try:
        float(x)
        return True
    except:
        return False

def normalize_numeric_string(x: str) -> str:
    if not is_numeric_string(x):
        return x   # 👈 leave untouched

    if "." in x:
        # true decimal number
        y = str(float(x))
        return y.rstrip("0").rstrip(".")
    else:
        # integer-like
        return str(int(float(x)))


def str_or_list(string):
    try:
        print(f'We are searching for {{string}} with type {{type(string)}}')
        texts = ast.literal_eval(string)

        # Case: single string
        if isinstance(texts, str):
            if "," in texts:
                items = [t.strip() for t in texts.split(",") if t.strip()]
                return [normalize_numeric_string(i) for i in items]
            return [normalize_numeric_string(texts)]

        # Case: numeric
        if isinstance(texts, (float, int)):
            return [normalize_numeric_string(str(texts))]

        # Case: list
        if isinstance(texts, list):
            output = []
            for item in texts:
                if isinstance(item, str):
                    try:
                        item = ast.literal_eval(item)
                    except (ValueError, SyntaxError):
                        pass  # plain string column name — use as-is
                item_str = normalize_numeric_string(str(item))
                output.append(item_str)
                print(f'We are searching for {{item_str}} with type {{type(item)}}')
            return output

        return [normalize_numeric_string(str(texts))]

    except:
        # Raw string fallback
        if isinstance(string, str) and "," in string:
            return [normalize_numeric_string(s.strip()) for s in string.split(",") if s.strip()]
        return [normalize_numeric_string(str(string))]
"""
