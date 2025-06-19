import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Libraries
    import marimo as mo
    import logging
    import polars as pl
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from matplotlib import pyplot as plt
    from typing import Literal, get_args, TypeAlias, Optional, TypedDict, Dict
    from great_tables import GT, style, md
    import re

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Parameters
    LOCAL_PATH_TO_SALES_CSV = "./data/sales_update202504.csv"

    # Styling
    StylePresetKey = Literal["default", "alert", "soft_pink"]


    class StylePreset(TypedDict):
        style: int
        style_color: str


    STYLE_PRESETS: Dict[StylePresetKey, StylePreset] = {
        "default": {"style": 1, "style_color": "gray"},
        "alert": {"style": 3, "style_color": "red"},
        "soft_pink": {"style": 4, "style_color": "pink"},
        "bluey": {"style": 6, "style_color": "blue"},
        "forest": {"style": 2, "style_color": "green"},
    }
    return (
        Dict,
        GT,
        LOCAL_PATH_TO_SALES_CSV,
        Optional,
        STYLE_PRESETS,
        StylePresetKey,
        md,
        mo,
        pd,
        pl,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Funciones Auxiliares""")
    return


@app.cell
def _(pl):
    def make_null_count_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Computes the count and percentage of null values for each column in a Polars LazyFrame.

        This function calculates the number of nulls per column and the corresponding
        percentage of null values relative to the total number of rows. It returns
        a new LazyFrame with columns representing the variable name, null count, and
        null percentage, sorted by null count in descending order.

        Args:
            lf (pl.LazyFrame): Input Polars LazyFrame to analyze.

        Returns:
            pl.LazyFrame: A LazyFrame with three columns:
                - "variable" (str): The name of each column in the input LazyFrame.
                - "null_count" (int): The count of null values in each column.
                - "null_percentage" (float): The percentage of null values relative to
                  the total number of rows, rounded to two decimal places.

        Raises:
            pl.ComputeError: If there is an error computing null counts or collecting the schema.

        Notes:
            - The total row count is computed eagerly via `.collect()` to accurately
              calculate the null percentages.
            - The returned LazyFrame is sorted in descending order by null count,
              so columns with the most missing data appear first.
            - This function is intended for data exploration and quality assessment
              in lazy execution pipelines.

        Example:
            >>> lf = pl.LazyFrame({
            ...     "a": [1, None, 3],
            ...     "b": [None, None, 3]
            ... })
            >>> null_stats = make_null_count_lf(lf).collect()
            >>> print(null_stats)
            shape: (2, 3)
            ┌──────────┬────────────┬─────────────────┐
            │ variable ┆ null_count ┆ null_percentage │
            │ ---      ┆ ---        ┆ ---             │
            │ str      ┆ u32        ┆ f64             │
            ╞══════════╪════════════╪═════════════════╡
            │ b        ┆ 2          ┆ 66.67           │
            │ a        ┆ 1          ┆ 33.33           │
            └──────────┴────────────┴─────────────────┘
        """
        height_expr = lf.select(pl.len()).collect().item()

        return (
            lf.null_count()
            .unpivot(variable_name="variable", value_name="null_count")
            .sort(by="null_count", descending=True)
            .with_columns(
                (pl.col("null_count") / height_expr * 100)
                .round(2)
                .alias("null_percentage")
            )
            .lazy()
        )
    return (make_null_count_lf,)


@app.cell
def _(pl):
    def get_lf_shape_summary(lf: pl.LazyFrame) -> pl.DataFrame:
        """
        Generate a summary DataFrame with total rows and columns from a LazyFrame.

        Args:
            lf (pl.LazyFrame): Input LazyFrame.

        Returns:
            pl.DataFrame: A DataFrame with two rows:
                          - metric: 'Total de observaciones' or 'Total de columnas'
                          - value: corresponding counts
        """
        total_rows = lf.select(pl.len()).collect().item()
        total_cols = len(lf.collect_schema())

        return pl.DataFrame(
            {
                "metric": ["Total de observaciones", "Total de columnas"],
                "value": [total_rows, total_cols],
            }
        )
    return (get_lf_shape_summary,)


@app.cell
def _(
    GT,
    STYLE_PRESETS: "Dict[StylePresetKey, StylePreset]",
    StylePresetKey,
    make_null_count_lf,
    md,
    pl,
):
    def gt_null_counts(
        lf: pl.LazyFrame,
        title: str = "**Valores nulos en el `dataset`**",
        subtitle: str = "Total de nulos con su respectivo porcentaje por cada variable",
        metric_label: str = "Variable",
        null_count_label: str = "Conteo de nulos",
        null_percentage_label: str = "Porcentaje de nulos",
        null_count_decimals: int = 0,
        null_percentage_scale_values: bool = False,
        null_percentage_drop_trailing_zeros: bool = True,
        null_percentage_palette: list[str] = None,
        style: StylePresetKey = "default",
    ) -> GT:
        """
            Generate a styled Great Tables (GT) table summarizing null counts and percentages per column.

            This function calculates null value statistics for each column in the provided Polars LazyFrame,
            returning a GT table with:
              - Null value counts per column.
              - Null value percentages relative to the total row count.

            The table supports flexible customization for titles, column labels, number formatting,
            percentage formatting, color gradients, and visual style presets.

            Args:
                lf (pl.LazyFrame): Polars LazyFrame to analyze.
                title (str, optional): Markdown-formatted table title. Defaults to a descriptive Spanish title.
                subtitle (str, optional): Markdown-formatted table subtitle.
                metric_label (str, optional): Label for the variable/column name column.
                null_count_label (str, optional): Label for the null count column.
                null_percentage_label (str, optional): Label for the null percentage column.
                null_count_decimals (int, optional): Number of decimals for null count formatting.
                null_percentage_scale_values (bool, optional): Whether to scale percentage values before formatting.
                null_percentage_drop_trailing_zeros (bool, optional): Whether to drop trailing zeros in percentages.
                null_percentage_palette (list[str], optional): Color gradient palette for the null percentage column.
                    Defaults to ["lightgreen", "lightyellow", "crimson"].
                stylet (StylePresetKey, optional): Key selecting a predefined style preset controlling
                    the table outline color and GT styling options. Defaults to "default".

            Returns:
                GT: A styled GT table presenting null counts and percentages, ready for display or export.

            Notes:
                - Internally calls `make_null_count_lf` to compute null statistics lazily.
                - Collecting the LazyFrame triggers data execution before GT rendering.
                - Conditional coloring is applied to the null percentage column based on the color palette.
                - Style presets unify table aesthetics across different usages.

            References:
                - Great Tables documentation: https://posit-dev.github.io/great-tables/articles/intro.html
        """
        if null_percentage_palette is None:
            null_percentage_palette = ["lightgreen", "lightyellow", "crimson"]

        null_stats_lf = make_null_count_lf(lf)
        null_stats_df = null_stats_lf.collect()

        preset = STYLE_PRESETS[style]

        return (
            GT(null_stats_df)
            .tab_header(title=md(title), subtitle=md(subtitle))
            .cols_label(
                variable=metric_label,
                null_count=null_count_label,
                null_percentage=null_percentage_label,
            )
            .fmt_number(columns="null_count", decimals=null_count_decimals)
            .fmt_percent(
                columns="null_percentage",
                scale_values=null_percentage_scale_values,
                drop_trailing_zeros=null_percentage_drop_trailing_zeros,
            )
            .data_color(columns="null_percentage", palette=null_percentage_palette)
            .opt_table_outline()
            .opt_stylize(style=preset["style"], color=preset["style_color"])
        )
    return (gt_null_counts,)


@app.cell
def _(
    GT,
    STYLE_PRESETS: "Dict[StylePresetKey, StylePreset]",
    StylePresetKey,
    get_lf_shape_summary,
    md,
    pl,
):
    def gt_dims(
        lf: pl.LazyFrame,
        title: str = "**Dimensiones del dataset**",
        subtitle: str = "Número total de observaciones (filas) y variables (columnas)",
        metric_label: str = "Métrica",
        value_label: str = "Valor",
        style: StylePresetKey = "default",
    ) -> GT:
        """
        Generate a styled GT table summarizing the dimensions (rows and columns) of a Polars LazyFrame.

        This function creates a GT (Great Tables) object presenting the total number of observations
        (rows) and variables (columns) of the given Polars LazyFrame. The table is customizable
        with markdown headers, column labels, and styling presets for consistent visual appearance.

        Args:
            lf (pl.LazyFrame): The Polars LazyFrame to analyze.
            title (str, optional): Markdown-formatted table title. Defaults to "**Dimensiones del dataset**".
            subtitle (str, optional): Markdown-formatted table subtitle. Defaults to
                "Número total de observaciones (filas) y variables (columnas)".
            metric_label (str, optional): Label for the metric column (e.g., "Métrica"). Defaults to "Métrica".
            value_label (str, optional): Label for the value column (e.g., "Valor"). Defaults to "Valor".
            style (StylePresetKey, optional): Key selecting a predefined style preset for
                the table outline and color.

        Returns:
            GT: A styled GT table summarizing the row and column counts of the input LazyFrame,
                ready for display or export.

        Raises:
            KeyError: If an invalid style key is provided.

        Example:
            >>> dims_summary_gt(lf, style="alert")
            # Returns a red-outlined GT table showing dataset dimensions.
        """
        preset = STYLE_PRESETS[style]

        dims_lf = get_lf_shape_summary(lf)

        return (
            GT(dims_lf)
            .tab_header(title=md(title), subtitle=md(subtitle))
            .cols_label(metric=metric_label, value=value_label)
            .opt_table_outline()
            .opt_stylize(style=preset["style"], color=preset["style_color"])
        )

    return (gt_dims,)


@app.cell
def _(pd, pl):
    def value_counts_with_proportion(
        series: pl.Series,
        count_col_alias: str = "count",
        proportion_col_alias: str = "proportion",
        cum_count_col_alias: str = "cum_count",
        cum_proportion_col_alias: str = "cum_proportion",
        sort: bool = True,
        parallel: bool = True,
        decimals: int = 2,
        convert_to_pandas: bool = False,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Compute value counts, proportions, and cumulative frequencies for a Polars Series.

        This function extends `polars.Series.value_counts()` by adding:
        - Proportions (relative frequencies) of each unique value.
        - Cumulative absolute frequencies (running sum of counts).
        - Cumulative relative frequencies (running sum of proportions).

        Parameters
        ----------
        series : pl.Series
            Input Polars Series for which to compute value counts.
        count_col_alias : str, optional (default='count')
            Name of the column containing the count of each unique value.
        proportion_col_alias : str, optional (default='proportion')
            Name of the column containing the proportion (relative frequency) of each unique value.
        cum_count_col_alias : str, optional (default='cum_count')
            Name of the column containing the cumulative absolute frequency.
        cum_proportion_col_alias : str, optional (default='cum_proportion')
            Name of the column containing the cumulative relative frequency.
        sort : bool, optional (default=True)
            If True, sort results by count in descending order.
        parallel : bool, optional (default=True)
            If True, use multithreading for faster computation (Polars parallel execution).
        decimals : int, optional (default=2)
            Number of decimal places to round the proportion columns.
        convert_to_pandas : bool, optional (default=False)
            If True, return result as a Pandas DataFrame; otherwise, return a Polars DataFrame.

        Returns
        -------
        pl.DataFrame | pd.DataFrame
            A DataFrame with columns for:
            - Unique values (`value`),
            - Counts (`count_col_alias`),
            - Proportions (`proportion_col_alias`),
            - Cumulative counts (`cum_count_col_alias`),
            - Cumulative proportions (`cum_proportion_col_alias`).
            Return type depends on `convert_to_pandas`.

        Examples
        --------
        >>> import polars as pl
        >>> s = pl.Series(["a", "b", "a", "c", "b", "a"])
        >>> value_counts_with_proportion(s)
        shape: (3, 5)
        ┌──────┬───────┬────────────┬───────────┬──────────────────┐
        │ value │ count │ proportion │ cum_count │ cum_proportion   │
        │ ---   │ ---   │ ---        │ ---       │ ---              │
        │ str   │ u32   │ f64        │ u32       │ f64              │
        ╞══════╪═══════╪════════════╪═══════════╪══════════════════╡
        │ "a"   │ 3     │ 0.5        │ 3         │ 0.5              │
        │ "b"   │ 2     │ 0.33       │ 5         │ 0.83             │
        │ "c"   │ 1     │ 0.17       │ 6         │ 1.0              │
        └──────┴───────┴────────────┴───────────┴──────────────────┘
        """
        total_count = series.len()

        result = series.value_counts(
            name=count_col_alias, sort=sort, parallel=parallel
        ).with_columns(
            (pl.col(count_col_alias) / total_count)
            .round(decimals)
            .alias(proportion_col_alias),
            pl.col(count_col_alias).cum_sum().alias(cum_count_col_alias),
            (pl.col(count_col_alias).cum_sum() / total_count)
            .round(decimals)
            .alias(cum_proportion_col_alias),
        )

        return result.to_pandas() if convert_to_pandas else result
    return


@app.cell
def _(Dict, Optional, pl, re):
    class LazyCSVLoader:
        """
        Lazy CSV loader using Polars with optional schema typing, decimal comma support,
        and snake_case renaming.

        Args:
            path (str): Path to CSV file.
            separator (str, optional): CSV delimiter (default ',').
            dtypes (Optional[Dict[str, pl.DataType]]): Optional schema mapping column names to Polars types.
            use_decimal_comma (bool, optional): If True, parses decimals with comma (e.g. "3,14" as 3.14).

        Methods:
            fetch_lf() -> pl.LazyFrame:
                Returns Polars LazyFrame with optional schema types and decimal comma support.
            fetch_schema() -> pl.Schema:
                Returns schema of CSV.
            fetch_col_names() -> list[str]:
                Returns list of column names.
            to_snake_case(name: str) -> str:
                Converts string to snake_case (static method).
            fetch_lf_renamed() -> pl.LazyFrame:
                Returns LazyFrame with columns renamed to snake_case.
        """
        def __init__(
            self, 
            path: str, 
            separator: str = ',', 
            dtypes: Optional[Dict[str, pl.DataType]] = None,
            use_decimal_comma: bool = False,
        ):
            self.path = path
            self.separator = separator
            self.dtypes = dtypes
            self.use_decimal_comma = use_decimal_comma
    
        def fetch_lf(self) -> pl.LazyFrame:
            """Return a Polars LazyFrame for the CSV, applying dtypes and decimal comma if provided."""
            return pl.scan_csv(
                self.path,
                separator=self.separator,
                schema_overrides=self.dtypes,
                decimal_comma=self.use_decimal_comma,
            )
    
        def fetch_schema(self) -> pl.Schema:
            """Return the schema of the CSV as a Polars Schema object."""
            return self.fetch_lf().collect_schema()
    
        def fetch_col_names(self) -> list[str]:
            """Return a list of column names from the CSV schema."""
            return self.fetch_schema().names()

        @staticmethod
        def to_snake_case(name: str) -> str:
            """Convert a string to snake_case."""
            return re.sub(r"[\s\-]+", "_", name.strip().lower())
    
        def fetch_lf_renamed(self) -> pl.LazyFrame:
            """Return a LazyFrame with columns renamed to snake_case."""
            lf = self.fetch_lf()
            rename_map = {col: self.to_snake_case(col) for col in self.fetch_col_names()}
            return lf.rename(rename_map)
    return (LazyCSVLoader,)


@app.cell
def _(mo):
    mo.md(r"""## Carga de Datos""")
    return


@app.cell
def _(LOCAL_PATH_TO_SALES_CSV, LazyCSVLoader):
    lazy_loader = LazyCSVLoader(
        path=LOCAL_PATH_TO_SALES_CSV, use_decimal_comma=True, separator=";"
    )

    sales_lf = lazy_loader.fetch_lf_renamed()

    sales_lf.head(n=10).collect()
    return (sales_lf,)


@app.cell
def _(mo):
    mo.md(r"""## EDA - Analisis Exploratorio de Datos""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Observaciones y Variables""")
    return


@app.cell
def _(gt_dims, sales_lf):
    gt_dims(
        lf=sales_lf,
        title="**Dimensión del dataset de ventas**",
        style="forest",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Valores Nulos""")
    return


@app.cell
def _(gt_null_counts, sales_lf):
    gt_null_counts(
        lf=sales_lf,
        title="**Valores nulos en el dataset de ventas**",
        style="forest",
    )
    return


if __name__ == "__main__":
    app.run()
