import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# AI Integration Libraries
try:
    from google import genai
    from google.genai import types

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import ollama

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class DataAnalyzerEngine:
    """Core engine responsible for data parsing, auditing, statistics, pattern discovery, and plotting."""

    def __init__(
        self,
        title="Custom Study",
        unit_col="Observational_Unit",
        qual_cols=None,
        quant_cols=None,
    ):
        self.title = title
        self.unit_col = unit_col
        self.qual_cols = qual_cols or []
        self.quant_cols = quant_cols or []
        self.df = pd.DataFrame()

        # Flags for data availability
        self.has_qualitative = False
        self.has_quantitative = False

    def set_data(self, df):
        """Clean data, audit data types, and check for missing qualitative/quantitative fields."""
        self.df = df.copy()

        # Clean string/qualitative columns
        if self.unit_col in self.df.columns:
            self.df[self.unit_col] = (
                self.df[self.unit_col].astype(str).str.strip()
            )

        valid_qual = []
        for col in self.qual_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str).str.strip().replace("", "N/A")
                )
                valid_qual.append(col)
        self.qual_cols = valid_qual

        # Coerce quantitative columns to numeric
        valid_quant = []
        for col in self.quant_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                # Retain column if at least one valid number exists
                if not self.df[col].dropna().empty:
                    valid_quant.append(col)
        self.quant_cols = valid_quant

        # Drop rows missing crucial observational unit IDs
        if self.unit_col in self.df.columns:
            self.df = self.df.dropna(subset=[self.unit_col]).reset_index(
                drop=True
            )

        # Audit Data Availability
        self.has_qualitative = len(self.qual_cols) > 0
        self.has_quantitative = len(self.quant_cols) > 0

    def run_data_audit(self):
        """Audits the dataset and displays explicit availability alerts."""
        print("\n" + "=" * 65)
        print(" DATA AVAILABILITY AUDIT & SYSTEM STATUS")
        print("=" * 65)
        print(f"Total Observational Units Recorded: {len(self.df)}")

        # Audit Qualitative Status
        if self.has_qualitative:
            print(
                f"✓ Qualitative Data: AVAILABLE ({len(self.qual_cols)} field(s): {', '.join(self.qual_cols)})"
            )
        else:
            print(
                "❌ Qualitative Data: NOT AVAILABLE (No categorical attributes provided or recognized)"
            )

        # Audit Quantitative Status
        if self.has_quantitative:
            print(
                f"✓ Quantitative Data: AVAILABLE ({len(self.quant_cols)} metric(s): {', '.join(self.quant_cols)})"
            )
        else:
            print(
                "❌ Quantitative Data: NOT AVAILABLE (No numeric metrics provided or successfully parsed)"
            )

    def prepare_ai_context(self):
        """Prepares a structured summary payload of the dataset for AI context injection."""
        data_summary = f"Study Title: {self.title}\n"
        data_summary += f"Total Observations: {len(self.df)}\n"
        data_summary += (
            f"Attributes: {', '.join(self.qual_cols + self.quant_cols)}\n\n"
        )

        data_summary += "--- RAW DATA SAMPLE ---\n"
        data_summary += self.df.to_string(index=False) + "\n\n"

        if self.has_quantitative:
            data_summary += "--- SUMMARY STATISTICS ---\n"
            stats = self.df[self.quant_cols].describe().T
            data_summary += stats.to_string() + "\n\n"

            if len(self.quant_cols) >= 2:
                data_summary += "--- CORRELATION MATRIX ---\n"
                data_summary += (
                    self.df[self.quant_cols].corr().to_string() + "\n\n"
                )

        return data_summary

    def run_ai_pattern_analysis(self):
        """Injects dataset information into an AI LLM (Gemini or Ollama) for pattern recognition."""
        print("\n" + "=" * 65)
        print(" AI PATTERN ANALYSIS & INSIGHT ENGINE")
        print("=" * 65)

        print("Select AI Provider:")
        print("  1. Google Gemini API (Requires GEMINI_API_KEY)(To access your Gemini API key, "
              "go to Google AI Studio (aistudio.google.com), sign in with your Google account, and click the \"Get API key\""
              " button in the sidebar. Once generated, simply copy your key.)")
        print("  2. Local Ollama Model (Requires local Ollama server running)(Not available)")

        provider = input("Choice (1 or 2): ").strip()

        context_payload = self.prepare_ai_context()
        system_instruction = input("What would you like to find out about this data?")
        prompt = f"{system_instruction}\n\nDATASET PAYLOAD:\n{context_payload}"

        if provider == "1":
            if not HAS_GEMINI:
                print(
                    "\n[Error] `google-genai` library is not installed. Run `pip install google-genai`."
                )
                return

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                api_key = input("Enter your Gemini API Key: ").strip()

            if not api_key:
                print("\n[Error] API key is required to run Gemini analysis.")
                return

            print("\nQuerying Gemini AI for dataset patterns...")
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=prompt,
                )
                print("\n--- GEMINI PATTERN ANALYSIS REPORT ---")
                print(response.text)
            except Exception as e:
                print(f"\n[AI Error] Failed to generate response: {e}")

        elif provider == "2":
            if not HAS_OLLAMA:
                print(
                    "\n[Error] `ollama` library is not installed. Run `pip install ollama`."
                )
                return

            model_name = (
                input(
                    "Enter local Ollama model name (default 'llama3'): "
                ).strip()
                or "llama3"
            )
            print(f"\nQuerying local model '{model_name}' via Ollama...")

            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                print("\n--- LOCAL AI PATTERN ANALYSIS REPORT ---")
                print(response["message"]["content"])
            except Exception as e:
                print(f"\n[AI Error] Failed to connect to Ollama: {e}")

    def generate_statistical_report(self):
        """Outputs comprehensive summary statistics (Mean, Median, Mode, MAD, Variance, Std Dev, etc.)."""
        print("\n" + "=" * 65)
        print(" DESCRIPTIVE STATISTICAL REPORT")
        print("=" * 65)

        if self.has_quantitative:
            print("\n--- Numerical Summary Metrics ---")
            stats_list = []

            for col in self.quant_cols:
                series = self.df[col].dropna()
                if series.empty:
                    continue

                mean_val = series.mean()
                median_val = series.median()
                modes = series.mode()
                mode_val = modes.iloc[0] if not modes.empty else np.nan
                mad_val = (series - median_val).abs().median()
                std_val = series.std()
                var_val = series.var()
                min_val = series.min()
                max_val = series.max()
                range_val = max_val - min_val

                stats_list.append({
                    "Metric": col,
                    "Mean": round(mean_val, 2),
                    "Median": round(median_val, 2),
                    "Mode": round(mode_val, 2) if pd.notnull(mode_val) else "N/A",
                    "MAD": round(mad_val, 2),
                    "Variance": round(var_val, 2),
                    "Std Dev": round(std_val, 2),
                    "Min": round(min_val, 2),
                    "Max": round(max_val, 2),
                    "Range": round(range_val, 2),
                })

            summary_df = pd.DataFrame(stats_list).set_index("Metric")
            print(summary_df.to_string())
        else:
            print(
                "\n[Notice] Numerical summary skipped: No quantitative data available."
            )

        if self.has_qualitative and self.has_quantitative:
            print(
                f"\n--- Grouped Averages by Primary Category ({self.qual_cols[0]}) ---"
            )
            grouped = self.df.groupby(self.qual_cols[0])[
                self.quant_cols
            ].mean()
            print(grouped.round(2).to_string())
        elif self.has_qualitative and not self.has_quantitative:
            print(
                f"\n--- Category Frequency Counts ({self.qual_cols[0]}) ---"
            )
            print(self.df[self.qual_cols[0]].value_counts().to_string())

    def pattern_discovery_engine(self):
        """Scans dataset for patterns based on available data types."""
        print("\n" + "=" * 65)
        print(" AUTOMATED PATTERN DISCOVERY ENGINE")
        print("=" * 65)

        if not self.has_quantitative and not self.has_qualitative:
            print(
                "\n[Insights] Insufficient data available to run pattern discovery."
            )
            return

        # A. Correlation Scanning
        if self.has_quantitative and len(self.quant_cols) >= 2:
            print("\n[Insights] Variable Relationships (Correlations):")
            corr_matrix = self.df[self.quant_cols].corr()
            found_corr = False
            for i in range(len(self.quant_cols)):
                for j in range(i + 1, len(self.quant_cols)):
                    val1, val2 = self.quant_cols[i], self.quant_cols[j]
                    r = corr_matrix.loc[val1, val2]
                    if abs(r) >= 0.5:
                        strength = "Strong" if abs(r) >= 0.7 else "Moderate"
                        direction = "positive" if r > 0 else "negative"
                        print(
                            f"  • {strength} {direction} correlation between '{val1}' and '{val2}' (r = {r:.2f})."
                        )
                        found_corr = True
            if not found_corr:
                print("  • No strong linear correlations detected.")
        elif not self.has_quantitative:
            print(
                "\n[Insights] Correlation scan skipped: No quantitative data available."
            )

        # B. Outlier Detection
        if self.has_quantitative:
            print("\n[Insights] Anomaly & Outlier Detection:")
            found_outliers = False
            for col in self.quant_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    z_scores = (self.df[col] - mean) / std
                    outliers = self.df[abs(z_scores) >= 2.0]
                    for _, row in outliers.iterrows():
                        val = row[col]
                        unit = row[self.unit_col]
                        diff = "above" if val > mean else "below"
                        print(
                            f"  • Outlier in '{col}': Unit '{unit}' with value {val:,.2f} ({abs((val-mean)/std):.1f} std devs {diff} avg)."
                        )
                        found_outliers = True
            if not found_outliers:
                print("  • No significant outliers detected (within ±2 std devs).")

        # C. Category Performance Extremes
        if self.has_qualitative and self.has_quantitative:
            print(
                f"\n[Insights] Category Extremes for '{self.quant_cols[0]}':"
            )
            primary_q = self.quant_cols[0]
            primary_cat = self.qual_cols[0]
            cat_means = self.df.groupby(primary_cat)[primary_q].mean()

            if not cat_means.empty:
                print(
                    f"  • Highest Performing: '{cat_means.idxmax()}' (Avg: {cat_means.max():,.2f})"
                )
                print(
                    f"  • Lowest Performing:  '{cat_means.idxmin()}' (Avg: {cat_means.min():,.2f})"
                )

    def _select_qual_col(self, prompt_label="Select Qualitative Variable"):
        """Helper to let the user select a qualitative column from available options."""
        options = self.qual_cols if self.has_qualitative else [self.unit_col]
        if len(options) == 1:
            return options[0]

        print(f"\n{prompt_label}:")
        for idx, col in enumerate(options, 1):
            print(f"  {idx}. {col}")

        while True:
            choice = input(f"Choice (1-{len(options)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            print("Invalid choice! Try again.")

    def _select_quant_col(self, prompt_label="Select Quantitative Variable"):
        """Helper to let the user select a quantitative column from available options."""
        if len(self.quant_cols) == 1:
            return self.quant_cols[0]

        print(f"\n{prompt_label}:")
        for idx, col in enumerate(self.quant_cols, 1):
            print(f"  {idx}. {col}")

        while True:
            choice = input(f"Choice (1-{len(self.quant_cols)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.quant_cols):
                return self.quant_cols[int(choice) - 1]
            print("Invalid choice! Try again.")

    def render_plot(self, chart_choice):
        """Renders graphs or a graphical table chart based on availability."""
        sns.set_theme(style="whitegrid")

        # OPTION 6: GRAPHICAL DATA TABLE CHART TYPE
        if chart_choice == "6":
            fig, ax = plt.subplots(
                figsize=(10, max(3, len(self.df) * 0.4 + 1.5))
            )
            ax.axis("tight")
            ax.axis("off")

            # Create table representation
            table_data = self.df.copy()
            # Format numbers for clean visual table rendering
            for col in self.quant_cols:
                table_data[col] = table_data[col].apply(
                    lambda x: (
                        f"{x:,.2f}"
                        if pd.notnull(x)
                        else "N/A (No Quantitative Data)"
                    )
                )

            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Style header row
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#2b5c8f")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#f2f4f7")

            plt.title(
                f"{self.title}: Structured Information Table Chart",
                fontsize=13,
                fontweight="bold",
                pad=20,
            )

        # STANDARD CHARTS (Require specific data types)
        else:
            if not self.has_quantitative and chart_choice in [
                "1",
                "2",
                "3",
                "4",
                "5",
            ]:
                print(
                    "\n[Error Cannot Render Chart] Selected chart type requires quantitative data, which is missing."
                )
                return
            if not self.has_qualitative and chart_choice in ["1", "3", "4"]:
                print(
                    "\n[Error Cannot Render Chart] Selected chart type requires qualitative categorical data, which is missing."
                )
                return

            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_choice == "1":
                x_cat = self._select_qual_col("Select Qualitative Variable for X-axis")
                y_num = self._select_quant_col("Select Quantitative Variable for Y-axis")
                hue_cat = self.qual_cols[1] if len(self.qual_cols) > 1 else None

                sns.barplot(
                    data=self.df,
                    x=x_cat,
                    y=y_num,
                    hue=hue_cat,
                    palette="viridis",
                    ax=ax,
                )
                plt.title(
                    f"{self.title}: {y_num} by {x_cat}",
                    fontsize=13,
                    fontweight="bold",
                )
                plt.xticks(rotation=15)

            elif chart_choice == "2":
                x_num = self._select_quant_col("Select Quantitative Variable for X-axis")
                y_scatter = self._select_quant_col("Select Quantitative Variable for Y-axis")

                # Base scatter plot
                sns.scatterplot(
                    data=self.df,
                    x=x_num,
                    y=y_scatter,
                    hue=self.qual_cols[0] if self.has_qualitative else None,
                    s=120,
                    palette="bright",
                    ax=ax,
                    zorder=3,  # Keep points on top of the line
                )

                # Annotate point labels
                for _, row in self.df.iterrows():
                    ax.annotate(
                        str(row[self.unit_col]),
                        (row[x_num], row[y_scatter]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )

                # --- Line of Best Fit & Equation ---
                valid_data = self.df[[x_num, y_scatter]].dropna()

                if len(valid_data) >= 2:
                    x_vals = valid_data[x_num]
                    y_vals = valid_data[y_scatter]

                    # Fit 1st degree polynomial (slope, intercept)
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)

                    # Generate line points spanning the range of x
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    y_line = slope * x_line + intercept

                    # Plot trendline
                    ax.plot(
                        x_line,
                        y_line,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="Line of Best Fit",
                        zorder=2,
                    )

                    # Format & display equation text box
                    sign = "+" if intercept >= 0 else "-"
                    eq_text = f"y = {slope:.2f}x {sign} {abs(intercept):.2f}"

                    ax.text(
                        0.05,
                        0.92,
                        eq_text,
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="white",
                            edgecolor="red",
                            alpha=0.8,
                        ),
                    )
                    ax.legend()

                ax.set_xlabel(x_num, fontsize=11, fontweight="bold")
                ax.set_ylabel(y_scatter, fontsize=11, fontweight="bold")

                plt.title(
                    f"{self.title}: {x_num} vs {y_scatter}",
                    fontsize=13,
                    fontweight="bold",
                )

            elif chart_choice == "3":
                x_cat = self._select_qual_col("Select Qualitative Variable for X-axis")
                y_num = self._select_quant_col("Select Quantitative Variable for Y-axis")
                hue_cat = self.qual_cols[1] if len(self.qual_cols) > 1 else None

                sns.stripplot(
                    data=self.df,
                    x=x_cat,
                    y=y_num,
                    hue=hue_cat,
                    jitter=0.2,
                    size=10,
                    palette="dark",
                    ax=ax,
                )
                plt.title(
                    f"{self.title}: Dot Distribution of {y_num} by {x_cat}",
                    fontsize=13,
                    fontweight="bold",
                )
                plt.xticks(rotation=15)

            elif chart_choice == "4":
                x_cat = self._select_qual_col("Select Categorical Variable for Segments")
                y_num = self._select_quant_col("Select Quantitative Variable for Value/Share")

                pie_data = self.df.groupby(x_cat)[y_num].sum()
                colors = sns.color_palette("Set2", len(pie_data))
                ax.pie(
                    pie_data,
                    labels=pie_data.index,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=colors,
                    pctdistance=0.75,
                    wedgeprops={"edgecolor": "white", "linewidth": 2},
                )
                centre_circle = plt.Circle((0, 0), 0.50, fc="white")
                fig.gca().add_artist(centre_circle)
                ax.axis("equal")
                plt.title(
                    f"{self.title}: Share of {y_num} by {x_cat}",
                    fontsize=13,
                    fontweight="bold",
                )

            elif chart_choice == "5":
                plt.close(fig)
                numeric_df = self.df[self.quant_cols]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    numeric_df.corr(),
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    ax=ax,
                )
                plt.title(
                    f"{self.title}: Multi-Variable Correlation Heatmap",
                    fontsize=13,
                    fontweight="bold",
                )

        plt.tight_layout()

        # Save Option
        save = (
            input("\nSave this chart as high-res PNG? (y/n): ").strip().lower()
        )
        if save == "y":
            fname = (
                input("Enter image filename (default: chart.png): ").strip()
                or "chart.png"
            )
            if not fname.endswith(".png"):
                fname += ".png"
            plt.savefig(fname, dpi=300)
            print(f"Chart saved as '{fname}'.")

        plt.show()


# --- TERMINAL INPUT ENGINE ---


def setup_interactive_schema():
    print("=" * 65)
    print(" ADVANCED DATA COLLECTION & PATTERN DISCOVERY SYSTEM")
    print("=" * 65)

    title = (
        input("Enter Study Title (Press Enter for 'Dataset Study'): ").strip()
        or "Dataset Study"
    )
    unit_col = (
        input(
            "Name of Observational Unit ID (e.g., Item, Store, Subject): "
        ).strip()
        or "Unit_ID"
    )

    qual_cols = []
    print("\n--- Categorical / Qualitative Columns ---")
    print("(Press Enter on the first prompt if NO qualitative data exists)")
    while True:
        col = input(
            f"Enter Category Column #{len(qual_cols)+1} (or press Enter to finish): "
        ).strip()
        if not col:
            break
        qual_cols.append(col)

    quant_cols = []
    print("\n--- Numerical / Quantitative Columns ---")
    print("(Press Enter on the first prompt if NO quantitative data exists)")
    while True:
        col = input(
            f"Enter Numeric Metric #{len(quant_cols)+1} (or press Enter to finish)(Recommended to put units in parentheses, for example: \"Height(cm)\": "
        ).strip()
        if not col:
            break
        quant_cols.append(col)

    return title, unit_col, qual_cols, quant_cols


def collect_dataset(unit_col, qual_cols, quant_cols):
    print("\nChoose Data Collection Method:")
    print("  1. Paste Tab-Delimited Block (Copied from Excel/Google Sheets)")
    print("  2. Enter Data Row-by-Row Interactively")

    while True:
        choice = input("Option (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice! Enter 1 or 2.")

    data = []

    if choice == "1":
        print("\nPaste raw data block below. Do NOT include header row.")
        print("Press Enter twice when finished pasting:")

        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        if lines:
            rows = [l.split("\t") for l in lines if l.strip()]
            for r in rows:
                if len(r) >= 1:
                    row_dict = {unit_col: r[0].strip()}
                    idx = 1
                    for q in qual_cols:
                        row_dict[q] = (
                            r[idx].strip() if idx < len(r) else "N/A"
                        )
                        idx += 1
                    for qn in quant_cols:
                        if idx < len(r):
                            try:
                                row_dict[qn] = float(r[idx].strip())
                            except ValueError:
                                row_dict[qn] = np.nan
                        else:
                            row_dict[qn] = np.nan
                        idx += 1
                    data.append(row_dict)

    else:
        print("\n--- Interactive Row-by-Row Entry ---")
        row_idx = 1
        while True:
            unit_val = input(
                f"\nRow #{row_idx} - {unit_col} Identifier (or 'done' to complete): "
            ).strip()
            if unit_val.lower() == "done":
                if not data:
                    print("You must enter at least one observational unit!")
                    continue
                break

            row_dict = {unit_col: unit_val}
            for q in qual_cols:
                val = input(f"  [{q}] (Category): ").strip()
                row_dict[q] = val if val else "N/A"

            for qn in quant_cols:
                val_str = input(
                    f"  [{qn}] (Numeric Value - press Enter if N/A): "
                ).strip()
                try:
                    row_dict[qn] = float(val_str)
                except ValueError:
                    row_dict[qn] = np.nan

            data.append(row_dict)
            row_idx += 1

    return pd.DataFrame(data)


def main():
    title, unit_col, qual_cols, quant_cols = setup_interactive_schema()
    df = collect_dataset(unit_col, qual_cols, quant_cols)

    if df.empty:
        print("\n[Error] No valid data collected. Program exiting.")
        sys.exit()

    engine = DataAnalyzerEngine(title, unit_col, qual_cols, quant_cols)
    engine.set_data(df)

    # 1. Audit Data Availability
    engine.run_data_audit()

    # 2. Run Analysis & Pattern Discovery
    engine.generate_statistical_report()
    engine.pattern_discovery_engine()

    # Interactive Menu Loop
    while True:
        print("\n" + "=" * 50)
        print(" CHART & TABLE VISUALIZATION MENU")
        print("=" * 50)
        print(
            "  1. Bar Plot / Grouped Bar Chart (Requires Qualitative & Quantitative)"
        )
        print("  2. Scatter Plot (Requires Quantitative)")
        print(
            "  3. Dot Plot / Strip Plot (Requires Qualitative & Quantitative)"
        )
        print("  4. Pie / Donut Chart (Requires Qualitative & Quantitative)")
        print("  5. Multi-Variable Correlation Heatmap (Requires Quantitative)")
        print(
            "  6. Render Data Table Chart (Organizes all units regardless of data presence)"
        )
        print("  7. AI Overview")
        print("  8. Exit Program")

        choice = input("\nSelect Chart Type (1-8): ").strip()
        if choice == "8":
            print("\nAnalysis session completed. Goodbye!")
            break
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            engine.render_plot(choice)
        elif choice == "7":
            engine.run_ai_pattern_analysis()
        else:
            print("Invalid selection! Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main()
