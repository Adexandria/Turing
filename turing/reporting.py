from datetime import datetime
import platform
import sys
from typing import Optional

from loguru import logger
import pandas as pd

from turing.config import REPORTS_DIR


class TestReportGenerator:
    """
    Handles the generation of structured Markdown reports specifically for test execution results.
    """

    def __init__(self, context_name: str, report_category: str):
        self.context_name = context_name
        self.report_category = report_category
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.content = []
        self.output_dir = REPORTS_DIR / self.report_category

    def add_header(self, text: str, level: int = 1):
        self.content.append(f"\n{'#' * level} {text}\n")

    def add_divider(self, style: str = "thin"):
        """Add a visual divider line."""
        dividers = {
            "thin": "---",
            "thick": "___",
            "section": "\n---\n",
        }
        self.content.append(f"\n{dividers.get(style, dividers['thin'])}\n")

    def add_code_block(self, content: str, language: str = ""):
        """Add a code block."""
        self.content.append(f"\n```{language}\n{content}\n```\n")

    def add_alert_box(self, message: str, box_type: str = "info"):
        """Add a styled alert box using blockquotes."""
        box_headers = {
            "info": "INFO",
            "success": "SUCCESS",
            "warning": "WARNING",
            "error": "ERROR",
        }
        header = box_headers.get(box_type, "INFO")
        self.content.append(f"\n> **{header}**: {message}\n")

    def add_progress_bar(self, passed: int, total: int, width: int = 50):
        """Add an ASCII progress bar."""
        if total == 0:
            percentage = 0
            filled = 0
        else:
            percentage = (passed / total * 100)
            filled = int(width * passed / total)
        
        empty = width - filled
        bar = "█" * filled + "░" * empty
        self.add_code_block(f"Progress: [{bar}] {percentage:.1f}%\nPassed: {passed}/{total} tests", "")

    def add_summary_box(self, total: int, passed: int, failed: int, skipped: int = 0):
        """Add a visually enhanced summary box."""
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Determine status
        if success_rate == 100:
            status = "ALL TESTS PASSED"
        elif success_rate >= 80:
            status = "MOSTLY PASSED"
        elif success_rate >= 50:
            status = "PARTIAL SUCCESS"
        else:
            status = "NEEDS ATTENTION"
        
        self.add_header("Executive Summary", level=2)
        self.add_text(f"**Overall Status:** {status}")
        self.add_text(f"**Success Rate:** {success_rate:.1f}%")
        
        # Summary table
        summary_data = [
            ["Total Tests", str(total)],
            ["Passed", str(passed)],
            ["Failed", str(failed)],
        ]
        
        if skipped > 0:
            summary_data.append(["Skipped", str(skipped)])
        
        summary_data.append(["Success Rate", f"{success_rate:.1f}%"])
        
        df = pd.DataFrame(summary_data, columns=["Metric", "Count"])
        self.add_dataframe(df, title=None, align=("left", "right"))
        
        # Progress bar
        self.add_text("**Visual Progress:**")
        self.add_progress_bar(passed, total)

    def add_environment_metadata(self):
        """Add enhanced environment metadata."""
        self.add_header("Environment Information", level=2)
        
        metadata = [
            ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Context", self.context_name.upper()],
            ["Python Version", sys.version.split()[0]],
            ["Platform", platform.platform()],
            ["Architecture", platform.machine()],
        ]
        df = pd.DataFrame(metadata, columns=["Parameter", "Value"])
        self.add_dataframe(df, title=None, align=("left", "left"))

    def add_text(self, text: str):
        self.content.append(f"\n{text}\n")

    def add_category_stats(self, df: pd.DataFrame, category: str):
        """Add statistics for a test category."""
        total = len(df)
        passed = len(df[df['Result'] == "PASS"])
        failed = len(df[df['Result'] == "FAIL"])
        skipped = len(df[df['Result'] == "SKIP"])
        
        stats = [
            ["Total", str(total)],
            ["Passed", f"{passed} ({passed/total*100:.1f}%)" if total > 0 else "0"],
            ["Failed", f"{failed} ({failed/total*100:.1f}%)" if total > 0 else "0"],
        ]
        
        if skipped > 0:
            stats.append(["Skipped", f"{skipped} ({skipped/total*100:.1f}%)"])
        
        stats_df = pd.DataFrame(stats, columns=["Status", "Count"])
        self.add_dataframe(stats_df, title="Statistics", align=("left", "right"))

    def add_dataframe(self, df: pd.DataFrame, title: Optional[str] = None, align: tuple = None):
        """Add a formatted dataframe table."""
        if title:
            self.add_header(title, level=3)

        if df.empty:
            self.content.append("\n_No data available._\n")
            return

        try:
            if not align:
                align = tuple(["left"] * len(df.columns))

            table_md = df.to_markdown(index=False, tablefmt="pipe", colalign=align)
            self.content.append(f"\n{table_md}\n")
        except Exception as e:
            logger.warning(f"Tabulate error: {e}. Using simple text.")
            self.content.append(f"\n```text\n{df.to_string(index=False)}\n```\n")

    def save(self, filename: str = "test_report.md") -> str:
        """Save the report to a file."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.output_dir / filename
            
            # Add footer
            self.add_divider("section")
            self.add_text(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
            self.add_text("*Powered by Turing Test Suite*")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.content))
            logger.info(f"Test report saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise
