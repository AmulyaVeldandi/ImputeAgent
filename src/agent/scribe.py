import json, pandas as pd

class Scribe:
    def render_report(self, summary_df: pd.DataFrame, best_policy: dict) -> str:
        lines = []
        lines.append("# Impute-Agent Report")
        lines.append("")
        lines.append("## Best Policy")
        lines.append("```json")
        lines.append(json.dumps(best_policy, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("## Summary Rows (CSV)")
        lines.append(summary_df.to_csv(index=False))
        return "\n".join(lines)
