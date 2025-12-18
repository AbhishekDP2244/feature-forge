"""
HTML report generator for feature engineering results
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns


class ReportGenerator:
    """
    Generate comprehensive HTML reports for feature engineering

    Parameters
    ----------
    X_original : pd.DataFrame
        Original feature matrix
    X_augmented : pd.DataFrame
        Augmented feature matrix with generated features
    y : pd.Series
        Target variable
    generated_features : dict
        Dictionary of generated features by strategy
    feature_scores : dict, optional
        Feature importance scores
    task : str
        Task type ('classification' or 'regression')
    """

    def __init__(
        self,
        X_original: pd.DataFrame,
        X_augmented: pd.DataFrame,
        y: pd.Series,
        generated_features: Dict,
        feature_scores: Optional[Dict] = None,
        task: str = "classification",
    ):
        self.X_original = X_original
        self.X_augmented = X_augmented
        self.y = y
        self.generated_features = generated_features
        self.feature_scores = feature_scores or {}
        self.task = task

    def create_html_report(self, output_path: str):
        """
        Create comprehensive HTML report

        Parameters
        ----------
        output_path : str
            Path to save HTML report
        """
        html = self._generate_html()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _generate_html(self) -> str:
        """
        Generate full HTML report

        Returns
        -------
        str
            Complete HTML content
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FeatureSmith Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 32px;
                    font-weight: bold;
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .feature-list {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .feature-tag {{
                    display: inline-block;
                    background-color: #3498db;
                    color: white;
                    padding: 5px 10px;
                    margin: 5px;
                    border-radius: 3px;
                    font-size: 12px;
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔥 FeatureSmith Report</h1>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Task Type:</strong> {self.task.capitalize()}</p>
                
                {self._generate_summary_section()}
                {self._generate_features_section()}
                {self._generate_importance_section()}
                {self._generate_stats_section()}
            </div>
        </body>
        </html>
        """
        return html

    def _generate_summary_section(self) -> str:
        """Generate summary metrics section"""
        n_original = len(self.X_original.columns)
        n_augmented = len(self.X_augmented.columns)
        n_generated = n_augmented - n_original
        n_samples = len(self.X_original)

        return f"""
        <h2>📊 Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{n_original}</div>
                <div class="metric-label">Original Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{n_generated}</div>
                <div class="metric-label">Generated Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{n_augmented}</div>
                <div class="metric-label">Total Features</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{n_samples}</div>
                <div class="metric-label">Samples</div>
            </div>
        </div>
        """

    def _generate_features_section(self) -> str:
        """Generate generated features section"""
        if not self.generated_features:
            return "<h2>Generated Features</h2><p>No features were generated.</p>"

        html = "<h2>🎯 Generated Features by Strategy</h2>"

        for strategy, features in self.generated_features.items():
            html += f"""
            <h3>{strategy.capitalize()}</h3>
            <div class="feature-list">
                <p><strong>Count:</strong> {len(features)} features</p>
                <div>
                    {''.join([f'<span class="feature-tag">{feat}</span>' for feat in features[:20]])}
                    {f'<span class="feature-tag">... and {len(features) - 20} more</span>' if len(features) > 20 else ''}
                </div>
            </div>
            """

        return html

    def _generate_importance_section(self) -> str:
        """Generate feature importance section"""
        if not self.feature_scores:
            return ""

        # Sort by importance
        sorted_features = sorted(
            self.feature_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_20 = sorted_features[:20]

        # Create importance chart
        chart_html = self._create_importance_chart(top_20)

        # Create table
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance Score</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
        """

        for i, (feat, score) in enumerate(top_20, 1):
            feat_type = self._get_feature_type(feat)
            table_html += f"""
            <tr>
                <td>{i}</td>
                <td>{feat}</td>
                <td>{score:.4f}</td>
                <td>{feat_type}</td>
            </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return f"""
        <h2>⭐ Feature Importance</h2>
        <div class="chart">
            {chart_html}
        </div>
        <h3>Top 20 Features</h3>
        {table_html}
        """

    def _generate_stats_section(self) -> str:
        """Generate statistics section"""
        # Original features stats
        orig_numeric = self.X_original.select_dtypes(include=[np.number])
        orig_categorical = self.X_original.select_dtypes(include=["object", "category"])

        # Augmented features stats
        aug_numeric = self.X_augmented.select_dtypes(include=[np.number])
        aug_categorical = self.X_augmented.select_dtypes(include=["object", "category"])

        return f"""
        <h2>📈 Dataset Statistics</h2>
        
        <h3>Original Features</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Numeric Features</td>
                <td>{len(orig_numeric.columns)}</td>
            </tr>
            <tr>
                <td>Categorical Features</td>
                <td>{len(orig_categorical.columns)}</td>
            </tr>
            <tr>
                <td>Missing Values</td>
                <td>{self.X_original.isnull().sum().sum()} ({(self.X_original.isnull().sum().sum() / self.X_original.size * 100):.2f}%)</td>
            </tr>
        </table>
        
        <h3>Augmented Features</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Numeric Features</td>
                <td>{len(aug_numeric.columns)}</td>
            </tr>
            <tr>
                <td>Categorical Features</td>
                <td>{len(aug_categorical.columns)}</td>
            </tr>
            <tr>
                <td>Missing Values</td>
                <td>{self.X_augmented.isnull().sum().sum()} ({(self.X_augmented.isnull().sum().sum() / self.X_augmented.size * 100):.2f}%)</td>
            </tr>
        </table>
        """

    def _create_importance_chart(self, top_features):
        """Create base64 encoded importance chart"""
        features, scores = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), scores, color="#3498db")
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.title("Top 20 Features by Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f'<img src="data:image/png;base64,{image_base64}" alt="Feature Importance Chart">'

    def _get_feature_type(self, feature_name: str) -> str:
        """Determine feature type from name"""
        if feature_name in self.X_original.columns:
            return "Original"

        for strategy, features in self.generated_features.items():
            if feature_name in features:
                return strategy.capitalize()

        return "Unknown"
