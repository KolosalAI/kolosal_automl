"""
Data Preview Generator

Generates summaries and previews of datasets for the UI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

try:
    from modules.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class DataPreviewGenerator:
    """Generate data summaries and previews for UI display"""
    
    def __init__(self):
        """Initialize the data preview generator"""
        self.max_preview_rows = 10
        self.max_unique_values_display = 10
    
    def generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        try:
            summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Add statistical summary for numeric columns
            if summary['numeric_columns']:
                numeric_summary = df[summary['numeric_columns']].describe().to_dict()
                summary['numeric_summary'] = numeric_summary
            
            # Add unique value counts for categorical columns
            categorical_summary = {}
            for col in summary['categorical_columns']:
                unique_count = df[col].nunique()
                categorical_summary[col] = {
                    'unique_count': unique_count,
                    'top_values': df[col].value_counts().head(self.max_unique_values_display).to_dict()
                }
            summary['categorical_summary'] = categorical_summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {'error': str(e)}
    
    def format_data_preview(self, df: pd.DataFrame, summary: Dict[str, Any]) -> str:
        """Format data preview as HTML string"""
        try:
            if 'error' in summary:
                return f"<div class='alert alert-danger'>Error: {summary['error']}</div>"
            
            shape = summary.get('shape', (0, 0))
            missing_total = sum(summary.get('missing_values', {}).values())
            memory_mb = summary.get('memory_usage', 0) / (1024 * 1024)
            
            preview_html = f"""
            <div class='data-preview'>
                <h4>📊 Dataset Overview</h4>
                <div class='row'>
                    <div class='col-md-6'>
                        <ul class='list-unstyled'>
                            <li><strong>Shape:</strong> {shape[0]:,} rows × {shape[1]:,} columns</li>
                            <li><strong>Memory Usage:</strong> {memory_mb:.2f} MB</li>
                            <li><strong>Missing Values:</strong> {missing_total:,} total</li>
                        </ul>
                    </div>
                    <div class='col-md-6'>
                        <ul class='list-unstyled'>
                            <li><strong>Numeric Columns:</strong> {len(summary.get('numeric_columns', []))}</li>
                            <li><strong>Categorical Columns:</strong> {len(summary.get('categorical_columns', []))}</li>
                            <li><strong>DateTime Columns:</strong> {len(summary.get('datetime_columns', []))}</li>
                        </ul>
                    </div>
                </div>
                
                <h5>🔍 Sample Data (First {min(self.max_preview_rows, shape[0])} rows)</h5>
                {df.head(self.max_preview_rows).to_html(classes='table table-striped table-sm', escape=False, border=0)}
                
                <h5>📈 Column Information</h5>
                <div class='table-responsive'>
                    {self._format_column_info(summary)}
                </div>
            </div>
            """
            
            return preview_html
            
        except Exception as e:
            logger.error(f"Error formatting data preview: {e}")
            return f"<div class='alert alert-danger'>Error formatting preview: {e}</div>"
    
    def _format_column_info(self, summary: Dict[str, Any]) -> str:
        """Format column information as HTML table"""
        try:
            columns = summary.get('columns', [])
            dtypes = summary.get('dtypes', {})
            missing_values = summary.get('missing_values', {})
            
            table_html = """
            <table class='table table-striped table-sm'>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Data Type</th>
                        <th>Missing Values</th>
                        <th>Missing %</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            total_rows = summary.get('shape', (0, 0))[0]
            
            for col in columns:
                dtype = str(dtypes.get(col, 'unknown'))
                missing = missing_values.get(col, 0)
                missing_pct = (missing / total_rows * 100) if total_rows > 0 else 0
                
                table_html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td><span class='badge badge-info'>{dtype}</span></td>
                    <td>{missing:,}</td>
                    <td>{missing_pct:.1f}%</td>
                </tr>
                """
            
            table_html += """
                </tbody>
            </table>
            """
            
            return table_html
            
        except Exception as e:
            logger.error(f"Error formatting column info: {e}")
            return f"<div class='alert alert-warning'>Error formatting column information: {e}</div>"
