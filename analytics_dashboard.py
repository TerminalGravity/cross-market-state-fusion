#!/usr/bin/env python3
"""
Rich analytics dashboard for trade analysis.
Serves an interactive HTML page to explore trading data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from datetime import datetime

app = Flask(__name__)
LOGS_DIR = Path(__file__).parent / "logs"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Analytics</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-primary: #30363d;
            --border-secondary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-blue: #58a6ff;
            --accent-purple: #a371f7;
            --accent-orange: #d29922;
            --accent-cyan: #39c5cf;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-primary);
        }
        h1 {
            color: var(--text-primary);
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        .subtitle {
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 4px;
        }
        select {
            padding: 8px 12px;
            font-size: 13px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            cursor: pointer;
            outline: none;
        }
        select:hover { border-color: var(--border-secondary); }
        select:focus { border-color: var(--accent-blue); box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15); }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border-primary);
        }
        .stat-label {
            color: var(--text-secondary);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 600;
            margin-top: 4px;
            font-variant-numeric: tabular-nums;
        }
        .stat-value.positive { color: var(--accent-green); }
        .stat-value.negative { color: var(--accent-red); }
        .stat-value.neutral { color: var(--text-primary); }
        .stat-sub {
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 4px;
            font-variant-numeric: tabular-nums;
        }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .chart-container {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
            border: 1px solid var(--border-primary);
        }
        .chart-title {
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }
        .full-width { grid-column: span 2; }

        /* Tables */
        .tables-section {
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-primary);
            overflow: hidden;
            margin-bottom: 24px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-secondary);
        }
        th {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 0.5px;
        }
        tr:last-child td { border-bottom: none; }
        tr:hover td { background: var(--bg-tertiary); }
        td { font-variant-numeric: tabular-nums; }

        /* Insights */
        .insights-section {
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-primary);
            overflow: hidden;
        }
        .section-header {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-secondary);
            background: var(--bg-tertiary);
        }
        .section-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .section-desc {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .insight-list { padding: 8px 0; }
        .insight-item {
            padding: 10px 16px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            border-bottom: 1px solid var(--border-secondary);
        }
        .insight-item:last-child { border-bottom: none; }
        .insight-icon {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            margin-top: 6px;
            flex-shrink: 0;
        }
        .insight-icon.positive { background: var(--accent-green); }
        .insight-icon.negative { background: var(--accent-red); }
        .insight-icon.neutral { background: var(--accent-blue); }
        .insight-icon.warning { background: var(--accent-orange); }
        .insight-text {
            color: var(--text-secondary);
            font-size: 12px;
            line-height: 1.5;
        }
        .insight-text strong { color: var(--text-primary); font-weight: 500; }

        /* Tags */
        .tag {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 500;
        }
        .tag-buy { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .tag-sell { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 4px;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-secondary);
        }
        .tab {
            padding: 6px 12px;
            font-size: 12px;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .tab:hover { background: var(--bg-secondary); color: var(--text-primary); }
        .tab.active { background: var(--bg-secondary); color: var(--text-primary); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        @media (max-width: 1200px) {
            .charts-grid { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Trading Analytics</h1>
            <div class="subtitle">Performance analysis and model evolution metrics</div>
        </div>
        <select id="fileSelect" onchange="loadData()">
            {% for f in files %}
            <option value="{{ f.name }}" {{ 'selected' if f.name == selected else '' }}>
                {{ f.name }} ({{ f.trades }} trades)
            </option>
            {% endfor %}
        </select>
    </div>

    <div id="dashboard">
        <div class="stats-grid" id="statsGrid"></div>
        <div class="charts-grid" id="chartsGrid"></div>
        <div class="tables-section" id="tablesSection"></div>
        <div class="insights-section" id="insightsSection"></div>
    </div>

    <script>
        const darkLayout = {
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#161b22',
            font: { color: '#8b949e', size: 10, family: '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif' },
            margin: { t: 20, r: 16, b: 36, l: 48 },
            xaxis: { gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: { size: 10 } },
            yaxis: { gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: { size: 10 } },
        };

        async function loadData() {
            const file = document.getElementById('fileSelect').value;
            const resp = await fetch(`/api/analyze?file=${file}`);
            const data = await resp.json();
            renderDashboard(data);
        }

        function fmtNum(n, decimals=2) {
            return n.toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
        }

        function renderDashboard(data) {
            // Stats cards
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-label">Total PnL</div>
                    <div class="stat-value ${data.total_pnl >= 0 ? 'positive' : 'negative'}">
                        $${fmtNum(data.total_pnl)}
                    </div>
                    <div class="stat-sub">${fmtNum(data.roi, 1)}% ROI</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Trades</div>
                    <div class="stat-value neutral">${data.total_trades.toLocaleString()}</div>
                    <div class="stat-sub">${fmtNum(data.duration_min, 1)} min session</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Win Rate</div>
                    <div class="stat-value ${data.win_rate >= 50 ? 'positive' : 'neutral'}">${fmtNum(data.win_rate, 1)}%</div>
                    <div class="stat-sub">${data.wins}W / ${data.losses}L</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg PnL</div>
                    <div class="stat-value ${data.avg_pnl >= 0 ? 'positive' : 'negative'}">
                        $${fmtNum(data.avg_pnl)}
                    </div>
                    <div class="stat-sub">Med: $${fmtNum(data.median_pnl)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Sharpe</div>
                    <div class="stat-value ${data.sharpe >= 1 ? 'positive' : 'neutral'}">${fmtNum(data.sharpe)}</div>
                    <div class="stat-sub">Sortino: ${fmtNum(data.sortino)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max Drawdown</div>
                    <div class="stat-value negative">$${fmtNum(Math.abs(data.max_drawdown))}</div>
                    <div class="stat-sub">${fmtNum(data.max_drawdown_pct, 1)}% of peak</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Profit Factor</div>
                    <div class="stat-value ${data.profit_factor >= 1 ? 'positive' : 'negative'}">${fmtNum(data.profit_factor)}</div>
                    <div class="stat-sub">+$${fmtNum(data.gross_profit, 0)} / -$${fmtNum(Math.abs(data.gross_loss), 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Edge Score</div>
                    <div class="stat-value ${data.edge_score >= 0 ? 'positive' : 'negative'}">${fmtNum(data.edge_score, 1)}%</div>
                    <div class="stat-sub">vs random baseline</div>
                </div>
            `;
            document.getElementById('statsGrid').innerHTML = statsHtml;

            // Charts
            document.getElementById('chartsGrid').innerHTML = `
                <div class="chart-container full-width">
                    <div class="chart-title">Equity Curve</div>
                    <div id="equityChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">PnL by Asset</div>
                    <div id="assetChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">PnL by Direction</div>
                    <div id="actionChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Entry Probability Distribution</div>
                    <div id="probChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Avg PnL by Entry Probability</div>
                    <div id="probPnlChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Avg PnL by Time Remaining</div>
                    <div id="timeChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Trade Duration Distribution</div>
                    <div id="durationChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">PnL Distribution</div>
                    <div id="pnlDistChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Binance Move vs Trade PnL (Signal Quality)</div>
                    <div id="binanceCorrelationChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Rolling Win Rate (50-trade window)</div>
                    <div id="rollingWinChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Rolling Avg PnL per Trade</div>
                    <div id="rollingPnlChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Rolling Sharpe Ratio</div>
                    <div id="rollingSharpeChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Rolling Profit Factor</div>
                    <div id="rollingProfitFactorChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Cumulative Win Rate Evolution</div>
                    <div id="cumulativeWinRateChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">PnL by Session Phase</div>
                    <div id="sessionPhaseChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Win Rate by Session Phase</div>
                    <div id="sessionWinRateChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title">Price Movement Capture Rate (Entry to Exit)</div>
                    <div id="priceMovementChart"></div>
                </div>
                <div class="chart-container full-width" style="margin-top: 24px; border-top: 2px solid var(--accent-purple);">
                    <div class="chart-title" style="color: var(--accent-purple);">Learning Progression: Decision Quality Over Time</div>
                    <div id="learningDecisionChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title" style="color: var(--accent-purple);">Learning Progression: Risk-Adjusted Performance</div>
                    <div id="learningRiskAdjustedChart"></div>
                </div>
                <div class="chart-container full-width">
                    <div class="chart-title" style="color: var(--accent-purple);">Learning Progression: Win/Loss Magnitude Ratio</div>
                    <div id="learningMagnitudeChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title" style="color: var(--accent-purple);">Action Distribution Evolution</div>
                    <div id="actionEvolutionChart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title" style="color: var(--accent-purple);">Optimal Hold Duration Discovery</div>
                    <div id="holdDurationChart"></div>
                </div>
            `;

            // Equity curve
            Plotly.newPlot('equityChart', [{
                y: data.equity_curve,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: data.total_pnl >= 0 ? '#3fb950' : '#f85149', width: 1.5 },
                fillcolor: data.total_pnl >= 0 ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)'
            }], {...darkLayout, height: 220, showlegend: false});

            // Asset chart
            Plotly.newPlot('assetChart', [{
                x: data.by_asset.assets,
                y: data.by_asset.pnl,
                type: 'bar',
                marker: { color: data.by_asset.pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 200});

            // Action chart
            Plotly.newPlot('actionChart', [{
                x: data.by_action.actions,
                y: data.by_action.pnl,
                type: 'bar',
                marker: { color: data.by_action.pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 200});

            // Prob distribution
            Plotly.newPlot('probChart', [{
                x: data.prob_hist.bins,
                y: data.prob_hist.counts,
                type: 'bar',
                marker: { color: '#58a6ff' }
            }], {...darkLayout, height: 200});

            // Prob vs PnL
            Plotly.newPlot('probPnlChart', [{
                x: data.by_prob.buckets,
                y: data.by_prob.avg_pnl,
                type: 'bar',
                marker: { color: data.by_prob.avg_pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 200});

            // Time remaining vs PnL
            Plotly.newPlot('timeChart', [{
                x: data.by_time.buckets,
                y: data.by_time.avg_pnl,
                type: 'bar',
                marker: { color: data.by_time.avg_pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 200});

            // Duration distribution
            Plotly.newPlot('durationChart', [{
                x: data.duration_hist.bins,
                y: data.duration_hist.counts,
                type: 'bar',
                marker: { color: '#a371f7' }
            }], {...darkLayout, height: 200});

            // PnL distribution
            Plotly.newPlot('pnlDistChart', [{
                x: data.pnl_hist.bins,
                y: data.pnl_hist.counts,
                type: 'bar',
                marker: { color: data.pnl_hist.bins.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 180});

            // Binance correlation scatter
            if (data.binance_correlation) {
                Plotly.newPlot('binanceCorrelationChart', [{
                    x: data.binance_correlation.binance_moves,
                    y: data.binance_correlation.pnls,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        color: data.binance_correlation.pnls.map(v => v >= 0 ? '#3fb950' : '#f85149'),
                        size: 4,
                        opacity: 0.5
                    }
                }, {
                    x: data.binance_correlation.regression_x,
                    y: data.binance_correlation.regression_y,
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: '#58a6ff', width: 2 }
                }], {...darkLayout, height: 200, showlegend: false,
                    xaxis: {...darkLayout.xaxis, title: {text: 'Binance % Move', font: {size: 10}}},
                    yaxis: {...darkLayout.yaxis, title: {text: 'Trade PnL ($)', font: {size: 10}}}
                });
            }

            // Rolling win rate
            Plotly.newPlot('rollingWinChart', [{
                y: data.rolling_win_rate,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#d29922', width: 1.5 }
            }, {
                y: Array(data.rolling_win_rate.length).fill(50),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#30363d', width: 1, dash: 'dash' }
            }], {...darkLayout, height: 180, showlegend: false});

            // Rolling avg PnL
            Plotly.newPlot('rollingPnlChart', [{
                y: data.rolling_avg_pnl,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3fb950', width: 1.5 }
            }, {
                y: Array(data.rolling_avg_pnl.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#30363d', width: 1, dash: 'dash' }
            }], {...darkLayout, height: 180, showlegend: false});

            // Rolling Sharpe
            Plotly.newPlot('rollingSharpeChart', [{
                y: data.rolling_sharpe,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#a371f7', width: 1.5 }
            }, {
                y: Array(data.rolling_sharpe.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#30363d', width: 1, dash: 'dash' }
            }], {...darkLayout, height: 180, showlegend: false});

            // Rolling Profit Factor
            Plotly.newPlot('rollingProfitFactorChart', [{
                y: data.rolling_profit_factor,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#39c5cf', width: 1.5 }
            }, {
                y: Array(data.rolling_profit_factor.length).fill(1),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#30363d', width: 1, dash: 'dash' }
            }], {...darkLayout, height: 180, showlegend: false});

            // Cumulative win rate
            Plotly.newPlot('cumulativeWinRateChart', [{
                y: data.cumulative_win_rate,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#58a6ff', width: 1.5 }
            }, {
                y: Array(data.cumulative_win_rate.length).fill(50),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#30363d', width: 1, dash: 'dash' }
            }], {...darkLayout, height: 180, showlegend: false});

            // Session phase PnL
            Plotly.newPlot('sessionPhaseChart', [{
                x: data.by_session_phase.phases,
                y: data.by_session_phase.avg_pnl,
                type: 'bar',
                marker: { color: data.by_session_phase.avg_pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
            }], {...darkLayout, height: 200});

            // Session phase win rate
            Plotly.newPlot('sessionWinRateChart', [{
                x: data.by_session_phase.phases,
                y: data.by_session_phase.win_rates,
                type: 'bar',
                marker: { color: '#58a6ff' }
            }], {...darkLayout, height: 200, yaxis: {...darkLayout.yaxis, range: [0, 100]}});

            // Price movement capture
            if (data.price_movement) {
                Plotly.newPlot('priceMovementChart', [{
                    x: ['Captured Moves', 'Missed Moves', 'Wrong Direction'],
                    y: [data.price_movement.captured, data.price_movement.missed, data.price_movement.wrong],
                    type: 'bar',
                    marker: { color: ['#3fb950', '#d29922', '#f85149'] }
                }], {...darkLayout, height: 200});
            }

            // Learning Progression Charts
            if (data.learning_metrics) {
                // Decision quality over time (direction accuracy + favorable entries)
                Plotly.newPlot('learningDecisionChart', [{
                    y: data.learning_metrics.rolling_direction_accuracy,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Direction Accuracy',
                    line: { color: '#3fb950', width: 1.5 }
                }, {
                    y: data.learning_metrics.rolling_favorable_entry,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Favorable Entry %',
                    line: { color: '#58a6ff', width: 1.5 }
                }, {
                    y: Array(data.learning_metrics.rolling_direction_accuracy.length).fill(50),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Random Baseline',
                    line: { color: '#30363d', width: 1, dash: 'dash' }
                }], {...darkLayout, height: 200, showlegend: true,
                    legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 10 } }
                });

                // Risk-adjusted performance (rolling Sharpe + Sortino)
                Plotly.newPlot('learningRiskAdjustedChart', [{
                    y: data.learning_metrics.rolling_sortino,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Sortino Ratio',
                    line: { color: '#a371f7', width: 1.5 }
                }, {
                    y: data.rolling_sharpe,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Sharpe Ratio',
                    line: { color: '#39c5cf', width: 1.5 }
                }, {
                    y: Array(data.rolling_sharpe.length).fill(0),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Zero Line',
                    line: { color: '#30363d', width: 1, dash: 'dash' }
                }], {...darkLayout, height: 200, showlegend: true,
                    legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 10 } }
                });

                // Win/Loss magnitude ratio
                Plotly.newPlot('learningMagnitudeChart', [{
                    y: data.learning_metrics.rolling_win_loss_ratio,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Avg Win / Avg Loss',
                    line: { color: '#d29922', width: 1.5 }
                }, {
                    y: Array(data.learning_metrics.rolling_win_loss_ratio.length).fill(1),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Breakeven Line',
                    line: { color: '#30363d', width: 1, dash: 'dash' }
                }], {...darkLayout, height: 200, showlegend: true,
                    legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 10 } }
                });

                // Action distribution evolution (stacked area)
                Plotly.newPlot('actionEvolutionChart', [{
                    y: data.learning_metrics.action_dist_hold,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'HOLD',
                    fill: 'tozeroy',
                    line: { color: '#8b949e', width: 0 },
                    fillcolor: 'rgba(139, 148, 158, 0.5)'
                }, {
                    y: data.learning_metrics.action_dist_buy,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'BUY',
                    fill: 'tozeroy',
                    line: { color: '#3fb950', width: 0 },
                    fillcolor: 'rgba(63, 185, 80, 0.5)'
                }, {
                    y: data.learning_metrics.action_dist_sell,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SELL',
                    fill: 'tozeroy',
                    line: { color: '#58a6ff', width: 0 },
                    fillcolor: 'rgba(88, 166, 255, 0.5)'
                }], {...darkLayout, height: 200, showlegend: true,
                    legend: { x: 0, y: 1.15, orientation: 'h', font: { size: 10 } }
                });

                // Hold duration by quartile
                Plotly.newPlot('holdDurationChart', [{
                    x: data.learning_metrics.duration_quartiles.labels,
                    y: data.learning_metrics.duration_quartiles.avg_pnl,
                    type: 'bar',
                    marker: { color: data.learning_metrics.duration_quartiles.avg_pnl.map(v => v >= 0 ? '#3fb950' : '#f85149') }
                }], {...darkLayout, height: 200});
            }

            // Tables section with tabs
            document.getElementById('tablesSection').innerHTML = `
                <div class="tabs">
                    <button class="tab active" onclick="showTab('assetTab', this)">By Asset</button>
                    <button class="tab" onclick="showTab('timingTab', this)">Timing</button>
                    <button class="tab" onclick="showTab('probTab', this)">Entry Prob</button>
                    <button class="tab" onclick="showTab('streakTab', this)">Streaks</button>
                    <button class="tab" onclick="showTab('phaseTab', this)">Session Phase</button>
                    <button class="tab" onclick="showTab('edgeTab', this)">Edge Analysis</button>
                    <button class="tab" onclick="showTab('learningTab', this)" style="color: var(--accent-purple);">Learning</button>
                </div>
                <div id="assetTab" class="tab-content active">
                    <table>
                        <thead>
                            <tr><th>Asset</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th><th>Best</th><th>Worst</th></tr>
                        </thead>
                        <tbody>
                            ${data.asset_details.map(a => `
                                <tr>
                                    <td><strong>${a.asset}</strong></td>
                                    <td>${a.trades}</td>
                                    <td>${fmtNum(a.win_rate, 1)}%</td>
                                    <td style="color: ${a.pnl >= 0 ? '#3fb950' : '#f85149'}">$${fmtNum(a.pnl)}</td>
                                    <td>$${fmtNum(a.avg)}</td>
                                    <td style="color: #3fb950">$${fmtNum(a.best)}</td>
                                    <td style="color: #f85149">$${fmtNum(a.worst)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div id="timingTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Time Remaining</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr>
                        </thead>
                        <tbody>
                            ${data.time_details.map(t => `
                                <tr>
                                    <td>${t.bucket}</td>
                                    <td>${t.trades}</td>
                                    <td>${fmtNum(t.win_rate, 1)}%</td>
                                    <td style="color: ${t.pnl >= 0 ? '#3fb950' : '#f85149'}">$${fmtNum(t.pnl)}</td>
                                    <td>$${fmtNum(t.avg)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div id="probTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Probability Range</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr>
                        </thead>
                        <tbody>
                            ${data.prob_details.map(p => `
                                <tr>
                                    <td>${p.bucket}</td>
                                    <td>${p.trades}</td>
                                    <td>${fmtNum(p.win_rate, 1)}%</td>
                                    <td style="color: ${p.pnl >= 0 ? '#3fb950' : '#f85149'}">$${fmtNum(p.pnl)}</td>
                                    <td>$${fmtNum(p.avg)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div id="streakTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Metric</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>Max Winning Streak</td><td style="color: #3fb950">${data.max_win_streak} trades</td></tr>
                            <tr><td>Max Losing Streak</td><td style="color: #f85149">${data.max_loss_streak} trades</td></tr>
                            <tr><td>Avg Winning Streak</td><td>${fmtNum(data.avg_win_streak, 1)} trades</td></tr>
                            <tr><td>Avg Losing Streak</td><td>${fmtNum(data.avg_loss_streak, 1)} trades</td></tr>
                            <tr><td>Current Streak</td><td>${data.current_streak > 0 ? '+' : ''}${data.current_streak} (${data.current_streak > 0 ? 'winning' : 'losing'})</td></tr>
                        </tbody>
                    </table>
                </div>
                <div id="phaseTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Phase</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr>
                        </thead>
                        <tbody>
                            ${data.by_session_phase.phases.map((phase, i) => `
                                <tr>
                                    <td><strong>${phase}</strong></td>
                                    <td>${data.by_session_phase.trades[i]}</td>
                                    <td>${fmtNum(data.by_session_phase.win_rates[i], 1)}%</td>
                                    <td style="color: ${data.by_session_phase.total_pnl[i] >= 0 ? '#3fb950' : '#f85149'}">$${fmtNum(data.by_session_phase.total_pnl[i])}</td>
                                    <td>$${fmtNum(data.by_session_phase.avg_pnl[i])}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div id="edgeTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Binance Signal Correlation</td>
                                <td style="color: ${data.edge_metrics.binance_corr >= 0.1 ? '#3fb950' : (data.edge_metrics.binance_corr < 0 ? '#f85149' : '#8b949e')}">${fmtNum(data.edge_metrics.binance_corr, 3)}</td>
                                <td style="color: #8b949e">${data.edge_metrics.binance_corr >= 0.2 ? 'Strong signal' : (data.edge_metrics.binance_corr >= 0.1 ? 'Weak signal' : 'No signal')}</td>
                            </tr>
                            <tr>
                                <td>Avg Price Movement Captured</td>
                                <td style="color: ${data.edge_metrics.avg_move_captured >= 0 ? '#3fb950' : '#f85149'}">${fmtNum(data.edge_metrics.avg_move_captured * 100, 2)}%</td>
                                <td style="color: #8b949e">Of total entry-to-exit price move</td>
                            </tr>
                            <tr>
                                <td>Favorable Entry Rate</td>
                                <td style="color: ${data.edge_metrics.favorable_entry_rate >= 50 ? '#3fb950' : '#f85149'}">${fmtNum(data.edge_metrics.favorable_entry_rate, 1)}%</td>
                                <td style="color: #8b949e">Entries at better-than-fair prices</td>
                            </tr>
                            <tr>
                                <td>Favorable Exit Rate</td>
                                <td style="color: ${data.edge_metrics.favorable_exit_rate >= 50 ? '#3fb950' : '#f85149'}">${fmtNum(data.edge_metrics.favorable_exit_rate, 1)}%</td>
                                <td style="color: #8b949e">Exits at better prices than entry</td>
                            </tr>
                            <tr>
                                <td>Avg Hold Duration</td>
                                <td>${fmtNum(data.avg_duration, 1)}s</td>
                                <td style="color: #8b949e">Median: ${fmtNum(data.median_duration, 1)}s</td>
                            </tr>
                            <tr>
                                <td>Direction Accuracy</td>
                                <td style="color: ${data.edge_metrics.direction_accuracy >= 50 ? '#3fb950' : '#f85149'}">${fmtNum(data.edge_metrics.direction_accuracy, 1)}%</td>
                                <td style="color: #8b949e">Correct directional bets</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div id="learningTab" class="tab-content">
                    <table>
                        <thead>
                            <tr><th>Learning Metric</th><th>Early (First 25%)</th><th>Late (Last 25%)</th><th>Change</th><th>Trend</th></tr>
                        </thead>
                        <tbody>
                            ${data.learning_metrics ? `
                            <tr>
                                <td>Direction Accuracy</td>
                                <td>${fmtNum(data.learning_metrics.early_direction_acc, 1)}%</td>
                                <td>${fmtNum(data.learning_metrics.late_direction_acc, 1)}%</td>
                                <td style="color: ${data.learning_metrics.late_direction_acc > data.learning_metrics.early_direction_acc ? '#3fb950' : '#f85149'}">${data.learning_metrics.late_direction_acc > data.learning_metrics.early_direction_acc ? '+' : ''}${fmtNum(data.learning_metrics.late_direction_acc - data.learning_metrics.early_direction_acc, 1)}%</td>
                                <td>${data.learning_metrics.late_direction_acc > data.learning_metrics.early_direction_acc + 2 ? 'Improving' : (data.learning_metrics.late_direction_acc < data.learning_metrics.early_direction_acc - 2 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            <tr>
                                <td>Favorable Entry Rate</td>
                                <td>${fmtNum(data.learning_metrics.early_favorable_entry, 1)}%</td>
                                <td>${fmtNum(data.learning_metrics.late_favorable_entry, 1)}%</td>
                                <td style="color: ${data.learning_metrics.late_favorable_entry > data.learning_metrics.early_favorable_entry ? '#3fb950' : '#f85149'}">${data.learning_metrics.late_favorable_entry > data.learning_metrics.early_favorable_entry ? '+' : ''}${fmtNum(data.learning_metrics.late_favorable_entry - data.learning_metrics.early_favorable_entry, 1)}%</td>
                                <td>${data.learning_metrics.late_favorable_entry > data.learning_metrics.early_favorable_entry + 2 ? 'Improving' : (data.learning_metrics.late_favorable_entry < data.learning_metrics.early_favorable_entry - 2 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            <tr>
                                <td>Avg Win Size</td>
                                <td>$${fmtNum(data.learning_metrics.early_avg_win, 2)}</td>
                                <td>$${fmtNum(data.learning_metrics.late_avg_win, 2)}</td>
                                <td style="color: ${data.learning_metrics.late_avg_win > data.learning_metrics.early_avg_win ? '#3fb950' : '#f85149'}">${data.learning_metrics.late_avg_win > data.learning_metrics.early_avg_win ? '+' : ''}$${fmtNum(data.learning_metrics.late_avg_win - data.learning_metrics.early_avg_win, 2)}</td>
                                <td>${data.learning_metrics.late_avg_win > data.learning_metrics.early_avg_win * 1.1 ? 'Improving' : (data.learning_metrics.late_avg_win < data.learning_metrics.early_avg_win * 0.9 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            <tr>
                                <td>Avg Loss Size</td>
                                <td>$${fmtNum(data.learning_metrics.early_avg_loss, 2)}</td>
                                <td>$${fmtNum(data.learning_metrics.late_avg_loss, 2)}</td>
                                <td style="color: ${Math.abs(data.learning_metrics.late_avg_loss) < Math.abs(data.learning_metrics.early_avg_loss) ? '#3fb950' : '#f85149'}">${Math.abs(data.learning_metrics.late_avg_loss) < Math.abs(data.learning_metrics.early_avg_loss) ? '' : '+'}$${fmtNum(data.learning_metrics.late_avg_loss - data.learning_metrics.early_avg_loss, 2)}</td>
                                <td>${Math.abs(data.learning_metrics.late_avg_loss) < Math.abs(data.learning_metrics.early_avg_loss) * 0.9 ? 'Improving' : (Math.abs(data.learning_metrics.late_avg_loss) > Math.abs(data.learning_metrics.early_avg_loss) * 1.1 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            <tr>
                                <td>Win/Loss Ratio</td>
                                <td>${fmtNum(data.learning_metrics.early_wl_ratio, 2)}</td>
                                <td>${fmtNum(data.learning_metrics.late_wl_ratio, 2)}</td>
                                <td style="color: ${data.learning_metrics.late_wl_ratio > data.learning_metrics.early_wl_ratio ? '#3fb950' : '#f85149'}">${data.learning_metrics.late_wl_ratio > data.learning_metrics.early_wl_ratio ? '+' : ''}${fmtNum(data.learning_metrics.late_wl_ratio - data.learning_metrics.early_wl_ratio, 2)}</td>
                                <td>${data.learning_metrics.late_wl_ratio > data.learning_metrics.early_wl_ratio * 1.1 ? 'Improving' : (data.learning_metrics.late_wl_ratio < data.learning_metrics.early_wl_ratio * 0.9 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>${fmtNum(data.learning_metrics.early_sharpe, 2)}</td>
                                <td>${fmtNum(data.learning_metrics.late_sharpe, 2)}</td>
                                <td style="color: ${data.learning_metrics.late_sharpe > data.learning_metrics.early_sharpe ? '#3fb950' : '#f85149'}">${data.learning_metrics.late_sharpe > data.learning_metrics.early_sharpe ? '+' : ''}${fmtNum(data.learning_metrics.late_sharpe - data.learning_metrics.early_sharpe, 2)}</td>
                                <td>${data.learning_metrics.late_sharpe > data.learning_metrics.early_sharpe + 0.1 ? 'Improving' : (data.learning_metrics.late_sharpe < data.learning_metrics.early_sharpe - 0.1 ? 'Declining' : 'Stable')}</td>
                            </tr>
                            ` : '<tr><td colspan="5">Not enough data for learning analysis</td></tr>'}
                        </tbody>
                    </table>
                    <div style="padding: 16px; color: var(--text-muted); font-size: 11px;">
                        <strong>How to interpret:</strong> These metrics compare the first 25% of trades to the last 25%.
                        Consistent improvement across multiple metrics indicates the model is learning effective trading patterns.
                        Key signals: Direction Accuracy > 50% shows predictive ability, Win/Loss Ratio > 1.0 shows profitable risk management.
                    </div>
                </div>
            `;

            // Insights
            let insightsHtml = `
                <div class="section-header">
                    <div class="section-title">Key Insights</div>
                    <div class="section-desc">Automated analysis of trading patterns and model behavior</div>
                </div>
                <div class="insight-list">
            `;
            data.insights.forEach(i => {
                insightsHtml += `<div class="insight-item">
                    <span class="insight-icon ${i.type}"></span>
                    <span class="insight-text">${i.text}</span>
                </div>`;
            });
            insightsHtml += '</div>';
            document.getElementById('insightsSection').innerHTML = insightsHtml;
        }

        function showTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }

        // Load on start
        loadData();
    </script>
</body>
</html>
"""

def get_trade_files():
    """Get all trade files with metadata."""
    files = []
    for f in sorted(LOGS_DIR.glob("trades_*.csv"), reverse=True):
        try:
            df = pd.read_csv(f)
            files.append({
                'name': f.name,
                'trades': len(df),
                'path': str(f)
            })
        except:
            pass
    return files

def analyze_trades(filepath):
    """Comprehensive trade analysis."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic stats
    total_pnl = df['pnl'].sum()
    total_trades = len(df)
    wins = (df['pnl'] > 0).sum()
    losses = (df['pnl'] <= 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # Duration
    duration_min = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60

    # PnL metrics
    avg_pnl = df['pnl'].mean()
    median_pnl = df['pnl'].median()
    std_pnl = df['pnl'].std()

    # Sharpe & Sortino (annualized approximation)
    sharpe = (avg_pnl / std_pnl * np.sqrt(total_trades)) if std_pnl > 0 else 0
    downside_std = df[df['pnl'] < 0]['pnl'].std() if len(df[df['pnl'] < 0]) > 0 else 1
    sortino = (avg_pnl / downside_std * np.sqrt(total_trades)) if downside_std > 0 else 0

    # Drawdown
    equity = df['pnl'].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / peak.max() * 100) if peak.max() > 0 else 0

    # Profit factor
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Duration stats
    avg_duration = df['duration_sec'].mean()
    median_duration = df['duration_sec'].median()

    # ROI (assuming $500 position size from the data)
    position_size = df['size'].iloc[0] if 'size' in df.columns else 500
    roi = (total_pnl / (position_size * 4)) * 100  # 4 markets

    # Edge Score: how much better than random (33% for 3 actions)
    expected_random_win_rate = 33.33
    edge_score = win_rate - expected_random_win_rate

    # === NEW: Edge Metrics ===

    # Binance correlation analysis
    binance_corr = 0.0
    binance_correlation_data = None
    if 'binance_change' in df.columns:
        # Filter out zero binance changes for meaningful correlation
        df_with_binance = df[df['binance_change'] != 0].copy()
        if len(df_with_binance) > 10:
            binance_corr = df_with_binance['binance_change'].corr(df_with_binance['pnl'])
            if np.isnan(binance_corr):
                binance_corr = 0.0

            # Prepare scatter data
            binance_moves = df_with_binance['binance_change'].tolist()
            pnls = df_with_binance['pnl'].tolist()

            # Simple linear regression for trend line
            if len(binance_moves) > 2:
                x_arr = np.array(binance_moves)
                y_arr = np.array(pnls)
                slope, intercept = np.polyfit(x_arr, y_arr, 1)
                x_line = [min(binance_moves), max(binance_moves)]
                y_line = [slope * x + intercept for x in x_line]
                binance_correlation_data = {
                    'binance_moves': binance_moves,
                    'pnls': pnls,
                    'regression_x': x_line,
                    'regression_y': y_line
                }

    # Price movement capture analysis
    df['price_move'] = df['exit_price'] - df['entry_price']
    df['favorable_move'] = ((df['action'] == 'BUY') & (df['price_move'] > 0)) | \
                           ((df['action'] == 'SELL') & (df['price_move'] < 0))

    # Calculate average movement captured
    avg_move_captured = df['price_move'].abs().mean() if len(df) > 0 else 0

    # Favorable entry rate: did we enter at a good price?
    # For BUY, entry_price < 0.5 is favorable (undervalued)
    # For SELL, entry_price > 0.5 is favorable (overvalued)
    df['favorable_entry'] = ((df['action'] == 'BUY') & (df['entry_price'] < 0.5)) | \
                            ((df['action'] == 'SELL') & (df['entry_price'] > 0.5))
    favorable_entry_rate = df['favorable_entry'].mean() * 100 if len(df) > 0 else 0

    # Favorable exit rate: did price move in our favor?
    favorable_exit_rate = df['favorable_move'].mean() * 100 if len(df) > 0 else 0

    # Direction accuracy: correct directional prediction
    direction_accuracy = favorable_exit_rate  # Same as favorable exit for binary outcomes

    # Price movement categorization for chart
    captured = (df['pnl'] > 0).sum()
    missed = ((df['pnl'] == 0) | ((df['pnl'] < 0) & (df['pnl'] > -5))).sum()  # Small losses
    wrong = (df['pnl'] < -5).sum()  # Significant losses

    price_movement_data = {
        'captured': int(captured),
        'missed': int(missed),
        'wrong': int(wrong)
    }

    edge_metrics = {
        'binance_corr': binance_corr,
        'avg_move_captured': avg_move_captured,
        'favorable_entry_rate': favorable_entry_rate,
        'favorable_exit_rate': favorable_exit_rate,
        'direction_accuracy': direction_accuracy
    }

    # By asset
    by_asset = df.groupby('asset').agg({
        'pnl': ['sum', 'mean', 'count', 'max', 'min']
    }).reset_index()
    by_asset.columns = ['asset', 'pnl', 'avg', 'trades', 'best', 'worst']
    by_asset['win_rate'] = df.groupby('asset').apply(lambda x: (x['pnl'] > 0).mean() * 100).values

    # By action
    by_action = df.groupby('action')['pnl'].agg(['sum', 'mean', 'count']).reset_index()

    # Probability buckets
    prob_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    prob_labels = ['<0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '>0.7']
    df['prob_bucket'] = pd.cut(df['prob_at_entry'], bins=prob_bins, labels=prob_labels)
    by_prob = df.groupby('prob_bucket', observed=True).agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    by_prob.columns = ['bucket', 'pnl', 'avg', 'trades']
    by_prob['win_rate'] = df.groupby('prob_bucket', observed=True).apply(
        lambda x: (x['pnl'] > 0).mean() * 100, include_groups=False
    ).values

    # Time buckets
    time_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    time_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['time_bucket'] = pd.cut(df['time_remaining'], bins=time_bins, labels=time_labels)
    by_time = df.groupby('time_bucket', observed=True).agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    by_time.columns = ['bucket', 'pnl', 'avg', 'trades']
    by_time['win_rate'] = df.groupby('time_bucket', observed=True).apply(
        lambda x: (x['pnl'] > 0).mean() * 100, include_groups=False
    ).values

    # Histograms
    prob_hist, prob_edges = np.histogram(df['prob_at_entry'], bins=20)
    duration_hist, duration_edges = np.histogram(df['duration_sec'].clip(0, 30), bins=20)
    pnl_hist, pnl_edges = np.histogram(df['pnl'].clip(-100, 100), bins=40)

    # Rolling win rate
    rolling_win = (df['pnl'] > 0).rolling(50).mean() * 100
    rolling_win = rolling_win.dropna().tolist()

    # Rolling avg PnL per trade
    rolling_avg_pnl = df['pnl'].rolling(50).mean()
    rolling_avg_pnl = rolling_avg_pnl.dropna().tolist()

    # Rolling Sharpe ratio (simplified - mean/std over window)
    def rolling_sharpe_calc(window):
        mean = window.mean()
        std = window.std()
        return mean / std if std > 0 else 0
    rolling_sharpe = df['pnl'].rolling(50).apply(rolling_sharpe_calc, raw=False)
    rolling_sharpe = rolling_sharpe.dropna().tolist()

    # Rolling profit factor
    def rolling_pf_calc(window):
        wins = window[window > 0].sum()
        losses = abs(window[window <= 0].sum())
        return min(wins / losses, 10) if losses > 0 else 10  # Cap at 10 for display
    rolling_pf = df['pnl'].rolling(50).apply(rolling_pf_calc, raw=False)
    rolling_pf = rolling_pf.dropna().tolist()

    # Cumulative win rate over time
    cumulative_wins = (df['pnl'] > 0).cumsum()
    cumulative_count = np.arange(1, len(df) + 1)
    cumulative_win_rate = (cumulative_wins / cumulative_count * 100).tolist()

    # Performance by session phase (split session into 5 equal parts)
    df['trade_idx'] = np.arange(len(df))
    df['session_phase'] = pd.cut(df['trade_idx'], bins=5, labels=['Start', 'Early', 'Mid', 'Late', 'End'])
    by_session = df.groupby('session_phase', observed=True).agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    by_session.columns = ['phase', 'total_pnl', 'avg_pnl', 'trades']
    by_session['win_rate'] = df.groupby('session_phase', observed=True).apply(
        lambda x: (x['pnl'] > 0).mean() * 100, include_groups=False
    ).values

    # Streak analysis
    streaks = []
    current = 0
    for pnl in df['pnl']:
        if pnl > 0:
            if current >= 0:
                current += 1
            else:
                streaks.append(current)
                current = 1
        else:
            if current <= 0:
                current -= 1
            else:
                streaks.append(current)
                current = -1
    streaks.append(current)

    win_streaks = [s for s in streaks if s > 0]
    loss_streaks = [abs(s) for s in streaks if s < 0]

    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = np.mean(win_streaks) if win_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0

    # Generate insights (no emojis, with type for styling)
    insights = []

    # Best/worst asset
    best_asset = by_asset.loc[by_asset['pnl'].idxmax()]
    worst_asset = by_asset.loc[by_asset['pnl'].idxmin()]
    insights.append({
        'type': 'positive',
        'text': f"<strong>Best performer:</strong> {best_asset['asset']} with +${best_asset['pnl']:.2f} ({best_asset['win_rate']:.1f}% win rate)"
    })
    if worst_asset['pnl'] < 0:
        insights.append({
            'type': 'negative',
            'text': f"<strong>Worst performer:</strong> {worst_asset['asset']} with ${worst_asset['pnl']:.2f} ({worst_asset['win_rate']:.1f}% win rate)"
        })

    # Timing insight
    best_time = by_time.loc[by_time['avg'].idxmax()] if len(by_time) > 0 else None
    if best_time is not None:
        insights.append({
            'type': 'neutral',
            'text': f"<strong>Optimal timing:</strong> {best_time['bucket']} time remaining yields ${best_time['avg']:.2f} avg PnL"
        })

    # Probability insight
    best_prob = by_prob.loc[by_prob['avg'].idxmax()] if len(by_prob) > 0 else None
    if best_prob is not None:
        insights.append({
            'type': 'neutral',
            'text': f"<strong>Best entry zone:</strong> {best_prob['bucket']} probability range with ${best_prob['avg']:.2f} avg PnL"
        })

    # Action bias
    buy_pnl = df[df['action'] == 'BUY']['pnl'].sum()
    sell_pnl = df[df['action'] == 'SELL']['pnl'].sum()
    if abs(buy_pnl - sell_pnl) > 100:
        better = 'BUY (UP bets)' if buy_pnl > sell_pnl else 'SELL (DOWN bets)'
        insights.append({
            'type': 'neutral',
            'text': f"<strong>Direction bias:</strong> {better} outperforming by ${abs(buy_pnl - sell_pnl):.2f}"
        })

    # Win rate vs profitability
    if win_rate < 30 and total_pnl > 0:
        insights.append({
            'type': 'positive',
            'text': f"<strong>Asymmetric payoffs working:</strong> Only {win_rate:.1f}% win rate but still profitable via position sizing"
        })

    # Streak warning
    if max_loss_streak > 15:
        insights.append({
            'type': 'warning',
            'text': f"<strong>Risk alert:</strong> Max losing streak of {max_loss_streak} trades - monitor risk management"
        })

    # Binance signal insight
    if abs(binance_corr) > 0.1:
        corr_type = 'positive' if binance_corr > 0 else 'warning'
        insights.append({
            'type': corr_type,
            'text': f"<strong>Binance signal correlation:</strong> {binance_corr:.3f} - {'exploiting' if binance_corr > 0 else 'inverse'} price lag"
        })

    # Evolution insights
    if len(cumulative_win_rate) > 100:
        early_wr = np.mean(cumulative_win_rate[:100])
        late_wr = np.mean(cumulative_win_rate[-100:])
        if late_wr > early_wr + 2:
            insights.append({
                'type': 'positive',
                'text': f"<strong>Improving performance:</strong> Win rate {early_wr:.1f}% early to {late_wr:.1f}% late (+{late_wr - early_wr:.1f}%)"
            })
        elif early_wr > late_wr + 2:
            insights.append({
                'type': 'warning',
                'text': f"<strong>Declining performance:</strong> Win rate {early_wr:.1f}% early to {late_wr:.1f}% late ({late_wr - early_wr:.1f}%)"
            })

    if len(rolling_avg_pnl) > 100:
        early_pnl = np.mean(rolling_avg_pnl[:50])
        late_pnl = np.mean(rolling_avg_pnl[-50:])
        if late_pnl > early_pnl + 1:
            insights.append({
                'type': 'positive',
                'text': f"<strong>PnL per trade improving:</strong> ${early_pnl:.2f} early to ${late_pnl:.2f} late"
            })
        elif early_pnl > late_pnl + 1:
            insights.append({
                'type': 'warning',
                'text': f"<strong>PnL per trade declining:</strong> ${early_pnl:.2f} early to ${late_pnl:.2f} late"
            })

    # Session phase insight
    if len(by_session) > 0:
        best_phase = by_session.loc[by_session['avg_pnl'].idxmax()]
        worst_phase = by_session.loc[by_session['avg_pnl'].idxmin()]
        if best_phase['avg_pnl'] > 0 and worst_phase['avg_pnl'] < 0:
            insights.append({
                'type': 'neutral',
                'text': f"<strong>Session pattern:</strong> Best in {best_phase['phase']} phase (${best_phase['avg_pnl']:.2f} avg), worst in {worst_phase['phase']} (${worst_phase['avg_pnl']:.2f} avg)"
            })

    # Edge analysis insight
    if favorable_entry_rate > 55:
        insights.append({
            'type': 'positive',
            'text': f"<strong>Entry timing edge:</strong> {favorable_entry_rate:.1f}% of entries at favorable prices (vs 50% random)"
        })
    elif favorable_entry_rate < 45:
        insights.append({
            'type': 'warning',
            'text': f"<strong>Poor entry timing:</strong> Only {favorable_entry_rate:.1f}% of entries at favorable prices"
        })

    # === LEARNING PROGRESSION METRICS ===
    learning_metrics = None
    if total_trades >= 100:  # Need enough data
        quarter_size = total_trades // 4
        early_df = df.iloc[:quarter_size]
        late_df = df.iloc[-quarter_size:]

        # Direction accuracy over time
        rolling_direction = df['favorable_move'].rolling(50).mean() * 100
        rolling_direction = rolling_direction.dropna().tolist()

        # Favorable entry rate over time
        rolling_favorable_entry = df['favorable_entry'].rolling(50).mean() * 100
        rolling_favorable_entry = rolling_favorable_entry.dropna().tolist()

        # Rolling Sortino ratio
        def rolling_sortino_calc(window):
            mean = window.mean()
            downside = window[window < 0].std()
            return mean / downside if downside > 0 else 0
        rolling_sortino = df['pnl'].rolling(50).apply(rolling_sortino_calc, raw=False)
        rolling_sortino = rolling_sortino.dropna().tolist()

        # Win/Loss magnitude ratio over time
        def rolling_wl_ratio_calc(window):
            wins = window[window > 0]
            losses = window[window < 0]
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            return min(avg_win / avg_loss, 5) if avg_loss > 0 else 5  # Cap at 5
        rolling_wl_ratio = df['pnl'].rolling(50).apply(rolling_wl_ratio_calc, raw=False)
        rolling_wl_ratio = rolling_wl_ratio.dropna().tolist()

        # Action distribution over time (rolling % of each action)
        # Since we only have trades (not HOLDs), track BUY vs SELL ratio
        action_buy = (df['action'] == 'BUY').astype(int)
        action_sell = (df['action'] == 'SELL').astype(int)
        rolling_buy = action_buy.rolling(50).mean() * 100
        rolling_sell = action_sell.rolling(50).mean() * 100
        # HOLD is implicit (100 - buy - sell in actual decisions, but in trade data it's 0)
        action_dist_hold = [0] * len(rolling_buy.dropna())  # No HOLDs in trade data
        action_dist_buy = rolling_buy.dropna().tolist()
        action_dist_sell = rolling_sell.dropna().tolist()

        # Duration quartile analysis
        duration_quartiles_edges = np.percentile(df['duration_sec'], [0, 25, 50, 75, 100])
        duration_labels = ['Q1 (Fastest)', 'Q2', 'Q3', 'Q4 (Slowest)']
        df['duration_quartile'] = pd.cut(df['duration_sec'],
                                          bins=duration_quartiles_edges,
                                          labels=duration_labels,
                                          include_lowest=True)
        duration_quartile_pnl = df.groupby('duration_quartile', observed=True)['pnl'].mean()

        # Early vs Late comparison metrics
        early_direction_acc = early_df['favorable_move'].mean() * 100 if len(early_df) > 0 else 0
        late_direction_acc = late_df['favorable_move'].mean() * 100 if len(late_df) > 0 else 0

        early_favorable_entry = early_df['favorable_entry'].mean() * 100 if len(early_df) > 0 else 0
        late_favorable_entry = late_df['favorable_entry'].mean() * 100 if len(late_df) > 0 else 0

        early_wins = early_df[early_df['pnl'] > 0]['pnl']
        late_wins = late_df[late_df['pnl'] > 0]['pnl']
        early_losses = early_df[early_df['pnl'] < 0]['pnl']
        late_losses = late_df[late_df['pnl'] < 0]['pnl']

        early_avg_win = early_wins.mean() if len(early_wins) > 0 else 0
        late_avg_win = late_wins.mean() if len(late_wins) > 0 else 0
        early_avg_loss = early_losses.mean() if len(early_losses) > 0 else 0
        late_avg_loss = late_losses.mean() if len(late_losses) > 0 else 0

        early_wl_ratio = early_avg_win / abs(early_avg_loss) if early_avg_loss != 0 else 0
        late_wl_ratio = late_avg_win / abs(late_avg_loss) if late_avg_loss != 0 else 0

        early_sharpe = early_df['pnl'].mean() / early_df['pnl'].std() if early_df['pnl'].std() > 0 else 0
        late_sharpe = late_df['pnl'].mean() / late_df['pnl'].std() if late_df['pnl'].std() > 0 else 0

        learning_metrics = {
            'rolling_direction_accuracy': rolling_direction,
            'rolling_favorable_entry': rolling_favorable_entry,
            'rolling_sortino': rolling_sortino,
            'rolling_win_loss_ratio': rolling_wl_ratio,
            'action_dist_hold': action_dist_hold,
            'action_dist_buy': action_dist_buy,
            'action_dist_sell': action_dist_sell,
            'duration_quartiles': {
                'labels': duration_labels,
                'avg_pnl': [float(duration_quartile_pnl.get(l, 0)) for l in duration_labels]
            },
            'early_direction_acc': early_direction_acc,
            'late_direction_acc': late_direction_acc,
            'early_favorable_entry': early_favorable_entry,
            'late_favorable_entry': late_favorable_entry,
            'early_avg_win': early_avg_win,
            'late_avg_win': late_avg_win,
            'early_avg_loss': early_avg_loss,
            'late_avg_loss': late_avg_loss,
            'early_wl_ratio': early_wl_ratio,
            'late_wl_ratio': late_wl_ratio,
            'early_sharpe': early_sharpe,
            'late_sharpe': late_sharpe,
        }

        # Learning-specific insights
        if late_direction_acc > early_direction_acc + 3:
            insights.append({
                'type': 'positive',
                'text': f"<strong>Learning signal:</strong> Direction accuracy improved from {early_direction_acc:.1f}% to {late_direction_acc:.1f}%"
            })
        if late_wl_ratio > early_wl_ratio * 1.2:
            insights.append({
                'type': 'positive',
                'text': f"<strong>Risk management improving:</strong> Win/Loss ratio from {early_wl_ratio:.2f} to {late_wl_ratio:.2f}"
            })
        if late_sharpe > early_sharpe + 0.2:
            insights.append({
                'type': 'positive',
                'text': f"<strong>Risk-adjusted returns improving:</strong> Sharpe from {early_sharpe:.2f} to {late_sharpe:.2f}"
            })

    return {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': win_rate,
        'duration_min': duration_min,
        'avg_pnl': avg_pnl,
        'median_pnl': median_pnl,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'profit_factor': min(profit_factor, 99.99),
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_duration': avg_duration,
        'median_duration': median_duration,
        'roi': roi,
        'edge_score': edge_score,
        'edge_metrics': edge_metrics,
        'equity_curve': equity.tolist(),
        'by_asset': {
            'assets': by_asset['asset'].tolist(),
            'pnl': by_asset['pnl'].tolist()
        },
        'by_action': {
            'actions': by_action['action'].tolist(),
            'pnl': by_action['sum'].tolist()
        },
        'by_prob': {
            'buckets': by_prob['bucket'].astype(str).tolist(),
            'avg_pnl': by_prob['avg'].tolist()
        },
        'by_time': {
            'buckets': by_time['bucket'].astype(str).tolist(),
            'avg_pnl': by_time['avg'].tolist()
        },
        'prob_hist': {
            'bins': ((prob_edges[:-1] + prob_edges[1:]) / 2).tolist(),
            'counts': prob_hist.tolist()
        },
        'duration_hist': {
            'bins': ((duration_edges[:-1] + duration_edges[1:]) / 2).tolist(),
            'counts': duration_hist.tolist()
        },
        'pnl_hist': {
            'bins': ((pnl_edges[:-1] + pnl_edges[1:]) / 2).tolist(),
            'counts': pnl_hist.tolist()
        },
        'rolling_win_rate': rolling_win,
        'rolling_avg_pnl': rolling_avg_pnl,
        'rolling_sharpe': rolling_sharpe,
        'rolling_profit_factor': rolling_pf,
        'cumulative_win_rate': cumulative_win_rate,
        'by_session_phase': {
            'phases': by_session['phase'].astype(str).tolist(),
            'avg_pnl': by_session['avg_pnl'].tolist(),
            'total_pnl': by_session['total_pnl'].tolist(),
            'win_rates': by_session['win_rate'].tolist(),
            'trades': by_session['trades'].tolist(),
        },
        'binance_correlation': binance_correlation_data,
        'price_movement': price_movement_data,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_win_streak': avg_win_streak,
        'avg_loss_streak': avg_loss_streak,
        'current_streak': streaks[-1] if streaks else 0,
        'insights': insights,
        'asset_details': by_asset.to_dict('records'),
        'time_details': by_time.to_dict('records'),
        'prob_details': by_prob.to_dict('records'),
        'learning_metrics': learning_metrics,
    }

@app.route('/')
def index():
    files = get_trade_files()
    selected = files[0]['name'] if files else None
    return render_template_string(HTML_TEMPLATE, files=files, selected=selected)

@app.route('/api/analyze')
def api_analyze():
    filename = request.args.get('file')
    filepath = LOGS_DIR / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        data = analyze_trades(filepath)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Trading Analytics Dashboard")
    print("=" * 60)
    print(f"Open: http://localhost:5002")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5002, debug=False)
