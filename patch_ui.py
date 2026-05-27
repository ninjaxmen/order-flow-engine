import re

html_path = 'utils/dashboard.html'
html = open(html_path, 'r', encoding='utf-8').read()

# Add TV container above Footprint Grid
tv_container = '''
            <div class="panel-header" style="border-top: 1px solid var(--border-glow)">
                <h2>Price Action & Liquidations</h2>
            </div>
            <div id="tvchart" style="width: 100%; height: 250px;"></div>
'''
html = html.replace('<div class="footprint-content">', tv_container + '<div class="footprint-content">')

# Add Slippage Table below Real-Time Completed Trades
slippage_table = '''
                    <div style="border-top: 1px solid var(--border-glow); padding-top: 8px; margin-top: 8px;">
                        <div class="agent-stat-label" style="margin-bottom: 4px;">Execution Slippage Log</div>
                        <div style="max-height: 100px; overflow-y: auto;">
                            <table class="trade-log-table">
                                <thead>
                                    <tr>
                                        <th>COIN</th>
                                        <th>PREDICTED</th>
                                        <th>FILLED</th>
                                        <th style="text-align: right;">SLIPPAGE</th>
                                    </tr>
                                </thead>
                                <tbody id="slippage-log-body">
                                    <!-- Slippage rows -->
                                </tbody>
                            </table>
                        </div>
                    </div>
'''
html = html.replace('<!-- Visual Neural Softmax bars -->', slippage_table + '\n<!-- Visual Neural Softmax bars -->')

# Add Flow Sentiment Gauge
gauge = '''
                    <div style="border-top: 1px solid var(--border-glow); padding-top: 8px; margin-bottom: 8px;">
                        <div class="agent-stat-label" style="margin-bottom: 4px;">Flow Sentiment Gauge (CVD + Imbalance)</div>
                        <div class="progress-track" style="height: 12px; margin: 0;">
                            <div id="sentiment-gauge" class="progress-bar-fill" style="width: 50%; background: linear-gradient(90deg, var(--color-down) 0%, #64748b 50%, var(--color-up) 100%);"></div>
                        </div>
                    </div>
'''
html = html.replace('<!-- Visual Neural Softmax bars -->', gauge + '\n<!-- Visual Neural Softmax bars -->')

open(html_path, 'w', encoding='utf-8').write(html)
print("UI Successfully Injected.")
