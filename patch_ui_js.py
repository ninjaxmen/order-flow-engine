import re

html_path = 'utils/dashboard.html'
html = open(html_path, 'r', encoding='utf-8').read()

js_injection = """
        // TRADINGVIEW CHART INITIALIZATION
        let tvChart = null;
        let candleSeries = null;
        let currentCandle = null;
        let liqLines = [];

        function initTradingView() {
            if (document.getElementById('tvchart') && !tvChart) {
                tvChart = LightweightCharts.createChart(document.getElementById('tvchart'), {
                    layout: { background: { color: 'transparent' }, textColor: '#cbd5e1' },
                    grid: { vertLines: { color: 'rgba(255, 255, 255, 0.05)' }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } },
                    rightPriceScale: { borderColor: 'rgba(255, 255, 255, 0.1)' },
                    timeScale: { borderColor: 'rgba(255, 255, 255, 0.1)', timeVisible: true, secondsVisible: true }
                });
                candleSeries = tvChart.addCandlestickSeries({
                    upColor: '#00ffcc', downColor: '#ff3366', borderVisible: false, wickUpColor: '#00ffcc', wickDownColor: '#ff3366'
                });
            }
        }

        function updateCandlestick(price) {
            initTradingView();
            const now = Math.floor(Date.now() / 1000);
            if (!currentCandle || now > currentCandle.time + 60) {
                // start a new 1-minute candle
                // Truncate to minute boundary to avoid weird rendering
                const minuteTime = now - (now % 60);
                currentCandle = { time: minuteTime, open: price, high: price, low: price, close: price };
            } else {
                currentCandle.high = Math.max(currentCandle.high, price);
                currentCandle.low = Math.min(currentCandle.low, price);
                currentCandle.close = price;
            }
            if(candleSeries) {
                try {
                    candleSeries.update(currentCandle);
                } catch(e) {}
                updateLiquidationHeatmap(price);
            }
        }

        function updateLiquidationHeatmap(currentPrice) {
            // Mock Liquidation clusters around current price
            if (liqLines.length === 0 && candleSeries) {
                liqLines.push(candleSeries.createPriceLine({ price: currentPrice * 1.002, color: 'rgba(255, 51, 102, 0.5)', lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: 'Short Liq Cluster' }));
                liqLines.push(candleSeries.createPriceLine({ price: currentPrice * 0.998, color: 'rgba(0, 255, 204, 0.5)', lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: 'Long Liq Cluster' }));
            }
        }

        function updateSlippageLog(recentTrades) {
            const tbody = document.getElementById("slippage-log-body");
            if (!tbody) return;
            tbody.innerHTML = "";
            recentTrades.forEach(t => {
                let slipStr = (t.slippage_pct * 100).toFixed(4) + "%";
                let slipColor = t.slippage_pct > 0.0005 ? 'var(--color-down)' : 'var(--color-up)';
                tbody.innerHTML += `
                    <tr>
                        <td>${t.asset}</td>
                        <td style="font-family: var(--font-mono);">${(t.predicted_price || t.entry_price).toFixed(4)}</td>
                        <td style="font-family: var(--font-mono);">${t.entry_price.toFixed(4)}</td>
                        <td style="text-align: right; font-family: var(--font-mono); color: ${slipColor};">${slipStr}</td>
                    </tr>
                `;
            });
        }

        function updateSentimentGauge(scannerData) {
            const gauge = document.getElementById("sentiment-gauge");
            if (!gauge || !scannerData["BTCUSD"]) return;
            let cvd = scannerData["BTCUSD"].cvd;
            // Map CVD roughly from -1M to +1M -> 0% to 100%
            let pct = 50 + (cvd / 1000000) * 50;
            pct = Math.max(0, Math.min(100, pct));
            gauge.style.width = pct + "%";
        }
"""

html = html.replace('// Limit to 50 alerts', js_injection + '\n            // Limit to 50 alerts')

tick_injection = """if (data.type === "TICK") {
                    if (data.symbol === activeSymbol) {
                        updateCandlestick(data.price);
                    }"""
html = html.replace('if (data.type === "TICK") {', tick_injection)

agent_injection = """} else if (data.type === "AGENT_UPDATE") {
                    updateSlippageLog(data.recent_trades || []);
                    updateSentimentGauge(data.scanner || {});"""
html = html.replace('} else if (data.type === "AGENT_UPDATE") {', agent_injection)

open(html_path, 'w', encoding='utf-8').write(html)
print("JS Successfully Injected.")
