import { useState } from "react";
import "./App.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const API_URL = "http://127.0.0.1:8000";

const MODEL_OPTIONS = [
  { id: "sma_crossover", label: "SMA Crossover" },
  { id: "logistic", label: "Logistic Regression" },
  { id: "random_forest", label: "Random Forest" },
];

function App() {
  const [ticker, setTicker] = useState("");
  const [quote, setQuote] = useState(null);
  const [status, setStatus] = useState("");
  const [loadingQuote, setLoadingQuote] = useState(false);

  const [years, setYears] = useState(3);
  const [selectedModels, setSelectedModels] = useState(["sma_crossover"]);
  const [loadingBacktest, setLoadingBacktest] = useState(false);
  const [backtests, setBacktests] = useState([]);

  // New controls
  const [smaFast, setSmaFast] = useState(10);
  const [smaSlow, setSmaSlow] = useState(50);
  const [probThreshold, setProbThreshold] = useState(0.55);

  async function handleLoadQuote() {
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) {
      setStatus("Please enter a ticker symbol.");
      setQuote(null);
      return;
    }

    setStatus("");
    setLoadingQuote(true);
    setQuote(null);

    try {
      const res = await fetch(
        `${API_URL}/api/quote?symbol=${encodeURIComponent(symbol)}`
      );
      if (!res.ok) throw new Error("Quote API error");
      const data = await res.json();
      setQuote(data);
    } catch (err) {
      console.error(err);
      setStatus("Error loading quote.");
    } finally {
      setLoadingQuote(false);
    }
  }

  function toggleModel(id) {
    setSelectedModels((prev) => {
      if (prev.includes(id)) {
        return prev.filter((m) => m !== id);
      }
      return [...prev, id];
    });
  }

  async function handleRunBacktest() {
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) {
      setStatus("Please enter a ticker symbol first.");
      setBacktests([]);
      return;
    }

    if (selectedModels.length === 0) {
      setStatus("Please select at least one model.");
      setBacktests([]);
      return;
    }

    setStatus("");
    setLoadingBacktest(true);
    setBacktests([]);

    try {
      const promises = selectedModels.map(async (modelId) => {
        const res = await fetch(`${API_URL}/api/backtest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol,
            model_type: modelId,
            years,
            sma_fast: smaFast,
            sma_slow: smaSlow,
            // keep indicator windows as defaults for now
            rsi_window: 14,
            vol_window: 20,
            mom_lookback: 10,
            prob_threshold: probThreshold,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Backtest API error for ${modelId}`);
        }

        const data = await res.json();
        return data;
      });

      const results = await Promise.all(promises);
      setBacktests(results);
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Error running backtest.");
    } finally {
      setLoadingBacktest(false);
    }
  }

  return (
    <div className="app">
      <div className="card">
        <h1 className="title">Quant Snapshot</h1>
        <p className="subtitle">
          Enter a stock/ETF ticker. Load a quote, then run backtests with one or
          more models over a chosen window.
        </p>

        {/* Ticker / quote section */}
        <div className="section">
          <label className="label" htmlFor="ticker-input">
            Ticker
          </label>
          <div className="row">
            <input
              id="ticker-input"
              className="input"
              placeholder="AAPL, SPY, TSLA..."
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
            />
            <button
              className="btn primary"
              onClick={handleLoadQuote}
              disabled={loadingQuote}
            >
              {loadingQuote ? "Loading..." : "Load"}
            </button>
          </div>

          <div className="chip-row">
            {["AAPL", "MSFT", "SPY", "QQQ", "TSLA"].map((s) => (
              <button
                key={s}
                type="button"
                className="chip"
                onClick={() => setTicker(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {status && <div className="status">{status}</div>}

        {quote && (
          <div className="section">
            <div className="quote-card">
              <div className="quote-header">
                <span className="symbol">{quote.symbol}</span>
                <span className="price">${quote.price.toFixed(2)}</span>
              </div>
              <div className="stat-row">
                <span>P/E</span>
                <span>{quote.pe?.toFixed(2) ?? "—"}</span>
              </div>
              <div className="stat-row">
                <span>Market Cap</span>
                <span>
                  {quote.market_cap
                    ? (quote.market_cap / 1e9).toFixed(1) + "B"
                    : "—"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Backtest controls */}
        <div className="section">
          <h2 className="section-title">Backtest Settings</h2>

          <div className="control-row">
            <div className="control">
              <div className="label">Window</div>
              <select
                className="select"
                value={years}
                onChange={(e) => setYears(Number(e.target.value))}
              >
                <option value={1}>1 Year</option>
                <option value={3}>3 Years</option>
                <option value={5}>5 Years</option>
              </select>
            </div>
          </div>

          <div className="model-options">
            <div className="label">Models</div>
            <div className="model-pill-row">
              {MODEL_OPTIONS.map((opt) => (
                <label key={opt.id} className="checkbox-pill">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(opt.id)}
                    onChange={() => toggleModel(opt.id)}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* New param controls */}
          <div className="param-block">
            <div className="label">Strategy Parameters</div>
            <div className="param-row">
              <div className="param-control">
                <div className="param-label">Fast SMA</div>
                <input
                  type="number"
                  min={2}
                  max={100}
                  className="input-compact"
                  value={smaFast}
                  onChange={(e) =>
                    setSmaFast(Number(e.target.value) || 0)
                  }
                />
              </div>
              <div className="param-control">
                <div className="param-label">Slow SMA</div>
                <input
                  type="number"
                  min={3}
                  max={300}
                  className="input-compact"
                  value={smaSlow}
                  onChange={(e) =>
                    setSmaSlow(Number(e.target.value) || 0)
                  }
                />
              </div>
            </div>

            <div className="param-row">
              <div className="param-control">
                <div className="param-label">ML Prob Threshold</div>
                <input
                  type="number"
                  step={0.01}
                  min={0.5}
                  max={0.8}
                  className="input-compact"
                  value={probThreshold}
                  onChange={(e) =>
                    setProbThreshold(Number(e.target.value) || 0)
                  }
                />
              </div>
            </div>
          </div>

          <button
            className="btn secondary full"
            onClick={handleRunBacktest}
            disabled={loadingBacktest}
          >
            {loadingBacktest ? "Running backtests..." : "Run Backtests"}
          </button>
        </div>

        {/* Backtest results */}
        {backtests.length > 0 && (
          <div className="section">
            <h2 className="section-title">Results</h2>

            {backtests.map((bt) => {
              const chartData = bt.equity_curve.map((row, idx) => {
                const [date, eq] = row;
                const benchRow = bt.benchmark_curve[idx];
                const benchVal = benchRow ? benchRow[1] : null;
                return {
                  date,
                  strategy: eq,
                  benchmark: benchVal,
                };
              });

              return (
                <div key={bt.model_type} className="result-card">
                  <div className="pill-row">
                    <span className="pill">
                      {bt.symbol} · {bt.years}y
                    </span>
                    <span className="pill">
                      {bt.model_type.replace("_", " ")}
                    </span>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric">
                      <div className="metric-label">CAGR</div>
                      <div className="metric-value">
                        {(bt.metrics.cagr * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Sharpe</div>
                      <div className="metric-value">
                        {bt.metrics.sharpe.toFixed(2)}
                      </div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Max Drawdown</div>
                      <div className="metric-value">
                        {(bt.metrics.max_drawdown * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Hit Rate</div>
                      <div className="metric-value">
                        {(bt.metrics.hit_rate * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart
                        data={chartData}
                        margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                      >
                        <XAxis dataKey="date" hide />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip
                          formatter={(value) =>
                            typeof value === "number"
                              ? value.toFixed(3)
                              : value
                          }
                          labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: "0.7rem", paddingTop: 4 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="strategy"
                          stroke="#22c55e"
                          dot={false}
                          strokeWidth={2}
                          name="Strategy"
                        />
                        <Line
                          type="monotone"
                          dataKey="benchmark"
                          stroke="#38bdf8"
                          dot={false}
                          strokeWidth={1.5}
                          name="Benchmark"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mini-table">
                    <div className="mini-table-header">
                      <span>Date</span>
                      <span>Strategy</span>
                      <span>Benchmark</span>
                    </div>

                    {bt.equity_curve.slice(-10).map((row, idx) => {
                      const [date, eq] = row;
                      const benchRow = bt.benchmark_curve.slice(-10)[idx];
                      const benchVal = benchRow ? benchRow[1] : null;

                      return (
                        <div key={date} className="mini-table-row">
                          <span>{date}</span>
                          <span>{eq.toFixed(3)}</span>
                          <span>
                            {benchVal !== null ? benchVal.toFixed(3) : "—"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <p className="footer">
          For research / education only. Not investment advice.
        </p>
      </div>
    </div>
  );
}

export default App;
