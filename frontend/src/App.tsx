import { useEffect, useState } from "react";
import { askQuestion, fetchSummary, triggerIngest } from "./api";
import type { AskResponse, SummaryResponse } from "./types";

const starterQuestions = [
  "What are the main system design themes in the corpus?",
  "How does FinRL-X describe modular trading infrastructure?",
  "What evaluation concerns are emphasized in the guidebook?"
];

export function App() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [question, setQuestion] = useState(starterQuestions[0]);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [asking, setAsking] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadSummary();
  }, []);

  async function loadSummary() {
    try {
      setLoadingSummary(true);
      setError(null);
      const data = await fetchSummary();
      setSummary(data);
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoadingSummary(false);
    }
  }

  async function handleIngest() {
    try {
      setIngesting(true);
      setError(null);
      const data = await triggerIngest();
      setSummary(data);
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setIngesting(false);
    }
  }

  async function handleAsk() {
    try {
      setAsking(true);
      setError(null);
      const data = await askQuestion(question);
      setAnswer(data);
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="hero-noise" />
      <main className="layout">
        <section className="hero card">
          <div>
            <p className="eyebrow">Graphify-Inspired Knowledge Graph RAG</p>
            <h1>Production-ready PDF intelligence for graph-grounded retrieval.</h1>
            <p className="lede">
              Build a corpus graph from technical PDFs, surface high-value entities and relations, and answer
              questions with evidence-backed retrieval.
            </p>
          </div>
          <div className="hero-actions">
            <button className="primary" onClick={handleIngest} disabled={ingesting}>
              {ingesting ? "Building graph..." : "Ingest corpus"}
            </button>
            <button className="secondary" onClick={loadSummary} disabled={loadingSummary}>
              Refresh summary
            </button>
          </div>
        </section>

        <section className="stats-grid">
          {[
            ["Documents", summary?.documents ?? 0],
            ["Chunks", summary?.chunks ?? 0],
            ["Entities", summary?.entities ?? 0],
            ["Relations", summary?.relations ?? 0]
          ].map(([label, value]) => (
            <article key={label} className="stat card">
              <span>{label}</span>
              <strong>{loadingSummary ? "..." : value}</strong>
            </article>
          ))}
        </section>

        <section className="content-grid">
          <article className="card panel">
            <div className="panel-header">
              <h2>Ask the corpus</h2>
              <span className="mono">Grounded retrieval + graph context</span>
            </div>
            <div className="question-list">
              {starterQuestions.map((item) => (
                <button key={item} className="question-chip" onClick={() => setQuestion(item)}>
                  {item}
                </button>
              ))}
            </div>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={5}
              placeholder="Ask a detailed question about the PDFs..."
            />
            <button className="primary wide" onClick={handleAsk} disabled={asking}>
              {asking ? "Querying..." : "Generate grounded answer"}
            </button>

            {answer && (
              <div className="answer-block">
                <h3>Answer</h3>
                <div className="answer-meta">
                  <span>{answer.retrieval_mode}</span>
                  <span>{answer.llm_provider}</span>
                  <span>{answer.llm_model ?? "fallback synthesis"}</span>
                </div>
                <p>{answer.answer}</p>
                <div className="evidence-grid">
                  <div>
                    <h4>Evidence</h4>
                    {answer.evidence.map((item) => (
                      <article key={item.chunk_id} className="evidence-card">
                        <div className="evidence-meta">
                          <span>{item.doc_id}</span>
                          <span>{item.score.toFixed(3)}</span>
                        </div>
                        <div className="score-row">
                          <span>lexical {item.lexical_score.toFixed(3)}</span>
                          <span>dense {item.dense_score.toFixed(3)}</span>
                          <span>graph {item.graph_score.toFixed(3)}</span>
                        </div>
                        <p>{item.text}</p>
                      </article>
                    ))}
                  </div>
                  <div>
                    <h4>Graph context</h4>
                    {answer.graph_context.length === 0 && <p className="muted">No graph neighbors matched the query.</p>}
                    {answer.graph_context.map((item, index) => (
                      <article key={`${item.source}-${item.target}-${index}`} className="relation-card">
                        <strong>{item.source}</strong>
                        <span>{item.relation}</span>
                        <strong>{item.target}</strong>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </article>

          <article className="card panel">
            <div className="panel-header">
              <h2>Top entities</h2>
              <span className="mono">Graph hotspots</span>
            </div>
            <div className="entity-list">
              {summary?.top_entities?.map((entity) => (
                <div key={entity.entity_id} className="entity-row">
                  <div>
                    <strong>{entity.name}</strong>
                    <p>{entity.label}</p>
                  </div>
                  <span>{entity.frequency}</span>
                </div>
              ))}
            </div>
          </article>
        </section>

        {error && <section className="card error-banner">{error}</section>}
      </main>
    </div>
  );
}
