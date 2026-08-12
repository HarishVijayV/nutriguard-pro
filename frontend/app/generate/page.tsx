"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const DAYS = ["day1","day2","day3","day4","day5","day6","day7"];
const DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

export default function GeneratePage() {
  const router = useRouter();
  const logRef = useRef<HTMLDivElement>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [plan, setPlan] = useState<any>(null);
  const [audit, setAudit] = useState<any>(null);
  const [evalData, setEvalData] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [approved, setApproved] = useState(false);
  const [patient, setPatient] = useState<any>(null);
  const [activeDay, setActiveDay] = useState("day1");

  useEffect(() => {
    const p = localStorage.getItem("patientData");
    if (p) setPatient(JSON.parse(p));
  }, []);

  const addLog = (msg: string) =>
    setLogs(p => [...p, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const handleGenerate = async () => {
    if (!patient) { alert("No patient data. Go back to form."); return; }
    setRunning(true);
    setPlan(null);
    setAudit(null);
    setEvalData(null);
    setApproved(false);
    setLogs([]);

    addLog("Connecting to NutriGuard backend...");
    addLog(`Patient: ${patient.name}, ${patient.age}y, ${patient.weight}kg`);

    try {
      const res = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: patient.name,
          age: parseInt(patient.age),
          weight: parseFloat(patient.weight),
          protein_req: parseFloat(patient.protein_req),
          state_name: patient.state_name,
          allergies: patient.allergies,
        }),
      });

      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        addLog(`Backend rejected the request (HTTP ${res.status}). ${detail.slice(0, 200)}`);
        return;
      }
      if (!res.body) {
        addLog("No response stream from backend.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      const handleFrame = (frame: string) => {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6).trim();
          if (!json) continue;
          try {
            const data = JSON.parse(json);
            if (data.log) addLog(data.log);
            if (data.done) {
              if (data.plan) setPlan(data.plan);
              if (data.audit) setAudit(data.audit);
              if (data.eval) setEvalData(data.eval);
              if (data.error) addLog(`ERROR: ${data.error}`);
            }
          } catch {
            addLog("Skipped an unreadable message from the backend.");
          }
        }
      };

      // Buffer across reads: one chunk can split a "data: ..." line in half,
      // and stream:true keeps multi-byte characters intact at the boundary.
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";   // keep the incomplete tail for next read
        for (const frame of frames) handleFrame(frame);
      }
      buffer += decoder.decode();
      if (buffer.trim()) handleFrame(buffer);
    } catch (e) {
      addLog(`Connection error: ${e}`);
    } finally {
      setRunning(false);
    }
  };

  const handleApprove = () => {
    setApproved(true);
    localStorage.setItem("approvedPlan", JSON.stringify({ plan, audit, eval: evalData, patient }));
    addLog("Doctor approved the plan. Saved.");
  };

  const verdictStyle = (v: string) => {
    if (v === "SAFE") return { color: "#16a34a", bg: "#f0fdf4", border: "#bbf7d0" };
    if (v === "UNSAFE") return { color: "#dc2626", bg: "#fef2f2", border: "#fecaca" };
    return { color: "#d97706", bg: "#fffbeb", border: "#fde68a" };
  };

  const agents = [
    { label: "Researcher", sub: "Tavily", color: "#16a34a" },
    { label: "Chef", sub: "Gemini", color: "#2563eb" },
    { label: "Auditor", sub: "RAG", color: "#d97706" },
    { label: "Judge", sub: "Gemini", color: "#7c3aed" },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-900">Agent Workflow</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            LangGraph · Researcher → Chef → Auditor (RAG) → Judge
          </p>
        </div>
        <button
          onClick={() => router.push("/form")}
          className="text-sm text-gray-500 hover:text-gray-800 border border-gray-200 px-3 py-1.5 rounded-lg transition bg-white"
        >
          ← Back
        </button>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-5">
        {/* Patient info */}
        {patient && (
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-3 flex flex-wrap gap-4 items-center">
            <span className="text-sm font-semibold text-gray-800">{patient.name}</span>
            <span className="text-gray-300">·</span>
            <span className="text-sm text-gray-500">{patient.age} years</span>
            <span className="text-gray-300">·</span>
            <span className="text-sm text-gray-500">{patient.weight} kg</span>
            <span className="text-gray-300">·</span>
            <span className="text-sm text-gray-500">{patient.state_name}</span>
            <span className="text-gray-300">·</span>
            <span className="text-sm text-gray-500">Protein: {patient.protein_req} g/day</span>
          </div>
        )}

        {/* Agent flow */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-3 flex items-center gap-2 flex-wrap">
          {agents.map((a, i) => (
            <div key={a.label} className="flex items-center gap-2">
              <div
                className="px-3 py-1.5 rounded-lg text-xs font-semibold"
                style={{ background: a.color + "12", border: `1px solid ${a.color}30`, color: a.color }}
              >
                <div>{a.label}</div>
                <div className="text-center" style={{ fontSize: "10px", opacity: 0.7 }}>{a.sub}</div>
              </div>
              {i < agents.length - 1 && <span className="text-gray-300 text-sm">→</span>}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Left */}
          <div className="space-y-4">
            <button
              onClick={handleGenerate}
              disabled={running}
              className="w-full py-3.5 rounded-xl font-bold text-white text-sm transition bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {running ? "Agents working..." : "Start Agent Workflow →"}
            </button>

            {/* Logs */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-2.5 border-b border-gray-100 flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${running ? "bg-green-500 animate-pulse" : "bg-gray-300"}`} />
                <span className="text-xs font-semibold text-gray-600">Live Agent Logs</span>
              </div>
              <div
                ref={logRef}
                className="bg-gray-50 border-t border-gray-100 p-3 h-72 overflow-y-auto font-mono text-xs space-y-1"
              >
                {logs.length === 0 && <p className="text-gray-400">Click Start to begin...</p>}
                {logs.map((l, i) => (
                  <p
                    key={i}
                    className={
                      l.includes("ERROR") ? "text-red-500"
                      : l.includes("approved") ? "text-green-600"
                      : "text-gray-600"
                    }
                  >
                    {l}
                  </p>
                ))}
                {running && <p className="text-blue-500 animate-pulse">Processing...</p>}
              </div>
            </div>

            {/* Audit */}
            {audit && (
              <div
                className="rounded-xl p-4 border text-sm"
                style={{
                  background: audit.status === "APPROVED" ? "#f0fdf4" : "#fef2f2",
                  borderColor: audit.status === "APPROVED" ? "#bbf7d0" : "#fecaca",
                }}
              >
                <p className="font-semibold mb-1" style={{ color: audit.status === "APPROVED" ? "#16a34a" : "#dc2626" }}>
                  Audit: {audit.status}
                </p>
                <p className="text-xs text-gray-600">{audit.message}</p>
                {audit.flags?.length > 0 && (
                  <p className="text-xs mt-1 text-red-600">Flagged: {audit.flags.join(", ")}</p>
                )}
              </div>
            )}
          </div>

          {/* Right — 7-day plan */}
          <div>
            {!plan && !running && (
              <div className="bg-white rounded-2xl border border-gray-200 shadow-sm h-full flex items-center justify-center py-20">
                <p className="text-sm text-gray-400 text-center">7-day plan will appear here after generation</p>
              </div>
            )}

            {plan && (() => {
              const vs = verdictStyle(plan.verdict);
              return (
                <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-gray-100 flex items-center justify-between">
                    <span className="text-sm font-bold text-gray-800">7-Day Meal Plan</span>
                    <span
                      className="text-xs font-bold px-3 py-1 rounded-full border"
                      style={{ color: vs.color, background: vs.bg, borderColor: vs.border }}
                    >
                      {plan.verdict}
                    </span>
                  </div>

                  <div className="p-4 space-y-3">
                    {plan.summary && (
                      <p className="text-xs text-gray-500 bg-gray-50 rounded-lg p-3 leading-relaxed">{plan.summary}</p>
                    )}

                    {plan.protein_note && (
                      <div className="rounded-lg p-3 border border-blue-200 bg-blue-50 text-xs text-blue-700">
                        {plan.protein_note}
                      </div>
                    )}

                    {/* Day tabs */}
                    <div className="flex gap-1.5 flex-wrap">
                      {DAYS.map((d, i) => (
                        <button
                          key={d}
                          onClick={() => setActiveDay(d)}
                          className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
                            activeDay === d
                              ? "bg-purple-600 text-white"
                              : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                          }`}
                        >
                          {DAY_LABELS[i]}
                        </button>
                      ))}
                    </div>

                    {/* Meals */}
                    {plan.plan?.[activeDay] && (
                      <div className="space-y-2">
                        {["breakfast","lunch","dinner","snack"].map(meal => (
                          <div key={meal} className="rounded-lg bg-gray-50 border border-gray-100 p-3">
                            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">{meal}</p>
                            <p className="text-sm text-gray-800">{plan.plan[activeDay][meal] || "—"}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {!approved ? (
                      <button
                        onClick={handleApprove}
                        className="w-full py-3 rounded-xl text-white text-sm font-bold transition bg-green-600 hover:bg-green-700"
                      >
                        Doctor Approves Plan
                      </button>
                    ) : (
                      <div className="space-y-2">
                        <div className="text-center py-2.5 rounded-xl text-sm font-semibold bg-green-50 text-green-700 border border-green-200">
                          Plan Approved & Saved
                        </div>
                        <button
                          onClick={() => router.push("/evaluation")}
                          className="w-full py-3 rounded-xl text-white text-sm font-bold transition bg-amber-600 hover:bg-amber-700"
                        >
                          View Evaluation →
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
}
