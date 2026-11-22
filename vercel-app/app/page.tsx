"use client";
import { useState, useRef, useCallback, useEffect } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */
type Project = { id: string; name: string };
type Branding = { id: string; name: string };
type DriveFile = { id: string; name: string; size?: string };
type FolderStatus = {
  branding: Branding;
  ok: boolean;
  error?: string;
  output_path: string;
  folder_id: string | null;
  files: DriveFile[];
};
type ProcessResult = {
  success?: boolean;
  branding_name: string;
  file_name: string;
  file_id?: string;
  drive_url?: string;
  output_path?: string;
  processing_time?: number;
  file_size?: number;
  error?: string;
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
  return json as T;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function Home() {
  /* auth */
  const [authed, setAuthed] = useState(false);
  const [authError, setAuthError] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const pwRef = useRef<HTMLInputElement>(null);

  /* drive */
  const [projects, setProjects] = useState<Project[]>([]);
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);

  /* selections */
  const [search, setSearch] = useState("");
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [brandings, setBrandings] = useState<Branding[]>([]);
  const [selectedBrandings, setSelectedBrandings] = useState<Branding[]>([]);
  const [brandingsLoading, setBrandingsLoading] = useState(false);
  const [fileTypes, setFileTypes] = useState<string[]>([]);
  const [fileTypesLoading, setFileTypesLoading] = useState(false);
  const [selectedType, setSelectedType] = useState("");
  const [includeFooter, setIncludeFooter] = useState(true);

  /* existing files */
  const [folderStatuses, setFolderStatuses] = useState<FolderStatus[]>([]);
  const [existingLoading, setExistingLoading] = useState(false);
  const [replaceAll, setReplaceAll] = useState(false);
  const [replaceMap, setReplaceMap] = useState<Record<string, boolean>>({});

  /* upload & process */
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [results, setResults] = useState<ProcessResult[]>([]);
  const [processError, setProcessError] = useState("");

  /* ---- auth ---- */
  const doLogin = useCallback(async () => {
    setPwBusy(true);
    setAuthError("");
    try {
      const { ok } = await api<{ ok: boolean }>("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: pwRef.current?.value || "" }),
      });
      if (ok) setAuthed(true);
      else setAuthError("Incorrect password");
    } catch (e: any) {
      setAuthError(e.message);
    } finally {
      setPwBusy(false);
    }
  }, []);

  /* ---- connect ---- */
  const doConnect = useCallback(async () => {
    setConnecting(true);
    try {
      const { projects: p } = await api<{ projects: Project[] }>("/api/connect");
      setProjects(p);
      setConnected(true);
    } catch (e: any) {
      alert("Drive connect failed: " + e.message);
    } finally {
      setConnecting(false);
    }
  }, []);

  /* ---- project change → load brandings ---- */
  useEffect(() => {
    if (!selectedProject) {
      setBrandings([]);
      setSelectedBrandings([]);
      return;
    }
    let cancelled = false;
    (async () => {
      setBrandingsLoading(true);
      try {
        const { brandings: b } = await api<{ brandings: Branding[] }>(
          `/api/brandings?project_id=${selectedProject.id}`
        );
        if (!cancelled) {
          setBrandings(b);
          setSelectedBrandings(b); // all selected by default
        }
      } catch {
        if (!cancelled) setBrandings([]);
      } finally {
        if (!cancelled) setBrandingsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedProject]);

  /* ---- brandings change → load file types ---- */
  useEffect(() => {
    if (!selectedBrandings.length) {
      setFileTypes([]);
      setSelectedType("");
      return;
    }
    let cancelled = false;
    (async () => {
      setFileTypesLoading(true);
      try {
        const allTypes = new Set<string>();
        await Promise.all(
          selectedBrandings.map(async (b) => {
            const { file_types } = await api<{ file_types: string[] }>(
              `/api/file_types?branding_id=${b.id}`
            );
            file_types.forEach((t) => allTypes.add(t));
          })
        );
        if (!cancelled) {
          const sorted = Array.from(allTypes).sort();
          setFileTypes(sorted);
          if (sorted.length && !sorted.includes(selectedType)) setSelectedType(sorted[0]);
        }
      } catch {
        if (!cancelled) setFileTypes([]);
      } finally {
        if (!cancelled) setFileTypesLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedBrandings]);

  /* ---- file type change → load existing files ---- */
  useEffect(() => {
    if (!selectedProject || !selectedBrandings.length || !selectedType) {
      setFolderStatuses([]);
      return;
    }
    let cancelled = false;
    (async () => {
      setExistingLoading(true);
      try {
        const statuses = await Promise.all(
          selectedBrandings.map(async (b) => {
            const data = await api<any>(
              `/api/existing_files?project_name=${encodeURIComponent(selectedProject.name)}&branding_name=${encodeURIComponent(b.name)}&file_type=${encodeURIComponent(selectedType)}`
            );
            return { branding: b, ...data } as FolderStatus;
          })
        );
        if (!cancelled) setFolderStatuses(statuses);
      } catch {
        if (!cancelled) setFolderStatuses([]);
      } finally {
        if (!cancelled) setExistingLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedProject, selectedBrandings, selectedType]);

  /* ---- process ---- */
  const doProcess = useCallback(async () => {
    if (!selectedProject || !selectedBrandings.length || !selectedType || !files.length) return;
    setProcessing(true);
    setProgress(0);
    setStatusMsg("");
    setResults([]);
    setProcessError("");

    const jobs: { file: File; branding: Branding }[] = [];
    for (const b of selectedBrandings) {
      for (const f of files) {
        jobs.push({ file: f, branding: b });
      }
    }
    const total = jobs.length;
    const collected: ProcessResult[] = [];
    let done = 0;

    try {
      for (const job of jobs) {
        const shouldReplace = replaceAll || !!replaceMap[job.branding.id];
        const isFirstFileForBranding = done === 0 || jobs[done - 1]?.branding.id !== job.branding.id;

        setStatusMsg(`[${done + 1}/${total}] ${job.branding.name} — ${job.file.name}`);

        const form = new FormData();
        form.append("file", job.file);
        form.append("project_name", selectedProject.name);
        form.append("branding_name", job.branding.name);
        form.append("file_type", selectedType);
        form.append("include_footer", includeFooter ? "1" : "0");
        form.append("replace_existing", shouldReplace && isFirstFileForBranding ? "1" : "0");

        try {
          const result = await api<ProcessResult>("/api/process", {
            method: "POST",
            body: form,
          });
          collected.push(result);
        } catch (e: any) {
          collected.push({
            branding_name: job.branding.name,
            file_name: job.file.name,
            error: e.message,
          });
        }
        done++;
        setProgress(done / total);
      }
      setResults(collected);
    } catch (e: any) {
      setProcessError(e.message);
    } finally {
      setProcessing(false);
      setStatusMsg("");
    }
  }, [selectedProject, selectedBrandings, selectedType, files, includeFooter, replaceAll, replaceMap]);

  /* ---- filtered projects ---- */
  const filtered = search
    ? projects.filter(
        (p) =>
          p.name.toLowerCase().includes(search.toLowerCase()) ||
          p.id.toLowerCase().includes(search.toLowerCase())
      )
    : projects;

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */

  /* ---- password gate ---- */
  if (!authed) {
    return (
      <div className="container password-page">
        <div style={{ fontSize: "3rem" }}>🔒</div>
        <h1>PDF Processor</h1>
        <p>Please enter the password to access this application.</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            doLogin();
          }}
        >
          <input ref={pwRef} type="password" placeholder="Password" autoFocus />
          <button className="btn btn-primary" type="submit" disabled={pwBusy}>
            {pwBusy ? <span className="spinner" /> : "Enter"}
          </button>
        </form>
        {authError && <div className="alert alert-error">{authError}</div>}
      </div>
    );
  }

  /* ---- main app ---- */
  return (
    <div className="container">
      <h1>🎨 PDF Processor</h1>
      <p className="subtitle">Simple, reliable PDF processing with instant feedback</p>

      {/* Connect */}
      {!connected ? (
        <div className="card" style={{ textAlign: "center" }}>
          <p style={{ marginBottom: "1rem" }}>Connect to Google Drive to get started.</p>
          <button className="btn btn-primary" onClick={doConnect} disabled={connecting}>
            {connecting ? (
              <>
                <span className="spinner" /> Connecting…
              </>
            ) : (
              "🔌 Connect to Google Drive"
            )}
          </button>
        </div>
      ) : (
        <>
          <div className="alert alert-success">
            ✅ Connected — {projects.length} projects loaded
          </div>

          {/* Step 1: Project */}
          <h2>1️⃣ Select Project</h2>
          <input
            type="text"
            placeholder="🔍 Search project by name or ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <div className="caption" style={{ margin: ".3rem 0" }}>
              Found {filtered.length} project(s)
            </div>
          )}
          <select
            value={selectedProject?.id || ""}
            onChange={(e) => {
              const p = projects.find((x) => x.id === e.target.value) || null;
              setSelectedProject(p);
              setSelectedBrandings([]);
              setFileTypes([]);
              setSelectedType("");
              setFolderStatuses([]);
              setFiles([]);
              setResults([]);
            }}
            style={{ marginTop: ".4rem" }}
          >
            <option value="">— select a project —</option>
            {filtered.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          {/* Step 2: Brandings */}
          {selectedProject && (
            <>
              <h2>2️⃣ Select Branding</h2>
              <p className="caption">
                Each PDF is processed and uploaded to every selected branding folder.
              </p>
              {brandingsLoading ? (
                <p>
                  <span className="spinner" /> Loading brandings…
                </p>
              ) : brandings.length === 0 ? (
                <div className="alert alert-warn">
                  No branding folders found inside this project.
                </div>
              ) : (
                <>
                  <div className="chip-bar">
                    {brandings.map((b) => {
                      const active = selectedBrandings.some((s) => s.id === b.id);
                      return (
                        <span
                          key={b.id}
                          className={`chip ${active ? "active" : ""}`}
                          onClick={() =>
                            setSelectedBrandings((prev) =>
                              active ? prev.filter((s) => s.id !== b.id) : [...prev, b]
                            )
                          }
                        >
                          {b.name}
                          {active && <span className="x">×</span>}
                        </span>
                      );
                    })}
                  </div>
                  {selectedBrandings.length > 0 && (
                    <div className="alert alert-info">
                      ✅ {selectedBrandings.length} branding(s) selected:{" "}
                      {selectedBrandings.map((b) => b.name).join(", ")}
                    </div>
                  )}
                  {selectedBrandings.length === 0 && (
                    <div className="alert alert-warn">Select at least one branding.</div>
                  )}
                </>
              )}
            </>
          )}

          {/* Step 3: File Type */}
          {selectedBrandings.length > 0 && (
            <>
              <h2>3️⃣ Select File Type</h2>
              {fileTypesLoading ? (
                <p>
                  <span className="spinner" /> Loading file types…
                </p>
              ) : fileTypes.length === 0 ? (
                <div className="alert alert-warn">No file type folders found in Raw folder.</div>
              ) : (
                <select value={selectedType} onChange={(e) => setSelectedType(e.target.value)}>
                  {fileTypes.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              )}
            </>
          )}

          {/* Step 4: Existing Files */}
          {selectedType && (
            <>
              <h2>4️⃣ Existing files in Drive</h2>
              {existingLoading ? (
                <p>
                  <span className="spinner" /> Checking existing files…
                </p>
              ) : (
                <>
                  {folderStatuses.some((f) => f.files.length > 0) && (
                    <div className="alert alert-warn">
                      ⚠️{" "}
                      {folderStatuses.reduce((n, f) => n + f.files.length, 0)} existing
                      file(s) found in{" "}
                      {folderStatuses.filter((f) => f.files.length > 0).length} folder(s).
                    </div>
                  )}
                  <label className="toggle" style={{ margin: ".5rem 0" }}>
                    <input
                      type="checkbox"
                      checked={replaceAll}
                      onChange={(e) => setReplaceAll(e.target.checked)}
                    />
                    Remove current files in ALL selected folders
                  </label>
                  {folderStatuses.map((fs) => (
                    <details key={fs.branding.id} className="expander card" open={fs.files.length > 0}>
                      <summary>
                        {fs.branding.name} — {fs.files.length} file(s)
                      </summary>
                      <div className="caption">{fs.output_path}</div>
                      {!fs.ok && (
                        <div className="alert alert-error">{fs.error}</div>
                      )}
                      {fs.files.length === 0 && fs.ok && (
                        <div className="alert alert-info" style={{ marginTop: ".3rem" }}>
                          This folder is empty.
                        </div>
                      )}
                      <ul style={{ paddingLeft: "1.2rem", fontSize: ".9rem" }}>
                        {fs.files.map((f) => (
                          <li key={f.id}>{f.name}</li>
                        ))}
                      </ul>
                      <label className="toggle" style={{ marginTop: ".4rem" }}>
                        <input
                          type="checkbox"
                          disabled={replaceAll}
                          checked={replaceAll || !!replaceMap[fs.branding.id]}
                          onChange={(e) =>
                            setReplaceMap((m) => ({ ...m, [fs.branding.id]: e.target.checked }))
                          }
                        />
                        Remove current files in {fs.branding.name} only
                      </label>
                    </details>
                  ))}
                </>
              )}
            </>
          )}

          {/* Step 5: Footer Options */}
          {selectedType && (
            <>
              <h2>5️⃣ Footer Options</h2>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={includeFooter}
                  onChange={(e) => setIncludeFooter(e.target.checked)}
                />
                Include footer image
              </label>
              <div className="alert alert-info" style={{ marginTop: ".3rem" }}>
                {includeFooter ? "✅ Will add: Watermark + Footer" : "ℹ️ Watermark only (no footer)"}
              </div>
            </>
          )}

          {/* Step 6: Upload */}
          {selectedType && (
            <>
              <h2>6️⃣ Upload PDFs</h2>
              <div
                className={`file-drop ${files.length ? "active" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const dropped = Array.from(e.dataTransfer.files).filter((f) =>
                    f.name.toLowerCase().endsWith(".pdf")
                  );
                  setFiles((prev) => [...prev, ...dropped]);
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={(e) => {
                    const selected = Array.from(e.target.files || []);
                    setFiles((prev) => [...prev, ...selected]);
                    e.target.value = "";
                  }}
                />
                {files.length === 0 ? (
                  <p>📄 Click or drag PDFs here (up to 50)</p>
                ) : (
                  <p>
                    ✅ {files.length} file(s) —{" "}
                    {(files.reduce((s, f) => s + f.size, 0) / 1024 / 1024).toFixed(2)} MB total.
                    Click to add more.
                  </p>
                )}
              </div>
              {files.length > 0 && (
                <>
                  <details className="expander" style={{ margin: ".5rem 0" }}>
                    <summary>View selected files</summary>
                    <ul style={{ paddingLeft: "1.2rem", fontSize: ".9rem" }}>
                      {files.map((f, i) => (
                        <li key={i}>
                          {f.name} ({(f.size / 1024 / 1024).toFixed(2)} MB)
                          <button
                            className="btn-sm btn-outline"
                            style={{ marginLeft: ".5rem" }}
                            onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                          >
                            ✕
                          </button>
                        </li>
                      ))}
                    </ul>
                  </details>

                  <button
                    className="btn btn-primary btn-block"
                    style={{ marginTop: ".5rem" }}
                    disabled={processing || !selectedBrandings.length}
                    onClick={doProcess}
                  >
                    {processing ? (
                      <>
                        <span className="spinner" /> Processing…
                      </>
                    ) : (
                      `🚀 Process ${files.length} PDF(s) × ${selectedBrandings.length} branding(s)`
                    )}
                  </button>
                </>
              )}

              {/* Progress */}
              {processing && (
                <div style={{ marginTop: ".75rem" }}>
                  <div className="progress-bar">
                    <div className="fill" style={{ width: `${Math.round(progress * 100)}%` }} />
                  </div>
                  <div className="caption">{statusMsg}</div>
                </div>
              )}

              {/* Results */}
              {results.length > 0 && (
                <div className="card" style={{ marginTop: "1rem" }}>
                  <h3 style={{ marginBottom: ".5rem" }}>
                    ✅ {results.filter((r) => r.success).length} of {results.length} upload(s)
                    completed
                  </h3>
                  {results.map((r, i) => (
                    <div key={i} className="result-item">
                      {r.success ? (
                        <>
                          <strong>{r.branding_name}</strong> — {r.file_name} —{" "}
                          {((r.file_size || 0) / 1024 / 1024).toFixed(2)} MB,{" "}
                          {r.processing_time?.toFixed(1)}s
                          <div className="caption">📁 {r.output_path}</div>
                          {r.drive_url && (
                            <a
                              href={r.drive_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="drive-link"
                            >
                              🔗 Open in Drive
                            </a>
                          )}
                        </>
                      ) : (
                        <div className="alert alert-error">
                          ❌ {r.branding_name} / {r.file_name}: {r.error}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {processError && (
                <div className="alert alert-error" style={{ marginTop: ".5rem" }}>
                  ❌ {processError}
                </div>
              )}
            </>
          )}
        </>
      )}

      <div className="footer">
        PDF Processor v2.1 (Vercel) — Fahad Javed (289) 536-9724 | No webhooks, no complexity, just
        works ✨
      </div>
    </div>
  );
}
