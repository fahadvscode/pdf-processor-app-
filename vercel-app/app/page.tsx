"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import type { Whiteout } from "./WhiteoutEditor";
import { FILE_ACCEPT, prepareUploads, type UploadFile } from "./prepareUpload";

const WhiteoutEditor = dynamic(() => import("./WhiteoutEditor"), { ssr: false });

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
  whiteouts?: number;
  error?: string;
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const text = await res.text();
  let json: { error?: string } = {};
  if (text) {
    try {
      json = JSON.parse(text);
    } catch {
      const plain = text.replace(/\s+/g, " ").trim().slice(0, 240);
      throw new Error(plain || `HTTP ${res.status}`);
    }
  }
  if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
  return json as T;
}

const DIRECT_UPLOAD_LIMIT = 4_000_000;

function fileKey(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
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
  const passwordRef = useRef("");

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
  const [files, setFiles] = useState<UploadFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [prepareLabel, setPrepareLabel] = useState("");
  const [addError, setAddError] = useState("");
  const addingRef = useRef(false);
  const addGeneration = useRef(0);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [results, setResults] = useState<ProcessResult[]>([]);
  const [processError, setProcessError] = useState("");
  const [whiteouts, setWhiteouts] = useState<Record<string, Whiteout[]>>({});
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  useEffect(() => {
    addGeneration.current += 1;
    setPrepareLabel("");
    setAddError("");
    setFiles([]);
    setWhiteouts({});
    setEditingIndex(null);
    setResults([]);
    setProcessError("");
    setProgress(0);
    setStatusMsg("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [selectedType]);

  const addIncoming = useCallback(
    async (list: File[]) => {
      if (!list.length || processing || addingRef.current) return;
      addingRef.current = true;
      const generation = addGeneration.current;
      setAddError("");
      setPrepareLabel("Checking files…");
      try {
        const { files: ready, skipped } = await prepareUploads(
          list,
          (label) => {
            if (addGeneration.current === generation) setPrepareLabel(label);
          },
          () => addGeneration.current !== generation
        );
        if (addGeneration.current !== generation) return;
        const room = Math.max(0, 50 - files.length);
        const allowed = ready.slice(0, room);
        if (ready.length > allowed.length) {
          skipped.push(`${ready.length - allowed.length} over the 50 file limit`);
        }
        if (allowed.length) setFiles((prev) => [...prev, ...allowed].slice(0, 50));
        if (skipped.length) {
          const shown = skipped.slice(0, 4).join(", ");
          const extra = skipped.length > 4 ? ` and ${skipped.length - 4} more` : "";
          setAddError(`Skipped: ${shown}${extra}`);
        }
      } finally {
        addingRef.current = false;
        if (addGeneration.current === generation) setPrepareLabel("");
      }
    },
    [processing, files.length]
  );

  /* ---- auth ---- */
  const doLogin = useCallback(async () => {
    setPwBusy(true);
    setAuthError("");
    const password = pwRef.current?.value || "";
    try {
      const { ok } = await api<{ ok: boolean }>("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (ok) {
        passwordRef.current = password;
        setAuthed(true);
      } else setAuthError("Incorrect password");
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
    const blobUrls = new Map<string, string>();
    const usesLeft = new Map<string, number>();
    for (const job of jobs) {
      const key = fileKey(job.file);
      usesLeft.set(key, (usesLeft.get(key) || 0) + 1);
    }
    let done = 0;

    try {
      for (const job of jobs) {
        const shouldReplace = replaceAll || !!replaceMap[job.branding.id];
        const isFirstFileForBranding = done === 0 || jobs[done - 1]?.branding.id !== job.branding.id;
        const key = fileKey(job.file);
        const large = job.file.size > DIRECT_UPLOAD_LIMIT;

        setStatusMsg(
          `[${done + 1}/${total}] ${job.branding.name} — ${job.file.name}` +
            (large ? " (large file)" : "")
        );

        const form = new FormData();
        form.append("project_name", selectedProject.name);
        form.append("branding_name", job.branding.name);
        form.append("file_type", selectedType);
        form.append("include_footer", includeFooter ? "1" : "0");
        form.append("replace_existing", shouldReplace && isFirstFileForBranding ? "1" : "0");
        form.append("whiteouts", JSON.stringify(whiteouts[key] || []));

        try {
          if (large) {
            let blobUrl = blobUrls.get(key);
            if (!blobUrl) {
              setStatusMsg(`[${done + 1}/${total}] Uploading ${job.file.name}…`);
              const { upload } = await import("@vercel/blob/client");
              const blob = await upload(job.file.name, job.file, {
                access: "public",
                handleUploadUrl: "/api/upload",
                contentType: "application/pdf",
                multipart: true,
                clientPayload: JSON.stringify({ password: passwordRef.current }),
              });
              blobUrl = blob.url;
              blobUrls.set(key, blobUrl);
            }
            const left = (usesLeft.get(key) || 1) - 1;
            usesLeft.set(key, left);
            form.append("blob_url", blobUrl);
            form.append("file_name", job.file.name);
            form.append("delete_blob", left === 0 ? "1" : "0");
            setStatusMsg(`[${done + 1}/${total}] ${job.branding.name} — ${job.file.name}`);
          } else {
            form.append("file", job.file);
          }

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
  }, [selectedProject, selectedBrandings, selectedType, files, includeFooter, replaceAll, replaceMap, whiteouts]);

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
              addGeneration.current += 1;
              const p = projects.find((x) => x.id === e.target.value) || null;
              setSelectedProject(p);
              setSelectedBrandings([]);
              setFileTypes([]);
              setSelectedType("");
              setFolderStatuses([]);
              setPrepareLabel("");
              setAddError("");
              setFiles([]);
              setWhiteouts({});
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
                <select
                  value={selectedType}
                  onChange={(e) => {
                    const next = e.target.value;
                    if (next === selectedType) return;
                    setSelectedType(next);
                  }}
                >
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
              <h2>6️⃣ Upload PDFs or images</h2>
              <div
                className={`file-drop ${files.length ? "active" : ""}`}
                onClick={() => {
                  if (!prepareLabel && !processing) fileInputRef.current?.click();
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = "copy";
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  void addIncoming(Array.from(e.dataTransfer.files));
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={FILE_ACCEPT}
                  multiple
                  onChange={(e) => {
                    void addIncoming(Array.from(e.target.files || []));
                    e.target.value = "";
                  }}
                />
                {prepareLabel ? (
                  <p>
                    <span className="spinner" /> {prepareLabel}
                  </p>
                ) : files.length === 0 ? (
                  <p>📄 Click or drag PDFs or images here (up to 50)</p>
                ) : (
                  <p>
                    ✅ {files.length} file(s) —{" "}
                    {(files.reduce((s, f) => s + f.size, 0) / 1024 / 1024).toFixed(2)} MB total.
                    Click to add more.
                  </p>
                )}
              </div>
              <p className="caption" style={{ marginTop: ".4rem" }}>
                JPG, PNG, WEBP, GIF, BMP, TIFF, HEIC, AVIF and SVG are turned into a PDF first.
                White-out, watermark and footer then work the same way on every branding.
              </p>
              {addError && (
                <div className="alert alert-error" style={{ marginTop: ".4rem" }}>
                  {addError}
                </div>
              )}
              {files.length > 0 && (
                <>
                  <p className="caption" style={{ marginTop: ".5rem" }}>
                    Need to hide something (agent name, phone, price…)? Click ✏️ White out next to
                    the file. It applies to every selected branding.
                  </p>
                  <ul className="file-list">
                    {files.map((f, i) => {
                      const key = fileKey(f);
                      const count = whiteouts[key]?.length || 0;
                      const label = f.sourceName || f.name;
                      return (
                        <li key={`${key}-${i}`} className="file-row">
                          <span className="name" title={f.sourceName ? `${f.sourceName} → ${f.name}` : f.name}>
                            {label}{" "}
                            <span className="caption">
                              ({(f.size / 1024 / 1024).toFixed(2)} MB{f.sourceName ? ", image" : ""})
                            </span>
                          </span>
                          {count > 0 && <span className="wo-badge">{count} area(s) hidden</span>}
                          <button
                            className="btn-sm btn-whiteout"
                            disabled={processing}
                            onClick={() => setEditingIndex(i)}
                          >
                            ✏️ {count ? "Edit white-out" : "White out"}
                          </button>
                          <button
                            className="btn-sm btn-outline"
                            disabled={processing}
                            aria-label={`Remove ${label}`}
                            onClick={() => {
                              const remaining = files.filter((_, j) => j !== i);
                              setFiles(remaining);
                              if (!remaining.some((r) => fileKey(r) === key)) {
                                setWhiteouts((w) => {
                                  const next = { ...w };
                                  delete next[key];
                                  return next;
                                });
                              }
                            }}
                          >
                            ✕
                          </button>
                        </li>
                      );
                    })}
                  </ul>

                  {editingIndex !== null && files[editingIndex] && (
                    <WhiteoutEditor
                      file={files[editingIndex]}
                      initial={whiteouts[fileKey(files[editingIndex])] || []}
                      onClose={() => setEditingIndex(null)}
                      onSave={(boxes) => {
                        const key = fileKey(files[editingIndex]);
                        setWhiteouts((w) => {
                          const next = { ...w };
                          if (boxes.length) next[key] = boxes;
                          else delete next[key];
                          return next;
                        });
                        setEditingIndex(null);
                      }}
                    />
                  )}

                  <button
                    className="btn btn-primary btn-block"
                    style={{ marginTop: ".5rem" }}
                    disabled={processing || !!prepareLabel || !selectedBrandings.length}
                    onClick={doProcess}
                  >
                    {processing ? (
                      <>
                        <span className="spinner" /> Processing…
                      </>
                    ) : (
                      `🚀 Process ${files.length} file(s) × ${selectedBrandings.length} branding(s)`
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
                          {!!r.whiteouts && (
                            <span className="wo-badge" style={{ marginLeft: ".4rem" }}>
                              {r.whiteouts} area(s) whited out
                            </span>
                          )}
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
