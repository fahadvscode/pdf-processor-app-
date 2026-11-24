"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";

/** Corners are fractions (0..1) of the page as displayed, origin top-left. */
export type Whiteout = { page: number; x0: number; y0: number; x1: number; y1: number };
export type FileEdits = { boxes: Whiteout[]; deletedPages: number[] };

type Props = {
  file: File;
  initial: FileEdits;
  onSave: (edits: FileEdits) => void;
  onClose: () => void;
};

type Draft = { startX: number; startY: number; x: number; y: number };
type Snapshot = { boxes: Whiteout[]; deleted: number[] };

const ZOOMS = [1, 1.5, 2, 3];
const MIN_BOX_PX = 4;
// Phones (iOS Safari especially) refuse or crash on canvases above ~16M pixels.
const MAX_CANVAS_PIXELS = 12_000_000;
const MAX_CANVAS_SIDE = 8192;
const HISTORY_LIMIT = 50;

let pdfjsPromise: Promise<typeof import("pdfjs-dist")> | null = null;
function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import("pdfjs-dist")
      .then((lib) => {
        lib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
        return lib;
      })
      .catch((err) => {
        pdfjsPromise = null;
        throw err;
      });
  }
  return pdfjsPromise;
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));
const isCancel = (e: unknown) =>
  !!e && typeof e === "object" && (e as { name?: string }).name === "RenderingCancelledException";

function friendlyLoadError(e: unknown): string {
  const name = (e as { name?: string })?.name || "";
  if (name === "PasswordException") return "This PDF is password-protected";
  if (name === "InvalidPDFException") return "This file isn't a readable PDF";
  return (e as { message?: string })?.message || "Could not open this PDF";
}

export default function WhiteoutEditor({ file, initial, onSave, onClose }: Props) {
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [loadError, setLoadError] = useState("");
  const [renderError, setRenderError] = useState("");
  const [pageIndex, setPageIndex] = useState(0);
  const [zoomIndex, setZoomIndex] = useState(0);
  const [rendering, setRendering] = useState(false);
  const [pageSize, setPageSize] = useState<{ w: number; h: number } | null>(null);
  const [boxes, setBoxes] = useState<Whiteout[]>(initial.boxes);
  const [deleted, setDeleted] = useState<number[]>(initial.deletedPages);
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [mode, setMode] = useState<"draw" | "scroll">("draw");
  const [viewportWidth, setViewportWidth] = useState(0);

  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<RenderTask | null>(null);

  const numPages = doc?.numPages ?? 0;
  const deletedSet = new Set(deleted);
  const pageDeleted = deletedSet.has(pageIndex);
  const keptPages = numPages - deleted.filter((p) => p < numPages).length;
  const pageBoxes = boxes.map((b, i) => ({ b, i })).filter(({ b }) => b.page === pageIndex);
  const dirty =
    JSON.stringify(boxes) !== JSON.stringify(initial.boxes) ||
    JSON.stringify([...deleted].sort((a, b) => a - b)) !==
      JSON.stringify([...initial.deletedPages].sort((a, b) => a - b));

  /* ---- lock background scroll while open ---- */
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  /* ---- load document ---- */
  useEffect(() => {
    let cancelled = false;
    let loaded: PDFDocumentProxy | null = null;
    (async () => {
      try {
        const pdfjs = await loadPdfjs();
        const data = new Uint8Array(await file.arrayBuffer());
        if (cancelled) return;
        loaded = await pdfjs.getDocument({ data, stopAtErrors: false, isEvalSupported: false }).promise;
        if (cancelled) {
          loaded.destroy();
          return;
        }
        setDoc(loaded);
      } catch (e) {
        if (!cancelled) setLoadError(friendlyLoadError(e));
      }
    })();
    return () => {
      cancelled = true;
      renderTaskRef.current?.cancel();
      loaded?.destroy();
    };
  }, [file]);

  /* ---- re-render when the window is resized or the phone rotates ---- */
  useEffect(() => {
    const el = viewportRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ro = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const w = Math.round(el.clientWidth);
        setViewportWidth((prev) => (Math.abs(prev - w) > 8 ? w : prev));
      }, 150);
    });
    ro.observe(el);
    return () => {
      clearTimeout(timer);
      ro.disconnect();
    };
  }, [doc]);

  /* ---- free the canvas memory on close ---- */
  useEffect(() => {
    const canvas = canvasRef.current;
    return () => {
      if (canvas) {
        canvas.width = 0;
        canvas.height = 0;
      }
    };
  }, [doc]);

  /* ---- render current page ---- */
  useEffect(() => {
    if (!doc) return;
    let cancelled = false;
    (async () => {
      setRendering(true);
      setRenderError("");
      try {
        const page = await doc.getPage(pageIndex + 1);
        // pdf.js refuses two renders on one canvas, so wait for any earlier one to stop.
        while (renderTaskRef.current) {
          const prev = renderTaskRef.current;
          prev.cancel();
          try {
            await prev.promise;
          } catch {
            /* cancelled */
          }
          if (renderTaskRef.current === prev) renderTaskRef.current = null;
        }
        if (cancelled) return;

        const base = page.getViewport({ scale: 1 });
        const available = Math.max(200, (viewportRef.current?.clientWidth ?? 800) - 24);
        const cssScale = (available / base.width) * ZOOMS[zoomIndex];
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        let pxScale = cssScale * dpr;
        const area = base.width * base.height * pxScale * pxScale;
        if (area > MAX_CANVAS_PIXELS) pxScale *= Math.sqrt(MAX_CANVAS_PIXELS / area);
        const longest = Math.max(base.width, base.height) * pxScale;
        if (longest > MAX_CANVAS_SIDE) pxScale *= MAX_CANVAS_SIDE / longest;

        const viewport = page.getViewport({ scale: pxScale });
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = Math.max(1, Math.floor(viewport.width));
        canvas.height = Math.max(1, Math.floor(viewport.height));
        const cssW = base.width * cssScale;
        const cssH = base.height * cssScale;
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${cssH}px`;
        setPageSize({ w: cssW, h: cssH });
        const ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) throw new Error("Your browser could not draw this page");

        const task = page.render({ canvasContext: ctx, viewport });
        renderTaskRef.current = task;
        try {
          await task.promise;
        } finally {
          if (renderTaskRef.current === task) renderTaskRef.current = null;
        }
        page.cleanup();
      } catch (e) {
        if (!cancelled && !isCancel(e)) {
          console.warn("Page render failed", e);
          setRenderError("This page could not be previewed. You can still delete it or move on.");
        }
      } finally {
        if (!cancelled) setRendering(false);
      }
    })();
    return () => {
      cancelled = true;
      renderTaskRef.current?.cancel();
    };
  }, [doc, pageIndex, zoomIndex, viewportWidth]);

  /* ---- editing helpers ---- */
  const commit = useCallback(
    (nextBoxes: Whiteout[], nextDeleted: number[] = deleted) => {
      setHistory((h) => [...h.slice(-(HISTORY_LIMIT - 1)), { boxes, deleted }]);
      setBoxes(nextBoxes);
      setDeleted(nextDeleted);
    },
    [boxes, deleted]
  );

  const undo = useCallback(() => {
    if (!history.length) return;
    const last = history[history.length - 1];
    setBoxes(last.boxes);
    setDeleted(last.deleted);
    setHistory(history.slice(0, -1));
    setSelected(null);
  }, [history]);

  const removeBox = useCallback(
    (index: number) => {
      commit(boxes.filter((_, i) => i !== index));
      setSelected(null);
    },
    [boxes, commit]
  );

  const clearPage = () => {
    if (!pageBoxes.length) return;
    commit(boxes.filter((b) => b.page !== pageIndex));
    setSelected(null);
  };

  const copyToAllPages = () => {
    if (!pageBoxes.length || numPages < 2) return;
    const template = pageBoxes.map(({ b }) => b);
    const next = boxes.filter((b) => b.page === pageIndex);
    for (let p = 0; p < numPages; p++) {
      if (p === pageIndex || deletedSet.has(p)) continue;
      for (const b of template) next.push({ ...b, page: p });
    }
    commit(next);
  };

  const toggleDeletePage = () => {
    if (!numPages) return;
    if (pageDeleted) {
      commit(boxes, deleted.filter((p) => p !== pageIndex));
      return;
    }
    if (keptPages <= 1) {
      window.alert("A PDF needs at least one page. Remove the whole file from the upload list instead.");
      return;
    }
    commit(boxes, [...deleted, pageIndex].sort((a, b) => a - b));
    setSelected(null);
    updateDraft(null);
  };

  const goTo = (p: number) => {
    if (!numPages) return;
    setSelected(null);
    updateDraft(null);
    setPageIndex(Math.min(numPages - 1, Math.max(0, p)));
  };

  /* ---- keyboard ---- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
      } else if ((e.key === "Delete" || e.key === "Backspace") && selected !== null) {
        e.preventDefault();
        removeBox(selected);
      } else if (e.key === "ArrowRight" || e.key === "PageDown") {
        goTo(pageIndex + 1);
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        goTo(pageIndex - 1);
      } else if (e.key === "Escape") {
        if (draft) updateDraft(null);
        else setSelected(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* ---- drawing ---- */
  const canDraw = mode === "draw" && !pageDeleted && !!pageSize;

  const pointFromEvent = (e: React.PointerEvent) => {
    const rect = overlayRef.current!.getBoundingClientRect();
    return {
      x: clamp((e.clientX - rect.left) / rect.width),
      y: clamp((e.clientY - rect.top) / rect.height),
    };
  };

  // Pointer events can arrive faster than React re-renders, so the live drag lives in a ref.
  const draftRef = useRef<Draft | null>(null);
  const updateDraft = (next: Draft | null) => {
    draftRef.current = next;
    setDraft(next);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest("[data-box]")) return;
    setSelected(null);
    if (!canDraw || (e.pointerType === "mouse" && e.button !== 0)) return;
    e.preventDefault();
    try {
      overlayRef.current?.setPointerCapture(e.pointerId);
    } catch {
      /* some browsers reject capture for synthetic pointers */
    }
    const p = pointFromEvent(e);
    updateDraft({ startX: p.x, startY: p.y, x: p.x, y: p.y });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const current = draftRef.current;
    if (!current) return;
    const p = pointFromEvent(e);
    updateDraft({ ...current, x: p.x, y: p.y });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const current = draftRef.current;
    if (!current || !pageSize) return;
    try {
      overlayRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
    const end = pointFromEvent(e);
    const x0 = Math.min(current.startX, end.x);
    const x1 = Math.max(current.startX, end.x);
    const y0 = Math.min(current.startY, end.y);
    const y1 = Math.max(current.startY, end.y);
    updateDraft(null);
    if ((x1 - x0) * pageSize.w < MIN_BOX_PX || (y1 - y0) * pageSize.h < MIN_BOX_PX) return;
    commit([...boxes, { page: pageIndex, x0, y0, x1, y1 }]);
    setSelected(boxes.length);
  };

  const boxStyle = (b: { x0: number; y0: number; x1: number; y1: number }) => ({
    left: `${b.x0 * 100}%`,
    top: `${b.y0 * 100}%`,
    width: `${(b.x1 - b.x0) * 100}%`,
    height: `${(b.y1 - b.y0) * 100}%`,
  });

  const countFor = (p: number) => boxes.filter((b) => b.page === p).length;

  const tryClose = () => {
    if (dirty && !window.confirm("Discard your changes for this file?")) return;
    onClose();
  };

  const save = () => {
    const keep = new Set(deleted);
    onSave({
      boxes: boxes.filter((b) => !keep.has(b.page)),
      deletedPages: Array.from(keep).filter((p) => !numPages || p < numPages).sort((a, b) => a - b),
    });
  };

  const liveBoxes = boxes.filter((b) => !deletedSet.has(b.page)).length;
  const summary = [
    liveBoxes ? `${liveBoxes} area(s) hidden` : "",
    deleted.length ? `${deleted.length} page(s) removed (${deleted.map((p) => p + 1).join(", ")})` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="wo-backdrop" onMouseDown={(e) => e.target === e.currentTarget && tryClose()}>
      <div className="wo-modal" role="dialog" aria-label={`Edit ${file.name}`}>
        <div className="wo-header">
          <div className="wo-title">
            <strong>✏️ White out &amp; pages</strong>
            <span className="caption" title={file.name}>{file.name}</span>
          </div>
          <button className="btn-sm btn-outline wo-icon" onClick={tryClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="wo-toolbar">
          <div className="wo-group">
            <button className="btn-sm btn-outline" onClick={() => goTo(pageIndex - 1)} disabled={pageIndex === 0} aria-label="Previous page">
              ◀
            </button>
            <select
              className="wo-page-select"
              value={pageIndex}
              onChange={(e) => goTo(Number(e.target.value))}
              disabled={!numPages}
              aria-label="Page"
            >
              {Array.from({ length: numPages || 1 }, (_, p) => (
                <option key={p} value={p}>
                  Page {p + 1} of {numPages || "…"}
                  {deletedSet.has(p) ? " — removed" : countFor(p) ? ` — ${countFor(p)} box(es)` : ""}
                </option>
              ))}
            </select>
            <button
              className="btn-sm btn-outline"
              onClick={() => goTo(pageIndex + 1)}
              disabled={!numPages || pageIndex >= numPages - 1}
              aria-label="Next page"
            >
              ▶
            </button>
          </div>

          <div className="wo-group">
            <button className="btn-sm btn-outline" onClick={() => setZoomIndex((z) => Math.max(0, z - 1))} disabled={zoomIndex === 0} aria-label="Zoom out">
              −
            </button>
            <span className="caption wo-zoom">{Math.round(ZOOMS[zoomIndex] * 100)}%</span>
            <button
              className="btn-sm btn-outline"
              onClick={() => setZoomIndex((z) => Math.min(ZOOMS.length - 1, z + 1))}
              disabled={zoomIndex === ZOOMS.length - 1}
              aria-label="Zoom in"
            >
              +
            </button>
          </div>

          <div className="wo-group wo-mode" role="radiogroup" aria-label="Touch mode">
            <button
              className={`btn-sm ${mode === "draw" ? "btn-primary" : "btn-outline"}`}
              onClick={() => setMode("draw")}
              aria-pressed={mode === "draw"}
            >
              ✏️ Draw
            </button>
            <button
              className={`btn-sm ${mode === "scroll" ? "btn-primary" : "btn-outline"}`}
              onClick={() => {
                setMode("scroll");
                updateDraft(null);
              }}
              aria-pressed={mode === "scroll"}
            >
              ✋ Scroll
            </button>
          </div>

          <div className="wo-group">
            <button className="btn-sm btn-outline" onClick={undo} disabled={!history.length}>
              ↶ Undo
            </button>
            <button className="btn-sm btn-outline" onClick={clearPage} disabled={!pageBoxes.length || pageDeleted}>
              Clear<span className="wo-long"> page</span>
            </button>
            <button
              className="btn-sm btn-outline"
              onClick={copyToAllPages}
              disabled={!pageBoxes.length || numPages < 2 || pageDeleted}
              title="Use this page's boxes on every page (replaces boxes on other pages)"
            >
              Copy<span className="wo-long"> to</span> all<span className="wo-long"> pages</span>
            </button>
            <button
              className={`btn-sm ${pageDeleted ? "btn-outline" : "btn-danger-outline"}`}
              onClick={toggleDeletePage}
              disabled={!numPages}
            >
              {pageDeleted ? "↩ Restore" : "🗑 Delete"}
              <span className="wo-long"> page</span>
            </button>
          </div>
        </div>

        <div className="wo-hint caption">
          {mode === "draw" ? (
            <>
              <span className="wo-long">Drag over anything you want hidden. Tap a box, then ✕ to remove it. Switch to ✋ Scroll to move around the page.</span>
              <span className="wo-short">Drag to hide · tap a box, then ✕ to remove · ✋ Scroll to move</span>
            </>
          ) : (
            "Scroll mode: move around freely. Switch back to ✏️ Draw to add boxes."
          )}
        </div>

        <div className="wo-viewport" ref={viewportRef}>
          {loadError ? (
            <div className="alert alert-error wo-message">
              Couldn&apos;t preview this PDF ({loadError}). It can still be uploaded without changes.
            </div>
          ) : !doc ? (
            <p className="wo-loading">
              <span className="spinner" /> Loading PDF…
            </p>
          ) : (
            <div className="wo-page" style={pageSize ? { width: pageSize.w, height: pageSize.h } : undefined}>
              <canvas ref={canvasRef} />
              <div
                ref={overlayRef}
                className={`wo-overlay ${canDraw ? "" : "wo-overlay-scroll"}`}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={() => updateDraft(null)}
              >
                {!pageDeleted &&
                  pageBoxes.map(({ b, i }) => (
                    <div
                      key={i}
                      data-box
                      className={`wo-box ${selected === i ? "selected" : ""}`}
                      style={boxStyle(b)}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        setSelected(i);
                      }}
                    >
                      <button
                        className="wo-box-x"
                        aria-label="Remove box"
                        onPointerDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          removeBox(i);
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                {draft && (
                  <div
                    className="wo-box draft"
                    style={boxStyle({
                      x0: Math.min(draft.startX, draft.x),
                      y0: Math.min(draft.startY, draft.y),
                      x1: Math.max(draft.startX, draft.x),
                      y1: Math.max(draft.startY, draft.y),
                    })}
                  />
                )}
              </div>
              {pageDeleted && (
                <div className="wo-deleted">
                  <strong>🗑 Page {pageIndex + 1} will be removed</strong>
                  <button className="btn-sm btn-outline" onClick={toggleDeletePage}>
                    ↩ Restore page
                  </button>
                </div>
              )}
              {renderError && <div className="wo-render-error alert alert-warn">{renderError}</div>}
              {rendering && (
                <div className="wo-rendering">
                  <span className="spinner" />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="wo-footer">
          <span className="caption">{summary || "No changes yet"}</span>
          <div className="wo-footer-actions">
            {(boxes.length > 0 || deleted.length > 0) && (
              <button
                className="btn-sm btn-outline"
                onClick={() => {
                  if (window.confirm("Remove every white-out box and restore all pages?")) {
                    commit([], []);
                    setSelected(null);
                  }
                }}
              >
                Reset
              </button>
            )}
            <button className="btn btn-outline" onClick={tryClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={save}>
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
