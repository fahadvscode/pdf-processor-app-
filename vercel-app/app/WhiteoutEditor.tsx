"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";

/** Corners are fractions (0..1) of the page as displayed, origin top-left. */
export type Whiteout = { page: number; x0: number; y0: number; x1: number; y1: number };

type Props = {
  file: File;
  initial: Whiteout[];
  onSave: (boxes: Whiteout[]) => void;
  onClose: () => void;
};

type Draft = { startX: number; startY: number; x: number; y: number };

const ZOOMS = [1, 1.5, 2, 3];
const MIN_BOX_PX = 4;

let pdfjsPromise: Promise<typeof import("pdfjs-dist")> | null = null;
function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import("pdfjs-dist").then((lib) => {
      lib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.js";
      return lib;
    });
  }
  return pdfjsPromise;
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));

export default function WhiteoutEditor({ file, initial, onSave, onClose }: Props) {
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [loadError, setLoadError] = useState("");
  const [pageIndex, setPageIndex] = useState(0);
  const [zoomIndex, setZoomIndex] = useState(0);
  const [rendering, setRendering] = useState(false);
  const [pageSize, setPageSize] = useState<{ w: number; h: number } | null>(null);
  const [boxes, setBoxes] = useState<Whiteout[]>(initial);
  const [history, setHistory] = useState<Whiteout[][]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);

  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<RenderTask | null>(null);

  const numPages = doc?.numPages ?? 0;
  const pageBoxes = boxes
    .map((b, i) => ({ b, i }))
    .filter(({ b }) => b.page === pageIndex);
  const dirty = JSON.stringify(boxes) !== JSON.stringify(initial);

  /* ---- load document ---- */
  useEffect(() => {
    let cancelled = false;
    let loaded: PDFDocumentProxy | null = null;
    (async () => {
      try {
        const pdfjs = await loadPdfjs();
        const data = new Uint8Array(await file.arrayBuffer());
        loaded = await pdfjs.getDocument({ data, stopAtErrors: false }).promise;
        if (cancelled) {
          loaded.destroy();
          return;
        }
        setDoc(loaded);
      } catch (e: any) {
        if (!cancelled) setLoadError(e?.message || "Could not open this PDF");
      }
    })();
    return () => {
      cancelled = true;
      renderTaskRef.current?.cancel();
      loaded?.destroy();
    };
  }, [file]);

  /* ---- render current page ---- */
  useEffect(() => {
    if (!doc) return;
    let cancelled = false;
    (async () => {
      setRendering(true);
      try {
        renderTaskRef.current?.cancel();
        const page = await doc.getPage(pageIndex + 1);
        if (cancelled) return;
        const base = page.getViewport({ scale: 1 });
        const available = Math.max(320, (viewportRef.current?.clientWidth ?? 800) - 24);
        const cssScale = (available / base.width) * ZOOMS[zoomIndex];
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const viewport = page.getViewport({ scale: cssScale * dpr });
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        const cssW = viewport.width / dpr;
        const cssH = viewport.height / dpr;
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${cssH}px`;
        setPageSize({ w: cssW, h: cssH });
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const task = page.render({ canvasContext: ctx, viewport });
        renderTaskRef.current = task;
        await task.promise;
      } catch (e: any) {
        if (e?.name !== "RenderingCancelledException" && !cancelled) {
          console.warn("Page render failed", e);
        }
      } finally {
        if (!cancelled) setRendering(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [doc, pageIndex, zoomIndex]);

  /* ---- editing helpers ---- */
  const commit = useCallback(
    (next: Whiteout[]) => {
      setHistory((h) => [...h.slice(-49), boxes]);
      setBoxes(next);
    },
    [boxes]
  );

  const undo = useCallback(() => {
    if (!history.length) return;
    setBoxes(history[history.length - 1]);
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
      if (p === pageIndex) continue;
      for (const b of template) next.push({ ...b, page: p });
    }
    commit(next);
  };

  const goTo = (p: number) => {
    if (!numPages) return;
    setSelected(null);
    setDraft(null);
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
        if (draft) setDraft(null);
        else setSelected(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* ---- drawing ---- */
  const pointFromEvent = (e: React.PointerEvent) => {
    const rect = overlayRef.current!.getBoundingClientRect();
    return {
      x: clamp((e.clientX - rect.left) / rect.width),
      y: clamp((e.clientY - rect.top) / rect.height),
    };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !pageSize) return;
    if ((e.target as HTMLElement).closest("[data-box]")) return;
    e.preventDefault();
    overlayRef.current?.setPointerCapture(e.pointerId);
    const p = pointFromEvent(e);
    setSelected(null);
    setDraft({ startX: p.x, startY: p.y, x: p.x, y: p.y });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!draft) return;
    const p = pointFromEvent(e);
    setDraft({ ...draft, x: p.x, y: p.y });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!draft || !pageSize) return;
    overlayRef.current?.releasePointerCapture(e.pointerId);
    const x0 = Math.min(draft.startX, draft.x);
    const x1 = Math.max(draft.startX, draft.x);
    const y0 = Math.min(draft.startY, draft.y);
    const y1 = Math.max(draft.startY, draft.y);
    setDraft(null);
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
    if (dirty && !window.confirm("Discard your white-out changes for this file?")) return;
    onClose();
  };

  return (
    <div className="wo-backdrop" onMouseDown={(e) => e.target === e.currentTarget && tryClose()}>
      <div className="wo-modal" role="dialog" aria-label={`White out ${file.name}`}>
        <div className="wo-header">
          <div className="wo-title">
            <strong>✏️ White out</strong>
            <span className="caption" title={file.name}>{file.name}</span>
          </div>
          <button className="btn-sm btn-outline" onClick={tryClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="wo-toolbar">
          <button className="btn-sm btn-outline" onClick={() => goTo(pageIndex - 1)} disabled={pageIndex === 0}>
            ◀ Prev
          </button>
          <select
            className="wo-page-select"
            value={pageIndex}
            onChange={(e) => goTo(Number(e.target.value))}
            disabled={!numPages}
          >
            {Array.from({ length: numPages || 1 }, (_, p) => (
              <option key={p} value={p}>
                Page {p + 1} of {numPages || "…"}
                {countFor(p) ? ` — ${countFor(p)} box(es)` : ""}
              </option>
            ))}
          </select>
          <button
            className="btn-sm btn-outline"
            onClick={() => goTo(pageIndex + 1)}
            disabled={!numPages || pageIndex >= numPages - 1}
          >
            Next ▶
          </button>
          <span className="wo-sep" />
          <button className="btn-sm btn-outline" onClick={() => setZoomIndex((z) => Math.max(0, z - 1))} disabled={zoomIndex === 0}>
            −
          </button>
          <span className="caption">{Math.round(ZOOMS[zoomIndex] * 100)}%</span>
          <button
            className="btn-sm btn-outline"
            onClick={() => setZoomIndex((z) => Math.min(ZOOMS.length - 1, z + 1))}
            disabled={zoomIndex === ZOOMS.length - 1}
          >
            +
          </button>
          <span className="wo-sep" />
          <button className="btn-sm btn-outline" onClick={undo} disabled={!history.length}>
            ↶ Undo
          </button>
          <button className="btn-sm btn-outline" onClick={clearPage} disabled={!pageBoxes.length}>
            Clear page
          </button>
          <button
            className="btn-sm btn-outline"
            onClick={copyToAllPages}
            disabled={!pageBoxes.length || numPages < 2}
            title="Use this page's boxes on every page (replaces boxes on other pages)"
          >
            Copy to all pages
          </button>
        </div>

        <div className="wo-hint caption">
          Drag over anything you want hidden. Click a box and press Delete (or its ✕) to remove it.
          Hidden areas are permanently removed from the uploaded PDF, then the watermark is added.
        </div>

        <div className="wo-viewport" ref={viewportRef}>
          {loadError ? (
            <div className="alert alert-error">
              Couldn&apos;t preview this PDF ({loadError}). It can still be uploaded without white-out.
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
                className="wo-overlay"
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={() => setDraft(null)}
              >
                {pageBoxes.map(({ b, i }) => (
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
              {rendering && (
                <div className="wo-rendering">
                  <span className="spinner" />
                </div>
              )}
            </div>
          )}
        </div>

        <div className="wo-footer">
          <span className="caption">
            {boxes.length
              ? `${boxes.length} area(s) on ${new Set(boxes.map((b) => b.page)).size} page(s)`
              : "No areas whited out yet"}
          </span>
          <div className="wo-footer-actions">
            {boxes.length > 0 && (
              <button
                className="btn-sm btn-outline"
                onClick={() => {
                  if (window.confirm("Remove every white-out box in this file?")) {
                    commit([]);
                    setSelected(null);
                  }
                }}
              >
                Remove all
              </button>
            )}
            <button className="btn btn-outline" onClick={tryClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={() => onSave(boxes)}>
              Save white-out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
