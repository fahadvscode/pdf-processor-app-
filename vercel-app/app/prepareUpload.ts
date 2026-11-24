/** Turn dropped images into PDFs so white-out, watermark and footer stay one path. */

export type UploadFile = File & { sourceName?: string };

export const FILE_ACCEPT =
  ".pdf,.jpg,.jpeg,.jpe,.jfif,.png,.gif,.webp,.bmp,.tif,.tiff,.heic,.heif,.avif,.svg,.ico,image/*";

const MAX_EDGE = 4096;
const MAX_PAGES = 100;
const MAX_PIXELS = 48_000_000;
const LOSSLESS_EXT = new Set(["png", "gif", "apng", "bmp", "ico", "cur", "svg"]);
const IMAGE_EXT = new Set([
  "jpg", "jpeg", "jpe", "jfif", "png", "apng", "gif", "webp", "bmp", "dib",
  "tif", "tiff", "heic", "heif", "avif", "svg", "ico", "cur", "jxl",
]);

type Kind = "pdf" | "raster" | "svg" | "heif" | "tiff" | "no";

const extension = (name: string) => (name.split(".").pop() || "").toLowerCase();

function pdfNameFor(original: string): string {
  const base = (original || "image").split(/[/\\]/).pop() || "image";
  const stem = base.replace(/\.[^.]+$/, "").replace(/[\u0000-\u001f]/g, "").trim() || "image";
  return `${stem.slice(0, 180)}.pdf`;
}

function readableError(err: unknown): string {
  const raw = err instanceof Error ? err.message : "";
  if (/memory|too large|exceeded|allocation/i.test(raw)) {
    return "This image is too large. Export it smaller and try again.";
  }
  if (raw && raw.length < 160 && !/failed to|notsupported|not supported/i.test(raw)) return raw;
  return "This image could not be opened";
}

function isPdfMagic(h: Uint8Array): boolean {
  return h.length >= 5 && h[0] === 0x25 && h[1] === 0x50 && h[2] === 0x44 && h[3] === 0x46 && h[4] === 0x2d;
}

function isTiffMagic(h: Uint8Array): boolean {
  return (
    (h[0] === 0x49 && h[1] === 0x49 && h[2] === 0x2a && h[3] === 0x00) ||
    (h[0] === 0x4d && h[1] === 0x4d && h[2] === 0x00 && h[3] === 0x2a)
  );
}

function isHeifMagic(h: Uint8Array): boolean {
  if (h.length < 12 || h[4] !== 0x66 || h[5] !== 0x74 || h[6] !== 0x79 || h[7] !== 0x70) return false;
  const brand = String.fromCharCode(h[8], h[9], h[10], h[11]).toLowerCase();
  return ["heic", "heix", "heif", "hevc", "mif1", "msf1"].includes(brand);
}

function isRasterMagic(h: Uint8Array): boolean {
  if (h.length < 4) return false;
  if (h[0] === 0xff && h[1] === 0xd8 && h[2] === 0xff) return true;
  if (h[0] === 0x89 && h[1] === 0x50 && h[2] === 0x4e && h[3] === 0x47) return true;
  if (h[0] === 0x47 && h[1] === 0x49 && h[2] === 0x46) return true;
  if (h[0] === 0x42 && h[1] === 0x4d) return true;
  if (h[0] === 0x00 && h[1] === 0x00 && h[2] === 0x01 && h[3] === 0x00) return true;
  return h.length >= 12 && h[0] === 0x52 && h[1] === 0x49 && h[2] === 0x46 && h[3] === 0x46 &&
    h[8] === 0x57 && h[9] === 0x45 && h[10] === 0x42 && h[11] === 0x50;
}

function isSvg(file: File, head: Uint8Array): boolean {
  const ext = extension(file.name);
  const text = new TextDecoder().decode(head).trimStart().toLowerCase();
  if (text.startsWith("<svg")) return true;
  return ext === "svg" && (text.startsWith("<?xml") || text.startsWith("<!doctype") || text.includes("<svg"));
}

async function detectKind(file: File): Promise<Kind> {
  if (!file.size) return "no";
  const head = new Uint8Array(await file.slice(0, 1024).arrayBuffer());
  if (isPdfMagic(head)) return "pdf";
  if (isTiffMagic(head)) return "tiff";
  if (isHeifMagic(head)) return "heif";
  if (isRasterMagic(head)) return "raster";
  if (isSvg(file, head)) return "svg";
  const ext = extension(file.name);
  if (ext === "heic" || ext === "heif") return "heif";
  if (ext === "tif" || ext === "tiff") return "tiff";
  if (ext === "svg" || file.type === "image/svg+xml") return "svg";
  if (IMAGE_EXT.has(ext) || file.type.startsWith("image/")) return "raster";
  return "no";
}

function drawSource(
  source: CanvasImageSource,
  sw: number,
  sh: number,
  destW?: number,
  destH?: number
): HTMLCanvasElement {
  if (!sw || !sh) throw new Error("This image has no size");
  const scale = Math.min(1, MAX_EDGE / Math.max(sw, sh));
  const w = Math.max(1, Math.min(MAX_EDGE, destW || Math.round(sw * scale)));
  const h = Math.max(1, Math.min(MAX_EDGE, destH || Math.round(sh * scale)));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not draw this image");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, w, h);
  return canvas;
}

/** EXIF orientation values 2–8. 6 and 8 are the phone rotations. */
function orientCanvas(source: HTMLCanvasElement, orientation: number): HTMLCanvasElement {
  if (orientation < 2 || orientation > 8) return source;
  const w = source.width;
  const h = source.height;
  const swap = orientation >= 5;
  const canvas = document.createElement("canvas");
  canvas.width = swap ? h : w;
  canvas.height = swap ? w : h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return source;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  switch (orientation) {
    case 2: ctx.setTransform(-1, 0, 0, 1, w, 0); break;
    case 3: ctx.setTransform(-1, 0, 0, -1, w, h); break;
    case 4: ctx.setTransform(1, 0, 0, -1, 0, h); break;
    case 5: ctx.setTransform(0, 1, 1, 0, 0, 0); break;
    case 6: ctx.setTransform(0, 1, -1, 0, h, 0); break;
    case 7: ctx.setTransform(0, -1, -1, 0, h, w); break;
    case 8: ctx.setTransform(0, -1, 1, 0, 0, w); break;
  }
  ctx.drawImage(source, 0, 0);
  return canvas;
}

function loadImageElement(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This image could not be opened"));
    };
    img.src = url;
  });
}

async function blobToCanvas(blob: Blob): Promise<HTMLCanvasElement> {
  try {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" } as ImageBitmapOptions);
    try {
      return drawSource(bmp, bmp.width, bmp.height);
    } finally {
      bmp.close();
    }
  } catch {
    const img = await loadImageElement(blob);
    return drawSource(img, img.naturalWidth, img.naturalHeight);
  }
}

async function svgToCanvas(file: File): Promise<HTMLCanvasElement> {
  const text = await file.text();
  const tag = text.match(/<svg\b[^>]*>/i)?.[0];
  if (!tag) throw new Error("This SVG could not be opened");
  const attr = (name: string) => {
    const m = tag.match(new RegExp(`\\b${name}\\s*=\\s*["']([\\d.]+)`));
    return m ? parseFloat(m[1]) : 0;
  };
  const vbMatch = tag.match(/viewBox\s*=\s*["']([^"']+)["']/i);
  const parts = vbMatch?.[1].trim().split(/[\s,]+/).map(Number) || [];
  const vbW = parts.length === 4 ? Math.abs(parts[2]) || 0 : 0;
  const vbH = parts.length === 4 ? Math.abs(parts[3]) || 0 : 0;
  const intrinsicW = attr("width") || vbW || 1200;
  const intrinsicH = attr("height") || vbH || 800;
  const viewBox = vbMatch?.[1] || `0 0 ${intrinsicW} ${intrinsicH}`;
  const scale = Math.min(MAX_EDGE / intrinsicW, MAX_EDGE / intrinsicH, intrinsicW < 1200 ? 2 : 1);
  const w = Math.max(1, Math.round(intrinsicW * scale));
  const h = Math.max(1, Math.round(intrinsicH * scale));
  // One width and height. A second copy makes browsers reject the SVG.
  const cleaned = tag
    .replace(/\swidth\s*=\s*["'][^"']*["']/gi, "")
    .replace(/\sheight\s*=\s*["'][^"']*["']/gi, "")
    .replace(/\sviewBox\s*=\s*["'][^"']*["']/gi, "");
  const open = cleaned.replace(/<svg\b/i, `<svg width="${w}" height="${h}" viewBox="${viewBox}"`);
  const img = await loadImageElement(new Blob([text.replace(tag, open)], { type: "image/svg+xml" }));
  if (!img.naturalWidth || !img.naturalHeight) throw new Error("This SVG could not be opened");
  return drawSource(img, img.naturalWidth, img.naturalHeight);
}

async function heifCanvases(file: File): Promise<HTMLCanvasElement[]> {
  const mod = await import("heic2any");
  const convert = mod.default;
  let result: Blob | Blob[];
  try {
    result = (await convert({ blob: file, toType: "image/jpeg", quality: 0.92 })) as Blob | Blob[];
  } catch {
    return [await blobToCanvas(file)];
  }
  const blobs = (Array.isArray(result) ? result : [result]).filter(Boolean);
  if (!blobs.length) throw new Error("This HEIC image could not be opened");
  if (blobs.length > MAX_PAGES) throw new Error(`This image has too many pages (limit is ${MAX_PAGES})`);
  const pages: HTMLCanvasElement[] = [];
  for (const blob of blobs) pages.push(await blobToCanvas(blob));
  return pages;
}

async function tiffCanvases(file: File): Promise<HTMLCanvasElement[]> {
  const mod = await import("utif");
  const UTIF = mod.default;
  const buf = await file.arrayBuffer();
  const ifds = UTIF.decode(buf);
  if (!ifds?.length) throw new Error("This TIFF has no pages");
  if (ifds.length > MAX_PAGES) throw new Error(`This image has too many pages (limit is ${MAX_PAGES})`);
  const pages: HTMLCanvasElement[] = [];
  for (let i = 0; i < ifds.length; i++) {
    const ifd = ifds[i];
    // Width and height are filled in by decodeImage, from the TIFF tags.
    try {
      UTIF.decodeImage(buf, ifd);
    } catch {
      throw new Error(`Couldn't read page ${i + 1} of this TIFF`);
    }
    const w = Number(ifd.width) || 0;
    const h = Number(ifd.height) || 0;
    if (!w || !h) throw new Error(`Couldn't read page ${i + 1} of this TIFF`);
    if (w * h > MAX_PIXELS) throw new Error("This image is too large. Export it smaller and try again.");
    const rgba = UTIF.toRGBA8(ifd);
    if (!rgba || rgba.length < w * h * 4) throw new Error(`Couldn't read page ${i + 1} of this TIFF`);
    const raw = document.createElement("canvas");
    raw.width = w;
    raw.height = h;
    const ctx = raw.getContext("2d");
    if (!ctx) throw new Error("Could not draw this image");
    ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba.slice(0, w * h * 4)), w, h), 0, 0);
    const tag = ifd.t274 as number | number[] | undefined;
    const orientation = Array.isArray(tag) ? Number(tag[0]) : Number(tag) || 1;
    const oriented = orientCanvas(raw, orientation);
    pages.push(drawSource(oriented, oriented.width, oriented.height));
  }
  return pages;
}

async function pagesFor(file: File, kind: Kind): Promise<HTMLCanvasElement[]> {
  if (kind === "svg") return [await svgToCanvas(file)];
  if (kind === "heif") return heifCanvases(file);
  if (kind === "tiff") return tiffCanvases(file);
  return [await blobToCanvas(file)];
}

function encodeCanvas(canvas: HTMLCanvasElement, png: boolean): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          reject(new Error("Could not encode this image"));
          return;
        }
        resolve(new Uint8Array(await blob.arrayBuffer()));
      },
      png ? "image/png" : "image/jpeg",
      png ? undefined : 0.92
    );
  });
}

async function canvasesToPdf(file: File, pages: HTMLCanvasElement[], lossless: boolean): Promise<UploadFile> {
  const { PDFDocument } = await import("pdf-lib");
  const pdf = await PDFDocument.create();
  for (const canvas of pages) {
    let bytes = await encodeCanvas(canvas, lossless);
    let image;
    try {
      image = lossless ? await pdf.embedPng(bytes) : await pdf.embedJpg(bytes);
    } catch {
      bytes = await encodeCanvas(canvas, false);
      image = await pdf.embedJpg(bytes);
    }
    const page = pdf.addPage([image.width, image.height]);
    page.drawImage(image, { x: 0, y: 0, width: image.width, height: image.height });
  }
  const saved = await pdf.save();
  const bytes = saved.buffer.slice(saved.byteOffset, saved.byteOffset + saved.byteLength) as ArrayBuffer;
  const out = new File([bytes], pdfNameFor(file.name), { type: "application/pdf" });
  return Object.assign(out, { sourceName: file.name });
}

export async function prepareUploads(
  incoming: File[],
  onProgress?: (label: string) => void,
  isCancelled?: () => boolean
): Promise<{ files: UploadFile[]; skipped: string[] }> {
  const files: UploadFile[] = [];
  const skipped: string[] = [];
  for (const file of incoming) {
    if (isCancelled?.()) return { files: [], skipped: [] };
    try {
      const kind = await detectKind(file);
      if (kind === "no") {
        skipped.push(file.size ? `${file.name} is not a PDF or a supported image` : `${file.name} is empty`);
        continue;
      }
      if (kind === "pdf") {
        files.push(file);
        continue;
      }
      onProgress?.(`Preparing ${file.name}…`);
      const lossless = kind === "svg" || LOSSLESS_EXT.has(extension(file.name));
      files.push(await canvasesToPdf(file, await pagesFor(file, kind), lossless));
    } catch (err) {
      skipped.push(`${file.name}: ${readableError(err)}`);
    }
  }
  return { files, skipped };
}
